"""EXP-3: params and FLOPs of every config, counted directly at 640x640 with both inputs.

Usage: python tools/profile_models.py --engine cft_engine --out results/flops_640.json [--latency]
FLOPs use torch.utils.flop_counter (counts matmul/attention correctly); thop at 640 is reported as a
cross-check. The old YOLOv5 practice (thop at 32x32 then x400) is also reported, because it is what
produced the 13,682.5 GFLOPs figure for CFT: the 8x8-token attention does not scale with image area.
--latency (CUDA only): FP16, batch 1, 50 warmup + 300 timed runs, cudnn.benchmark on.
"""
import argparse, contextlib, io, json, logging, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="cft_engine")
ap.add_argument("--out", default="results/flops_640.json")
ap.add_argument("--img", type=int, default=640)
ap.add_argument("--latency", action="store_true")
a = ap.parse_args()
eng = Path(a.engine).resolve()
sys.path.insert(0, str(eng))
logging.disable(logging.CRITICAL)
import torch
from torch.utils.flop_counter import FlopCounterMode
import thop
import models.yolo_test as yt

FUSION = ("GPT", "CMAFM_Fusion", "CMAFMVar")

def build(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return yt.Model(str(cfg), ch=3, nc=6).eval()

def flops(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()

cfgs = sorted((eng / "models" / "spc").glob("*.yaml"))
res = {}
S = a.img
for cfg in cfgs:
    name = cfg.stem
    m = build(cfg)
    x = (torch.zeros(1, 3, S, S), torch.zeros(1, 3, S, S))
    with torch.no_grad():
        total = flops(lambda: m(*x))
        macs, _ = thop.profile(build(cfg), inputs=x, verbose=False)
        macs32, _ = thop.profile(build(cfg), inputs=(torch.zeros(1, 3, 32, 32),) * 2, verbose=False)
    feats = []
    fm = [l for l in m.model if type(l).__name__ in FUSION]
    hs = [l.register_forward_hook(lambda mod, i, o: feats.append((mod, [t.clone() for t in i[0]]))) for l in fm]
    with torch.no_grad():
        m(*x)
    for h in hs:
        h.remove()
    per = []
    for mod, inp in feats:
        with torch.no_grad():
            per.append({"hw": list(inp[0].shape[2:]), "C": inp[0].shape[1],
                        "GFLOPs": round(flops(lambda: mod(inp)) / 1e9, 2),
                        "params_M": round(sum(p.numel() for p in mod.parameters()) / 1e6, 2)})
    r = {"params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
         "GFLOPs": round(total / 1e9, 1),
         "GFLOPs_thop640": round(2 * macs / 1e9, 1),
         "GFLOPs_old_thop32x400": round(2 * macs32 / 1e9 * (S / 32) ** 2, 1),
         "fusion_GFLOPs": round(sum(p["GFLOPs"] for p in per), 2),
         "fusion_params_M": round(sum(p["params_M"] for p in per), 2),
         "fusion_per_scale": per}
    if a.latency and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        mg = m.cuda().half()
        # Detect.grid is an ordinary list, so Module.cuda()/half() do not move it.
        for layer in mg.modules():
            if type(layer).__name__ == "Detect":
                layer.grid = [g.to(device="cuda", dtype=torch.float16) for g in layer.grid]
                layer.stride = layer.stride.to(device="cuda", dtype=torch.float16)
        xg = tuple(t.cuda().half() for t in x)
        with torch.no_grad():
            for _ in range(50):
                mg(*xg)
            torch.cuda.synchronize()
            ts = []
            for _ in range(300):
                t0 = time.perf_counter(); mg(*xg); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        r["latency_ms"] = {"mean": round(sum(ts) / len(ts), 2), "median": round(ts[len(ts) // 2], 2),
                           "p95": round(ts[int(0.95 * len(ts))], 2), "fps": round(1e3 / (sum(ts) / len(ts)), 1),
                           "gpu": torch.cuda.get_device_name(0)}
    res[name] = r
    print(f"{name:22s} params {r['params_M']:7.2f} M  GFLOPs {r['GFLOPs']:7.1f}  fusion {r['fusion_GFLOPs']:6.2f}  "
          f"old-method {r['GFLOPs_old_thop32x400']:9.1f}  {r.get('latency_ms', '')}", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
