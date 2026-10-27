"""Give every config the same COCO YOLOv5l initialization (fixes the init mismatch between CMAFM and CFT).

The old CMAFM run started from weights/CMAFM_Pretrained.pt (hand-written layer map), while the CFT
baseline was started from yolov5l.pt directly, where train.py's name-based intersect_dicts only
matches the first few layers of stream one. Here every model gets the same mapping rule:
each backbone layer of yolov5l.pt may initialize one layer per stream, each head
layer at most one, matched in network order by module type and exact tensor shapes. Fusion
modules (GPT, CMAFM*, controls) and Detect stay randomly initialized (per --seed) for every model.

Usage: python tools/init_weights.py --engine cft_engine --src weights/yolov5l.pt --out weights/spc_init
writes weights/spc_init/<config>.pt plus init_report.json (tensors transferred / total, per config).
"""
import argparse, contextlib, io, json, logging, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="cft_engine")
ap.add_argument("--src", default="weights/yolov5l.pt")
ap.add_argument("--out", default="weights/spc_init")
a = ap.parse_args()
eng = Path(a.engine).resolve()
sys.path.insert(0, str(eng))
logging.disable(logging.CRITICAL)
import torch
import models.yolo_test as yt

N_BACKBONE = 10  # yolov5l.yaml layers 0-9 are the backbone

src = torch.load(a.src, map_location="cpu", weights_only=False)["model"].float()
src_layers = list(src.model)
assert len(src_layers) == 25 and type(src_layers[8]).__name__ == "SPP", (
    "Expected the YOLOv5 v5.0 YOLOv5l checkpoint with SPP, not a later SPPF model"
)

def sig(layer):
    return type(layer).__name__, tuple((k, tuple(v.shape)) for k, v in layer.state_dict().items())

src_sig = [sig(l) for l in src_layers]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
report = {}
for cfg in sorted((eng / "models" / "spc").glob("*.yaml")):
    with contextlib.redirect_stdout(io.StringIO()):
        m = yt.Model(str(cfg), ch=3, nc=6)
    n_streams = sum(type(l).__name__ == "Focus" for l in m.model)
    used = [0] * len(src_layers)
    moved = total = 0
    mapping = {}
    for i, layer in enumerate(m.model):
        sd = layer.state_dict()
        total += len(sd)
        if not sd or type(layer).__name__ == "Detect":
            continue
        s = sig(layer)
        for j, ss in enumerate(src_sig):
            cap = n_streams if j < N_BACKBONE else 1
            if ss == s and used[j] < cap:
                layer.load_state_dict(src_layers[j].state_dict())
                used[j] += 1; moved += len(sd); mapping[i] = j
                break
    # save only the transferred tensors (plain state dict, which train.py accepts), so fusion modules and
    # Detect are freshly initialized under each run's --seed instead of sharing one random draw
    keep = {k: v for k, v in m.state_dict().items() if k.split(".")[1].isdigit() and int(k.split(".")[1]) in mapping}
    torch.save({"model": keep, "optimizer": None, "epoch": -1, "init_map": mapping}, out / f"{cfg.stem}.pt")
    report[cfg.stem] = {"transferred": moved, "total": total, "ratio": round(moved / total, 4), "map": mapping}
    print(f"{cfg.stem:22s} {moved}/{total} tensors ({100 * moved / total:.1f}%)  layers {len(mapping)}")
json.dump(report, open(out / "init_report.json", "w"), indent=1)
