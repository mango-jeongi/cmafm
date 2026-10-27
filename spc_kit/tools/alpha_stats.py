"""EXP-9: statistics of the learnable fusion gate alpha (weight on the RGB stream) over whole subsets.

For each image the mean alpha of each scale (P3/P4/P5) is recorded through a forward hook on every
`fusion_gate` module (works for best.pt from the WACV code and for CMAFMVar). Reports mean, std and
quartiles per subset and scale, and a Mann-Whitney U test day vs night.
Usage: python tools/alpha_stats.py --engine cft_engine --weights runs/spc/cmafm_full/seed_1/.../best.pt \
           --subsets data/spc_subsets --names m3fd_day,m3fd_night,flir_day,flir_night --out results/alpha.json
"""
import argparse, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="cft_engine")
ap.add_argument("--weights", required=True)
ap.add_argument("--subsets", default="data/spc_subsets")
ap.add_argument("--names", default="m3fd_day,m3fd_night,flir_day,flir_night")
ap.add_argument("--out", default="results/alpha.json")
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
sys.path.insert(0, str(Path(a.engine).resolve()))
import cv2, numpy as np, torch
from scipy import stats
from models.experimental import attempt_load
from utils.datasets import letterbox

dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
model = attempt_load(a.weights, map_location=dev).eval()
gates = [m for n, m in model.named_modules() if n.endswith("fusion_gate")]
assert gates, "no fusion_gate found (alpha exists only in the full CMAFM)"
cur = []
for g in gates:
    g.register_forward_hook(lambda mod, i, o: cur.append(float(o.float().mean())))

def load(p):
    im = letterbox(cv2.imread(p), 640, auto=False)[0][:, :, ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(im)).float().div(255).unsqueeze(0).to(dev)

res = {}
for name in a.names.split(","):
    sp = Path(a.subsets)
    rgb = [l.strip() for l in open(sp / f"{name}_rgb.txt") if l.strip()]
    ir = [l.strip() for l in open(sp / f"{name}_ir.txt") if l.strip()]
    per = []
    with torch.no_grad():
        for r, i in zip(rgb, ir):
            cur.clear(); model(load(r), load(i)); per.append(list(cur))
    arr = np.array(per)  # images x scales
    res[name] = {"n": len(per), "per_scale": [{"mean": float(c.mean()), "std": float(c.std()),
                 "q25": float(np.percentile(c, 25)), "median": float(np.median(c)), "q75": float(np.percentile(c, 75))}
                 for c in arr.T], "raw": arr.tolist()}
    print(name, len(per), [round(s["mean"], 3) for s in res[name]["per_scale"]])
for ds in ("m3fd", "flir"):
    if f"{ds}_day" in res and f"{ds}_night" in res:
        d, n = np.array(res[f"{ds}_day"]["raw"]), np.array(res[f"{ds}_night"]["raw"])
        res[f"{ds}_day_vs_night_mannwhitney_p"] = [float(stats.mannwhitneyu(d[:, k], n[:, k]).pvalue) for k in range(d.shape[1])]
        print(ds, "day vs night p per scale:", res[f"{ds}_day_vs_night_mannwhitney_p"])
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
