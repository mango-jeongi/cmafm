"""Evaluate every trained run on every fixed subset with identical test settings.

Run layout (from slurm/run_matrix.sh): runs/<config>/seed_<k>/weights/best.pt  (best.pt is chosen on
the sel split, never on the test set). Each (run, subset) is evaluated with cft_engine/test.py
(conf 0.001, NMS IoU 0.6, 640, batch 16) and the "all" row plus per-class AP@0.5 are parsed.
Output: results/eval.csv with one row per (config, seed, subset). Already evaluated rows are skipped.
"""
import argparse, csv, re, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="cft_engine")
ap.add_argument("--runs", default="runs/spc")
ap.add_argument("--subsets", default="data/spc_subsets")
ap.add_argument("--out", default="results/eval.csv")
ap.add_argument("--weights", default="best.pt", help="best.pt (selected on sel) or last.pt")
ap.add_argument("--only", default="", help="comma-separated subset names (default: all)")
ap.add_argument("--device", default="0")
ap.add_argument("--expected-runs", type=int, default=0)
a = ap.parse_args()
eng = Path(a.engine).resolve()
subsets = sorted(Path(a.subsets).resolve().glob("*.yaml"))
if a.only:
    keep = set(a.only.split(","))
    subsets = [s for s in subsets if s.stem in keep]
outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
FIELDS = ["config", "seed", "subset", "images", "labels", "P", "R", "mAP50", "mAP75", "mAP50_95",
          "AP50_People", "AP50_Car", "AP50_Bus", "AP50_Motorcycle", "AP50_Lamp", "AP50_Truck"]
done = set()
if outp.exists():
    done = {(r["config"], r["seed"], r["subset"]) for r in csv.DictReader(open(outp))}
new = not outp.exists()
f = open(outp, "a", newline=""); w = csv.DictWriter(f, FIELDS)
if new:
    w.writeheader()
ROW = re.compile(r"^\s*(\S+)\s+(\d+)\s+(\d+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*$")
weights = sorted(Path(a.runs).resolve().glob(f"*/seed_*/**/weights/{a.weights}"))
assert weights and subsets, "No checkpoints or subset YAMLs found"
if a.expected_runs:
    assert len(weights) == a.expected_runs, f"Expected {a.expected_runs} checkpoints, found {len(weights)}"
for wpath in weights:
    cfg = wpath.relative_to(Path(a.runs).resolve()).parts[0]
    seed = re.search(r"seed_(\d+)", str(wpath)).group(1)
    for sub in subsets:
        if (cfg, seed, sub.stem) in done:
            continue
        cmd = [sys.executable, "test.py", "--weights", str(wpath), "--data", str(sub), "--img-size", "640",
               "--batch-size", "16", "--conf-thres", "0.001", "--iou-thres", "0.6", "--task", "val", "--verbose",
               "--device", a.device, "--project", str(Path(a.out).resolve().parent / "test_runs"),
               "--name", f"{cfg}_s{seed}_{sub.stem}", "--exist-ok"]
        proc = subprocess.run(cmd, cwd=eng, capture_output=True, text=True)
        txt = proc.stdout
        log = outp.parent / "eval_logs" / f"{cfg}_s{seed}_{sub.stem}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(txt + "\nSTDERR:\n" + proc.stderr)
        if proc.returncode:
            raise RuntimeError(f"Evaluation failed: {cfg} seed {seed} {sub.stem}; see {log}")
        rows = {m.group(1): m.groups() for m in map(ROW.match, txt.splitlines()) if m}
        if "all" not in rows:
            raise RuntimeError(f"No all-class metrics: {cfg} seed {seed} {sub.stem}; see {log}")
        _, n, nl, p, r, m50, m75, m = rows["all"]
        rec = dict(config=cfg, seed=seed, subset=sub.stem, images=n, labels=nl, P=p, R=r, mAP50=m50, mAP75=m75, mAP50_95=m)
        for c in ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]:
            rec[f"AP50_{c}"] = rows[c][5] if c in rows else ""
        w.writerow(rec); f.flush()
        print(cfg, seed, sub.stem, m50, m)
