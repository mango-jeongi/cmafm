"""Turn results/eval.csv into the paper tables: mean +- std over seeds, Welch t-test vs a reference
config, and per-subset gains of each fusion model over the baselines on the SAME subset
(this replaces the WACV "night vs day" comparison, which compared two subsets of one model).

Usage: python tools/aggregate.py --eval results/eval.csv --flops results/flops_640.json --ref cmafm_full
Writes results/tables.md.
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--eval", default="results/eval.csv")
ap.add_argument("--flops", default="results/flops_640.json")
ap.add_argument("--ref", default="cmafm_full")
ap.add_argument("--out", default="results/tables.md")
a = ap.parse_args()
rows = list(csv.DictReader(open(a.eval)))
fl = json.load(open(a.flops)) if Path(a.flops).exists() else {}
V = defaultdict(lambda: defaultdict(list))  # V[subset][config] -> list of (mAP50, mAP50_95)
for r in rows:
    V[r["subset"]][r["config"]].append((100 * float(r["mAP50"]), 100 * float(r["mAP50_95"])))

def ms(xs):
    return f"{mean(xs):.2f} ± {stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"

def welch(x, y):
    if len(x) < 2 or len(y) < 2:
        return "n/a"
    t, p = stats.ttest_ind(x, y, equal_var=False)
    return f"{p:.3g}"

order = ["rgb_only", "thermal_only", "dual_add", "ctrl_concat_1x1", "ctrl_se_self", "ctrl_cbam_self", "cft",
         "cmafm_channel_only", "cmafm_spatial_only", "cmafm_no_alpha", "cmafm_v_sigmoid", "cmafm_full"]
L = []
for sub in sorted(V):
    L += [f"\n## {sub}\n", "| config | n | mAP@0.5 | mAP@[.5:.95] | Δ mAP@0.5 vs ref | p (Welch) | params (M) | GFLOPs | fusion GFLOPs | latency (ms) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    ref50 = [x[0] for x in V[sub].get(a.ref, [])]
    for c in [c for c in order if c in V[sub]] + sorted(set(V[sub]) - set(order)):
        x50 = [x[0] for x in V[sub][c]]; x95 = [x[1] for x in V[sub][c]]
        d = f"{mean(x50) - mean(ref50):+.2f}" if ref50 and c != a.ref else "—"
        f = fl.get(c, {})
        lat = f.get("latency_ms", {}).get("mean", "")
        L.append(f"| {c} | {len(x50)} | {ms(x50)} | {ms(x95)} | {d} | {welch(x50, ref50) if c != a.ref else '—'} | "
                 f"{f.get('params_M', '')} | {f.get('GFLOPs', '')} | {f.get('fusion_GFLOPs', '')} | {lat} |")
# same-subset fusion gains (EXP-5)
L += ["\n## Fusion gain on the same subset (mAP@0.5, mean over seeds)\n",
      "| subset | CMAFM | − RGB-only | − Thermal-only | − Dual+Add | − Dual+Concat | − CFT |", "|---|---|---|---|---|---|---|"]
for sub in sorted(V):
    if a.ref not in V[sub]:
        continue
    r = mean(x[0] for x in V[sub][a.ref])
    cell = lambda c: f"{r - mean(x[0] for x in V[sub][c]):+.2f}" if c in V[sub] else ""
    L.append(f"| {sub} | {r:.2f} | {cell('rgb_only')} | {cell('thermal_only')} | {cell('dual_add')} | {cell('ctrl_concat_1x1')} | {cell('cft')} |")
Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
