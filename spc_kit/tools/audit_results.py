"""Verify the complete SPC training matrix and final evidence before paper use."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, low=0.0, high=1.0):
    value = float(value)
    require(math.isfinite(value) and low <= value <= high, f"Invalid {label}: {value}")
    return value


def audit(root, phase):
    matrices = [root / "spc_kit/slurm" / f"matrix_{group}.txt" for group in ("main", "ablation")]
    configs = [line.strip() for path in matrices for line in path.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    require(len(configs) == len(set(configs)) == 12, "Expected 12 unique matrix configurations")
    expected_runs = set(itertools.product(configs, range(1, 6)))
    actual_dirs = {p.relative_to(root / "runs/spc").as_posix() for p in (root / "runs/spc").glob("*/seed_*/train")}
    expected_dirs = {f"{cfg}/seed_{seed}/train" for cfg, seed in expected_runs}
    require(actual_dirs == expected_dirs, f"Training directory mismatch: missing={expected_dirs-actual_dirs}, extra={actual_dirs-expected_dirs}")
    commits = set()
    for cfg, seed in sorted(expected_runs):
        run = root / "runs/spc" / cfg / f"seed_{seed}/train"
        opt = yaml.safe_load((run / "opt.yaml").read_text())
        require(opt["seed"] == seed and opt["epochs"] == 100 and opt["batch_size"] == 8, f"Protocol mismatch: {run}")
        require(opt["img_size"] == [640, 640], f"Image-size mismatch: {run}")
        require(Path(opt["cfg"]).stem == cfg, f"Model mismatch: {run}")
        epochs = [int(line.split()[0].split("/")[0]) for line in (run / "results.txt").read_text().splitlines() if line.strip()]
        require(epochs == list(range(100)), f"Expected epochs 0..99 exactly once: {run}")
        for name in ("best.pt", "last.pt"):
            p = run / "weights" / name
            require(p.is_file() and p.stat().st_size > 0, f"Missing checkpoint: {p}")
        commit = (run / "source_commit.txt").read_text().strip()
        require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), f"Invalid source provenance: {run}")
        commits.add(commit)
    report = {"status": "passed", "phase": phase, "runs": len(expected_runs), "epochs_per_run": 100,
              "source_commits": sorted(commits)}
    if phase == "training":
        return report
    counts = json.loads((root / "data/spc_subsets/subsets.json").read_text())["counts"]
    expected_subsets = {f"{dataset}_{subset}" for dataset in ("m3fd", "flir") for subset in ("all", "day", "night")}
    require(set(counts) == expected_subsets, f"Unexpected evaluation subsets: {set(counts)}")
    rows = list(csv.DictReader((root / "results/eval.csv").open()))
    keys = [(r["config"], int(r["seed"]), r["subset"]) for r in rows]
    expected_keys = {(cfg, seed, sub) for cfg, seed in expected_runs for sub in expected_subsets}
    require(len(keys) == len(set(keys)) and set(keys) == expected_keys,
            f"Incomplete or duplicate evaluation coverage: {len(keys)} rows; expected {len(expected_keys)}")
    effective = {}
    for sub in sorted(expected_subsets):
        subset_rows = [r for r in rows if r["subset"] == sub]
        images = {int(r["images"]) for r in subset_rows}
        labels = {int(r["labels"]) for r in subset_rows}
        require(len(images) == len(labels) == 1, f"Different samples evaluated across models: {sub}")
        n = next(iter(images))
        require(0 < n <= counts[sub], f"Invalid sample count: {sub}: {n}")
        effective[sub] = {"nominal_images": counts[sub], "evaluated_images": n, "labels": next(iter(labels))}
    for row in rows:
        for metric in ("P", "R", "mAP50", "mAP75", "mAP50_95"):
            number(row[metric], f"{row['config']}/{row['seed']}/{row['subset']}/{metric}")
        require(float(row["mAP50_95"]) <= float(row["mAP50"]) + .002, "AP50:95 exceeds AP50")
        require(float(row["mAP75"]) <= float(row["mAP50"]) + .002, "AP75 exceeds AP50")
        for name, value in row.items():
            if name.startswith("AP50_") and value:
                number(value, name)
    flops = json.loads((root / "results/flops_640.json").read_text())
    require(set(flops) == set(configs), "Profiling configuration coverage mismatch")
    for cfg, values in flops.items():
        for name in ("params_M", "GFLOPs"):
            number(values[name], f"{cfg}/{name}", 1e-9, float("inf"))
        number(values["latency_ms"]["mean"], f"{cfg}/latency", 1e-9, float("inf"))
    files = [root / "results" / name for name in ("eval.csv", "flops_640.json", "tables.md")]
    for seed in range(1, 6):
        p = root / "results" / f"alpha_seed{seed}.json"
        alpha = json.loads(p.read_text())
        for sub in ("m3fd_day", "m3fd_night", "flir_day", "flir_night"):
            values = alpha[sub]
            require(values["n"] == counts[sub] == len(values["raw"]), f"Incomplete alpha coverage: {seed}/{sub}")
            require(len(values["per_scale"]) == 3, f"Missing alpha scales: {seed}/{sub}")
            for sample in values["raw"]:
                require(len(sample) == 3, f"Missing per-image alpha scales: {seed}/{sub}")
                for value in sample:
                    number(value, f"alpha/{seed}/{sub}")
        files.append(p)
    tables = (root / "results/tables.md").read_text()
    require(all(f"## {sub}\n" in tables for sub in expected_subsets), "Missing subset tables")
    require(all(f"| {cfg} |" in tables for cfg in configs), "Missing model table rows")
    report.update(evaluation_rows=len(rows), subsets=effective,
                  sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("training", "all"), default="all")
    args = parser.parse_args()
    report = audit(args.root.resolve(), args.phase)
    output = args.root / "results" / f"audit_{args.phase}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
