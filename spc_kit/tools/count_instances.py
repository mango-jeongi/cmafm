"""Count images and per-class object instances across SPC dataset splits and subsets."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CLASS_NAMES = ("People", "Car", "Bus", "Motorcycle", "Lamp", "Truck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=Path("cft_engine"))
    parser.add_argument("--split", type=Path, default=Path("data/spc_split"))
    parser.add_argument("--subsets", type=Path, default=Path("data/spc_subsets"))
    parser.add_argument("--out", type=Path, default=Path("results/dataset_stats.md"))
    return parser.parse_args()


def read_image_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inspect_split(img_paths: list[str], img2label_paths_fn) -> dict:
    """Count nominal and effective (duplicate-label-filtered) images and per-class instances."""
    lbl_paths = img2label_paths_fn(img_paths)
    nom_counts: Counter[int] = Counter()
    eff_counts: Counter[int] = Counter()
    excluded_dups: list[str] = []
    missing_labels: list[str] = []
    empty_labels: list[str] = []
    eff_images = 0

    for img_p, lbl_p in zip(img_paths, lbl_paths):
        lp = Path(lbl_p)
        if not lp.is_file():
            missing_labels.append(Path(img_p).stem)
            continue
        raw_lines = [line.split() for line in lp.read_text(encoding="utf-8").strip().splitlines() if line.strip()]
        if not raw_lines:
            empty_labels.append(Path(img_p).stem)
            eff_images += 1
            continue
        arr = np.array(raw_lines, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        classes = arr[:, 0].astype(int).tolist()
        for cls_id in classes:
            nom_counts[cls_id] += 1

        has_duplicate = np.unique(arr, axis=0).shape[0] != arr.shape[0]
        if has_duplicate:
            excluded_dups.append(Path(img_p).stem)
            continue

        eff_images += 1
        for cls_id in classes:
            eff_counts[cls_id] += 1

    return {
        "nominal_images": len(img_paths),
        "effective_images": eff_images,
        "nominal_instances": sum(nom_counts.values()),
        "effective_instances": sum(eff_counts.values()),
        "nominal_by_class": nom_counts,
        "effective_by_class": eff_counts,
        "excluded_duplicates": excluded_dups,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
    }


def format_row(name: str, stats: dict, mode: str) -> str:
    imgs = stats[f"{mode}_images"]
    inst = stats[f"{mode}_instances"]
    by_cls = stats[f"{mode}_by_class"]
    cls_cols = " | ".join(str(by_cls.get(i, 0)) for i in range(len(CLASS_NAMES)))
    return f"| {name} | {imgs} | {inst} | {cls_cols} |"


def build_markdown(rows: list[tuple[str, dict]]) -> str:
    header = "| Split / Subset | Images | Total Instances | " + " | ".join(CLASS_NAMES) + " |"
    sep = "|---|---:|---:|" + "|".join("---:" for _ in CLASS_NAMES) + "|"

    lines = [
        "# SPC Dataset Statistics",
        "",
        "## Effective Loader Counts (Duplicate-Label Images Excluded)",
        "",
        "Matches `cft_engine/utils/datasets.py` (`LoadImagesAndLabels.cache_labels`), which drops images containing exact duplicate label rows.",
        "",
        header,
        sep,
    ]
    for name, stats in rows:
        lines.append(format_row(name, stats, "effective"))

    lines.extend([
        "",
        "## Nominal Split Counts (All Listed Pairs)",
        "",
        header,
        sep,
    ])
    for name, stats in rows:
        lines.append(format_row(name, stats, "nominal"))

    dup_notes = [(name, stats["excluded_duplicates"]) for name, stats in rows if stats["excluded_duplicates"]]
    if dup_notes:
        lines.extend(["", "## Excluded Duplicate-Label Stems", ""])
        for name, stems in dup_notes:
            lines.append(f"- **{name}** ({len(stems)}): {', '.join(stems)}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.engine.resolve()))
    from utils.datasets import img2label_paths

    split_dir = args.split
    subsets_dir = args.subsets

    train_imgs = read_image_list(split_dir / "train_rgb.txt")
    train_m3fd = [p for p in train_imgs if "flir" not in p.lower()]
    train_flir = [p for p in train_imgs if "flir" in p.lower()]
    sel_imgs = read_image_list(split_dir / "sel_rgb.txt")
    test_imgs = read_image_list(split_dir / "test_rgb.txt")
    flir_test_imgs = read_image_list(split_dir / "flir_test_rgb.txt")

    groups: list[tuple[str, list[str]]] = [
        ("train_all", train_imgs),
        ("train_m3fd", train_m3fd),
        ("train_flir", train_flir),
        ("sel_m3fd", sel_imgs),
        ("test_m3fd", test_imgs),
        ("test_flir", flir_test_imgs),
    ]

    for sub_name in ("m3fd_all", "m3fd_day", "m3fd_night", "flir_all", "flir_day", "flir_night"):
        sub_imgs = read_image_list(subsets_dir / f"{sub_name}_rgb.txt")
        if sub_imgs:
            groups.append((sub_name, sub_imgs))

    results: list[tuple[str, dict]] = []
    for name, imgs in groups:
        if imgs:
            results.append((name, inspect_split(imgs, img2label_paths)))

    md = build_markdown(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
