"""Performance analysis script by day/night conditions."""

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from dataset import build_datasets, collate_fn
from evaluate import evaluate_model
from model import build_model

CHECKPOINT = str(Path(__file__).parent / "runs" / "best.pth")
BRIGHTNESS_THRESHOLD = 60  # Average RGB brightness boundary (< 60 = night)
BATCH_SIZE = 2


def classify_by_brightness(dataset, vis_dir: Path, threshold: float):
    """Classify dataset samples into night and day scenes based on brightness."""
    night_idx, day_idx = [], []
    for i, img_id in enumerate(dataset.samples):
        img_path = vis_dir / f"{img_id}.png"
        arr = np.array(Image.open(img_path).convert("L"))
        brightness = arr.mean()
        if brightness < threshold:
            night_idx.append(i)
        else:
            day_idx.append(i)
    return night_idx, day_idx


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT}\n")

    # Load dataset (using validation set only)
    _, val_dataset = build_datasets(cfg.data)
    vis_dir = Path(__file__).parent / "data" / "M3FD" / "Vis"

    print("Analysing image brightness...")
    night_idx, day_idx = classify_by_brightness(val_dataset, vis_dir, BRIGHTNESS_THRESHOLD)
    print(f"  Total Validation: {len(val_dataset)} images")
    print(f"  Night (brightness < {BRIGHTNESS_THRESHOLD}): {len(night_idx)} images")
    print(f"  Day (brightness >= {BRIGHTNESS_THRESHOLD}): {len(day_idx)} images\n")

    # Load model
    model = build_model(cfg.model, num_classes=cfg.data.num_classes)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print("Model loaded successfully\n")

    results = {}

    # Complete validation evaluation
    print("=" * 50)
    print("[1/3] Evaluating Complete Validation Set")
    print("=" * 50)
    all_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    results["all"] = evaluate_model(model, all_loader, device, cfg.data)
    r = results["all"]
    print(f"  mAP@0.5:      {r['mAP50']:.4f}")
    print(f"  mAP@[.5:.95]: {r['mAP50_95']:.4f}")
    print(f"  Precision:    {r['precision']:.4f}")
    print(f"  Recall:       {r['recall']:.4f}")
    print(f"  F1-Score:     {r['f1']:.4f}")
    print(f"  Miss Rate:    {r['miss_rate']:.4f}")
    print(f"  FPS:          {r['fps']:.1f}\n")

    # Night evaluation
    if len(night_idx) > 0:
        print("=" * 50)
        print(f"[2/3] Evaluating Night Subset ({len(night_idx)} images)")
        print("=" * 50)
        night_loader = DataLoader(Subset(val_dataset, night_idx), batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0, collate_fn=collate_fn)
        results["night"] = evaluate_model(model, night_loader, device, cfg.data)
        r = results["night"]
        print(f"  mAP@0.5:      {r['mAP50']:.4f}")
        print(f"  mAP@[.5:.95]: {r['mAP50_95']:.4f}")
        print(f"  Precision:    {r['precision']:.4f}")
        print(f"  Recall:       {r['recall']:.4f}")
        print(f"  F1-Score:     {r['f1']:.4f}")
        print(f"  Miss Rate:    {r['miss_rate']:.4f}\n")

    # Day evaluation
    if len(day_idx) > 0:
        print("=" * 50)
        print(f"[3/3] Evaluating Day Subset ({len(day_idx)} images)")
        print("=" * 50)
        day_loader = DataLoader(Subset(val_dataset, day_idx), batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=0, collate_fn=collate_fn)
        results["day"] = evaluate_model(model, day_loader, device, cfg.data)
        r = results["day"]
        print(f"  mAP@0.5:      {r['mAP50']:.4f}")
        print(f"  mAP@[.5:.95]: {r['mAP50_95']:.4f}")
        print(f"  Precision:    {r['precision']:.4f}")
        print(f"  Recall:       {r['recall']:.4f}")
        print(f"  F1-Score:     {r['f1']:.4f}")
        print(f"  Miss Rate:    {r['miss_rate']:.4f}\n")

    # Final Summary
    print("\n" + "=" * 70)
    print("Final Summary")
    print("=" * 70)
    classes = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
    print(f"{'Condition':<10} {'mAP@0.5':>9} {'mAP@.5:.95':>12} {'Precision':>11} {'Recall':>8} {'F1':>8} {'MissRate':>10}")
    print("-" * 70)
    for cond, label in [("all", "All"), ("night", "Night"), ("day", "Day")]:
        if cond in results:
            r = results[cond]
            print(f"{label:<10} {r['mAP50']:>9.4f} {r['mAP50_95']:>12.4f} "
                  f"{r['precision']:>11.4f} {r['recall']:>8.4f} "
                  f"{r['f1']:>8.4f} {r['miss_rate']:>10.4f}")

    print("\nAP@0.5 / Recall / Miss Rate by Class (Night vs Day)")
    print("-" * 75)
    print(f"{'Class':<14}", end="")
    for cond, label in [("all", "All"), ("night", "Night"), ("day", "Day")]:
        if cond in results:
            print(f"{label+' AP':>10}{label+' Rec':>10}{label+' MR':>9}", end="")
    print()
    for cls in classes:
        print(f"{cls:<14}", end="")
        for cond in ["all", "night", "day"]:
            if cond in results:
                ap  = results[cond].get(f"AP50_{cls}", -1)
                rec = results[cond].get(f"recall_{cls}", -1)
                mr  = results[cond].get(f"miss_rate_{cls}", -1)
                ap_s  = f"{ap:.4f}"  if ap  >= 0 else "  —  "
                rec_s = f"{rec:.4f}" if rec >= 0 else "  —  "
                mr_s  = f"{mr:.4f}"  if mr  >= 0 else "  —  "
                print(f"{ap_s:>10}{rec_s:>10}{mr_s:>9}", end="")
        print()

    # Save results
    out_path = Path(__file__).parent / "runs" / "night_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "brightness_threshold": BRIGHTNESS_THRESHOLD,
            "night_count": len(night_idx),
            "day_count": len(day_idx),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
