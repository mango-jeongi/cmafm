"""Run ablation study: train all variants and collect results."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import build_datasets, collate_fn
from evaluate import evaluate_model, evaluate_late_fusion
from ablation_models import build_ablation_model


VARIANTS = ["rgb_only", "thermal_only", "early_fusion", "dual_no_attn",
            "simplified_cmafm", "late_fusion", "full"]


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1.0
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0

    lr_warmup = None
    if epoch == 0:
        warmup_iters = min(500, len(data_loader) - 1)
        lr_warmup = warmup_lr_scheduler(optimizer, warmup_iters, 0.001)

    for i, (rgb, thermal, targets) in enumerate(data_loader):
        rgb = rgb.to(device)
        thermal = thermal.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if not all(t["boxes"].shape[0] > 0 for t in targets):
            continue

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(rgb, thermal, targets)
            losses = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(losses):
            continue

        optimizer.zero_grad()
        if scaler:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        if lr_warmup:
            lr_warmup.step()

        total_loss += losses.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def run_variant(variant, cfg, train_loader, val_loader, device, epochs=20):
    """Train and evaluate one ablation variant."""
    print(f"\n{'='*60}")
    print(f"  ABLATION: {variant}")
    print(f"{'='*60}\n")

    model = build_ablation_model(variant, cfg.model, cfg.data.num_classes)
    model.to(device)

    # Optimizer
    if variant in ["dual_no_attn", "full"]:
        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "rgb_stages" in name or "thermal_stages" in name:
                backbone_params.append(param)
            elif "layer1" in name or "layer2" in name or "layer3" in name or "layer4" in name or "stem" in name or "conv1" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.SGD([
            {"params": backbone_params, "lr": cfg.train.lr * 0.1},
            {"params": other_params, "lr": cfg.train.lr},
        ], momentum=0.9, weight_decay=5e-4)
    else:
        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if any(k in name for k in ["layer1", "layer2", "layer3", "layer4", "stem", "conv1"]):
                backbone_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.SGD([
            {"params": backbone_params, "lr": cfg.train.lr * 0.1},
            {"params": other_params, "lr": cfg.train.lr},
        ], momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

    best_map = 0.0
    best_metrics = {}
    save_dir = Path(cfg.train.save_dir) / "ablation"
    save_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
        scheduler.step()

        # Evaluate every 5 epochs + last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            metrics = evaluate_model(model, val_loader, device, cfg.data)
            mAP50 = metrics["mAP50"]
            mAP50_95 = metrics["mAP50_95"]
            print(f"  [{variant}] Epoch {epoch}: loss={avg_loss:.4f}  "
                  f"mAP@0.5={mAP50:.4f}  mAP@[.5:.95]={mAP50_95:.4f}")

            if mAP50 > best_map:
                best_map = mAP50
                best_metrics = metrics
                torch.save(model.state_dict(), save_dir / f"{variant}_best.pth")
        else:
            print(f"  [{variant}] Epoch {epoch}: loss={avg_loss:.4f}")

    elapsed = time.time() - t_start
    best_metrics["training_time"] = elapsed
    best_metrics["variant"] = variant

    print(f"\n  [{variant}] BEST: mAP@0.5={best_metrics.get('mAP50', 0):.4f}  "
          f"mAP@[.5:.95]={best_metrics.get('mAP50_95', 0):.4f}  "
          f"time={elapsed:.0f}s")

    # Cleanup GPU
    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=VARIANTS,
                        help="Which variants to run")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs per variant (default 30)")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = Config()
    cfg.train.batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build datasets (shared across all variants)
    train_ds, val_ds = build_datasets(cfg.data)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    results = {}
    for variant in args.variants:
        # Skip "full" if best.pth already exists (use existing result)
        if variant == "full":
            best_pth = Path(cfg.train.save_dir) / "best.pth"
            if best_pth.exists():
                print(f"\n{'='*60}")
                print(f"  ABLATION: full (using existing checkpoint)")
                print(f"{'='*60}")
                from model import build_model
                model = build_model(cfg.model, cfg.data.num_classes)
                ckpt = torch.load(best_pth, map_location=device)
                model.load_state_dict(ckpt["model"])
                model.to(device)
                metrics = evaluate_model(model, val_loader, device, cfg.data)
                metrics["variant"] = "full"
                metrics["training_time"] = 0
                results["full"] = metrics
                print(f"  [full] mAP@0.5={metrics['mAP50']:.4f}  "
                      f"mAP@[.5:.95]={metrics['mAP50_95']:.4f}")
                del model
                torch.cuda.empty_cache()
                continue

        # Handle "late_fusion" ensembling
        if variant == "late_fusion":
            print(f"\n{'='*60}")
            print(f"  ABLATION: late_fusion (ensembling rgb_only and thermal_only)")
            print(f"{'='*60}")
            ablation_dir = Path(cfg.train.save_dir) / "ablation"
            rgb_path = ablation_dir / "rgb_only_best.pth"
            thermal_path = ablation_dir / "thermal_only_best.pth"

            if not rgb_path.exists() or not thermal_path.exists():
                print("WARNING: rgb_only_best.pth or thermal_only_best.pth not found. "
                      "Make sure both 'rgb_only' and 'thermal_only' variants are trained first. "
                      "Skipping late_fusion.")
                continue

            model_rgb = build_ablation_model("rgb_only", cfg.model, cfg.data.num_classes)
            model_rgb.load_state_dict(torch.load(rgb_path, map_location=device))
            model_rgb.to(device)

            model_thermal = build_ablation_model("thermal_only", cfg.model, cfg.data.num_classes)
            model_thermal.load_state_dict(torch.load(thermal_path, map_location=device))
            model_thermal.to(device)

            metrics = evaluate_late_fusion(model_rgb, model_thermal, val_loader, device, cfg.data)
            metrics["variant"] = "late_fusion"
            metrics["training_time"] = 0
            results["late_fusion"] = metrics
            print(f"  [late_fusion] mAP@0.5={metrics['mAP50']:.4f}  "
                  f"mAP@[.5:.95]={metrics['mAP50_95']:.4f}")

            del model_rgb, model_thermal
            torch.cuda.empty_cache()
            continue

        results[variant] = run_variant(variant, cfg, train_loader, val_loader,
                                        device, epochs=args.epochs)

    # Save results
    save_path = Path(cfg.train.save_dir) / "ablation" / "results.json"
    # Convert non-serializable values
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: float(vv) if isinstance(vv, (int, float)) else vv
                           for kk, vv in v.items()}
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*70}")
    print(f"  ABLATION STUDY RESULTS")
    print(f"{'='*70}")
    print(f"{'Variant':<20} {'mAP@0.5':>10} {'mAP@[.5:.95]':>14} {'Time(min)':>10}")
    print(f"{'-'*54}")
    for variant in VARIANTS:
        if variant in results:
            r = results[variant]
            t = r.get('training_time', 0) / 60
            print(f"{variant:<20} {r.get('mAP50', 0):>10.4f} "
                  f"{r.get('mAP50_95', 0):>14.4f} {t:>10.1f}")
    print(f"{'='*70}")
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
