import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path

from config import Config
from dataset import build_datasets, collate_fn
from evaluate import evaluate_model
from generate_report_plots import build_ablation_model

from dotenv import load_dotenv

def plot_curves():
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=repo_root / ".env")
    
    data_dir_env = os.environ.get("CMAFM_DATA_DIR", "../data")
    data_dir_path = Path(data_dir_env)
    if not data_dir_path.is_absolute():
        data_dir_path = (repo_root / data_dir_path).resolve()
    data_root = data_dir_path / "M3FD"
    
    runs_dir_env = os.environ.get("CMAFM_RUNS_DIR", "./runs")
    runs_dir_path = Path(runs_dir_env)
    if not runs_dir_path.is_absolute():
        runs_dir_path = (repo_root / runs_dir_path).resolve()
    save_dir = runs_dir_path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.data.root = str(data_root)
    
    _, val_dataset = build_datasets(cfg.data)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    ckpt_path = save_dir / "ablation/full_best.pth"
    print("Loading Full CMAFM model...")
    model = build_ablation_model("full", cfg.model, cfg.data.num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    
    print("Running evaluation on Full CMAFM...")
    evaluate_model(model, val_loader, device, cfg.data)
    
    import glob
    import json
    import tempfile
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    tmpdir = tempfile.gettempdir()
    files = glob.glob(f'{tmpdir}/*.json')
    files.sort(key=os.path.getmtime)
    
    with open(files[-2]) as f:
        d1 = json.load(f)
        
    if isinstance(d1, dict) and 'annotations' in d1:
        gt_path, pred_path = files[-2], files[-1]
    else:
        gt_path, pred_path = files[-1], files[-2]

    print("Generating precision-recall curves...")
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    p = coco_eval.eval["precision"][0, :, :, 0, -1] 
    r = coco_eval.params.recThrs
    
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    class_names = [c['name'] for c in categories]

    figures_dir_env = os.environ.get("CMAFM_FIGURES_DIR", "../figures")
    figures_dir_path = Path(figures_dir_env)
    if not figures_dir_path.is_absolute():
        figures_dir_path = (repo_root / figures_dir_path).resolve()

    os.makedirs(figures_dir_path, exist_ok=True)

    # 1. Plot PR Curve
    plt.figure(figsize=(10, 8))
    mean_p = np.zeros_like(r)
    counts = np.zeros_like(r)
    
    for i in range(p.shape[1]):
        valid_p = p[:, i]
        mask = valid_p > -1
        if np.any(mask):
            plt.plot(r[mask], valid_p[mask], label=class_names[i], linewidth=2, alpha=0.7)
            mean_p[mask] += valid_p[mask]
            counts[mask] += 1
            
    valid_mean = counts > 0
    if np.any(valid_mean):
        mean_p[valid_mean] /= counts[valid_mean]
        plt.plot(r[valid_mean], mean_p[valid_mean], label='Mean mAP@0.5', linewidth=4, color='black', linestyle='--')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (CMAFM Faster R-CNN @ IoU=0.50)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_dir_path / 'ablation_PR_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Plot F1 Curve
    plt.figure(figsize=(10, 8))
    for i in range(p.shape[1]):
        valid_p = p[:, i]
        mask = valid_p > -1
        if np.any(mask):
            f1 = 2 * (valid_p[mask] * r[mask]) / (valid_p[mask] + r[mask] + 1e-16)
            plt.plot(r[mask], f1, label=class_names[i], linewidth=2, alpha=0.7)
            
    if np.any(valid_mean):
        f1_mean = 2 * (mean_p[valid_mean] * r[valid_mean]) / (mean_p[valid_mean] + r[valid_mean] + 1e-16)
        plt.plot(r[valid_mean], f1_mean, label='Mean F1', linewidth=4, color='black', linestyle='--')
        
    plt.xlabel('Confidence Threshold (Recall Proxy)')
    plt.ylabel('F1 Score')
    plt.title('F1 Curve (CMAFM Faster R-CNN @ IoU=0.50)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_dir_path / 'ablation_F1_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Successfully generated figures/ablation_PR_curve.png and figures/ablation_F1_curve.png")

if __name__ == "__main__":
    plot_curves()
