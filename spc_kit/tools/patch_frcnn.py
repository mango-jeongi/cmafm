"""Bring the Faster R-CNN family onto the SPC protocol (src/fusion/).

1. dataset.py: if CMAFM_TEST_STEMS is set, the M3FD split comes from data/spc_split
   (same 840 test images as the YOLO family; sel stems removed from training), instead of a
   numpy RandomState shuffle that produced a different 840-image set than the YOLO pipeline.
2. run_ablation.py: --seed (set for every variant), --retrain-full (the full model is trained by the
   same loop instead of reusing a best.pth from train.py), StepLR uses cfg.train.lr_step_size (10, the
   value used for the full model, instead of 8), and the reported metrics are the LAST epoch's,
   not the best of the test-set evaluations (which selected on the test set).
Usage (from repo root): python spc_kit/tools/patch_frcnn.py && \
  CMAFM_TEST_STEMS=data/spc_split/m3fd_test_stems.txt CMAFM_SEL_STEMS=data/spc_split/m3fd_sel_stems.txt \
  python src/fusion/run_ablation.py --seed 1 --retrain-full --variants rgb_only thermal_only early_fusion dual_no_attn full
"""
from pathlib import Path

def edit(path, pairs):
    p = Path(path); s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        if new in s:
            continue
        assert old in s, f"{path}: pattern not found:\n{old}"
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print("patched", path)

edit("src/fusion/dataset.py", [(
"""    n_val = int(len(all_ids) * cfg.val_ratio)
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]
""",
"""    n_val = int(len(all_ids) * cfg.val_ratio)
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    import os
    if os.environ.get("CMAFM_TEST_STEMS"):  # SPC protocol: same split as the YOLO family
        test = set(Path(os.environ["CMAFM_TEST_STEMS"]).read_text().split())
        sel = set(Path(os.environ["CMAFM_SEL_STEMS"]).read_text().split()) if os.environ.get("CMAFM_SEL_STEMS") else set()
        val_ids = sorted(i for i in all_ids if i in test)
        train_ids = sorted(i for i in all_ids if i not in test and i not in sel)
""")])

edit("src/fusion/run_ablation.py", [
("    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)",
 "    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.lr_step_size, gamma=cfg.train.lr_gamma)"),
("""            if mAP50 > best_map:
                best_map = mAP50
                best_metrics = metrics
                torch.save(model.state_dict(), save_dir / f"{variant}_best.pth")""",
"""            if epoch == epochs - 1:  # SPC protocol: report the last epoch, no selection on the test set
                best_metrics = metrics
                torch.save(model.state_dict(), save_dir / f"{variant}_best.pth")"""),
("""    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
""",
"""    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--retrain-full", action="store_true")
    args = parser.parse_args()
    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
"""),
("""    cfg.train.batch_size = args.batch_size
""",
"""    cfg.train.batch_size = args.batch_size
    cfg.train.save_dir = str(Path(cfg.train.save_dir) / f"seed_{args.seed}")
"""),
("""        if variant == "full":
            best_pth""",
"""        if variant == "full" and not args.retrain_full:
            best_pth"""),
("    save_path = Path(cfg.train.save_dir) / \"ablation\" / \"results.json\"",
 "    save_path = Path(cfg.train.save_dir) / \"ablation\" / f\"results_seed{args.seed}.json\""),
])
