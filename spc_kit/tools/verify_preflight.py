"""Audit real splits, transferred tensors and CUDA forwards before SPC training."""
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys

import torch
import yaml

root = Path.cwd()
engine = root / "cft_engine"
sys.path.insert(0, str(engine))
from models.yolo_test import Model
logging.disable(logging.CRITICAL)
assert torch.cuda.is_available(), "CUDA unavailable inside the Slurm allocation"
source = (engine / "train.py").read_text()
assert source.count("init_seeds(opt.seed + max(rank, 0))") == 2
assert "init_seeds(2 + rank)" not in source
counts, stems = {}, {}
for split in ("train", "sel", "test", "flir_test"):
    paths = []
    for modality in ("rgb", "ir"):
        p = root / "data/spc_split" / f"{split}_{modality}.txt"
        rows = [Path(line) for line in p.read_text().splitlines() if line.strip()]
        assert rows and len(set(rows)) == len(rows), str(p)
        assert all(row.is_file() for row in rows), f"missing image in {p}"
        assert all(Path(str(row).replace("/images_ir/", "/labels/").replace("/images/", "/labels/")).with_suffix(".txt").is_file() for row in rows), f"missing labels in {p}"
        paths.append(rows)
    assert [p.stem for p in paths[0]] == [p.stem for p in paths[1]], split
    counts[split] = len(paths[0])
    stems[split] = {p.stem for p in paths[0]}
assert counts["test"] == 840, counts
assert counts["sel"] == 336, counts
for left in stems:
    for right in stems:
        if left < right:
            assert stems[left].isdisjoint(stems[right]), f"overlap: {left}/{right}"
print("split counts", counts, flush=True)
report = json.loads((root / "weights/spc_init/init_report.json").read_text())
results = {}
configs = sorted((engine / "models/spc").glob("*.yaml"))
assert len(configs) == 12, len(configs)
for cfg in configs:
    # train_rgb_ir constructs with ch=6; verify that actual entrypoint contract.
    model = Model(str(cfg), ch=6, nc=6)
    checkpoint = torch.load(root / "weights/spc_init" / f"{cfg.stem}.pt", map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    target = model.state_dict()
    assert all(k in target and v.shape == target[k].shape for k, v in state.items()), cfg.stem
    assert len(state) == report[cfg.stem]["transferred"], cfg.stem
    assert all(int(k.split(".")[1]) in checkpoint["init_map"] for k in state), cfg.stem
    assert not any(k.startswith(f"model.{len(model.model)-1}.") for k in state), "Detect must stay freshly initialized"
    model.load_state_dict(state, strict=False)
    model = model.eval().cuda()
    with torch.no_grad():
        output = model(torch.zeros(1,3,640,640,device="cuda"), torch.zeros(1,3,640,640,device="cuda"))[0]
    assert output.shape[-1] == 11 and torch.isfinite(output).all(), cfg.stem
    results[cfg.stem] = {"output_shape": list(output.shape), "transferred_tensors": len(state)}
    print(cfg.stem, results[cfg.stem], flush=True)
    del model, output, checkpoint, state, target
    gc.collect()
    torch.cuda.empty_cache()
files = [*list((root / "spc_kit").rglob("*.py")), *list((root / "spc_kit/slurm").glob("*.sh")),
         *list((root / "data/spc_split").glob("*.txt")), root / "data/hyp.scratch.yaml", root / "weights/yolov5l_v5.0.pt"]
manifest = {}
for p in files:
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    manifest[str(p.relative_to(root))] = digest.hexdigest()
result = {"status": "passed", "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
          "counts": counts, "models": results, "sha256": manifest,
          "dataset_note": "Existing aligned FLIR subset; do not call this the full official ADAS v2 test set without verifying provenance."}
(root / "results/spc_preflight.json").write_text(json.dumps(result, indent=2)+"\n")
print("SPC PREFLIGHT PASSED", flush=True)
