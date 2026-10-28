"""Finalize fully trained SPC checkpoints separately from training worker memory.

Original unstripped files are backed up outside runs/spc. Replacements are atomic
and require identical learned tensors plus a successful CPU inference check.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_training(run):
    opt = yaml.safe_load((run / "opt.yaml").read_text())
    require(opt["epochs"] == 100 and opt["batch_size"] == 8 and opt["img_size"] == [640, 640], f"Protocol mismatch: {run}")
    require(opt["seed"] == int(run.parent.name.split("_")[1]), f"Seed mismatch: {run}")
    require(Path(opt["cfg"]).stem == run.parent.parent.name, f"Config mismatch: {run}")
    rows = [line.split() for line in (run / "results.txt").read_text().splitlines() if line.strip()]
    require([int(row[0].split("/")[0]) for row in rows] == list(range(100)), f"Training incomplete: {run}")
    require(all(len(row) == 16 and all(math.isfinite(float(x)) for x in row[2:]) for row in rows), f"Nonfinite or malformed training metrics: {run}")
    return rows


def inference_check(model, torch):
    model = model.float().eval()
    require(model.model[-1].nc == 6, "Checkpoint has the wrong class count")
    with torch.no_grad():
        prediction = model(torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 64, 64))[0]
    require(prediction.shape == (1, 252, 11) and torch.isfinite(prediction).all().item(), "CPU checkpoint inference failed")


def finalize_one(path, archive, rows, torch):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), f"Invalid checkpoint: {path}")
    epoch = checkpoint.get("epoch")
    require(isinstance(epoch, int) and -1 <= epoch < 100, f"Invalid checkpoint epoch: {path}")
    if epoch >= 0:
        require(path.name != "last.pt" or epoch == 99, f"Last checkpoint is not epoch 99: {path}")
        stored = checkpoint.get("training_results", "")
        require(len([line for line in stored.splitlines() if line.strip()]) == epoch + 1, f"Checkpoint training history mismatch: {path}")
        fitness = checkpoint.get("best_fitness")
        require(fitness is not None, f"Missing selection fitness: {path}")
        fitness = float(fitness.item() if hasattr(fitness, "item") else fitness)
        expected = max(.1 * float(row[10]) + .9 * float(row[12]) for row in rows[:epoch + 1])
        require(abs(fitness - expected) <= .0002, f"Selection fitness mismatch: {path}: {fitness} vs {expected}")
    selected = checkpoint.get("ema")
    if selected is None:
        selected = checkpoint.get("model")
    require(isinstance(selected, torch.nn.Module), f"Missing model: {path}")
    learned_hash = tensor_hash(selected)
    original_hash = file_hash(path)
    recovered = epoch >= 0 or checkpoint.get("optimizer") is not None
    if recovered:
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            require(file_hash(archive) == original_hash, f"Backup collision: {archive}")
        else:
            shutil.copy2(path, archive)
            require(file_hash(archive) == original_hash, f"Backup copy mismatch: {archive}")
        checkpoint["model"] = selected.half()
        for key in ("optimizer", "training_results", "wandb_id", "ema", "updates"):
            checkpoint[key] = None
        checkpoint["epoch"] = -1
        for parameter in checkpoint["model"].parameters():
            parameter.requires_grad = False
        temporary = path.with_name(f".{path.name}.finalizing.{os.getpid()}.pt")
        try:
            torch.save(checkpoint, temporary)
            candidate = torch.load(temporary, map_location="cpu", weights_only=False)
            require(tensor_hash(candidate["model"]) == learned_hash, f"Learned tensors changed: {path}")
            inference_check(candidate["model"], torch)
            del candidate
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        inference_check(selected, torch)
    result = {"checkpoint": str(path), "original_epoch": epoch,
              "status": "recovered" if recovered else "validated_unchanged",
              "learned_tensors_sha256": learned_hash, "original_file_sha256": original_hash,
              "final_file_sha256": file_hash(path)}
    if recovered:
        result["original_backup"] = str(archive)
    del checkpoint, selected
    gc.collect()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Optional single config/seed_k, otherwise validate the entire matrix")
    parser.add_argument("--out", default="results/checkpoint_finalization.json")
    args = parser.parse_args()
    root = Path.cwd()
    sys.path.insert(0, str(root / "cft_engine"))
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    if args.run:
        relative = Path(args.run)
        require(len(relative.parts) == 2 and relative.parts[0] not in (".", "..") and relative.parts[1].startswith("seed_"), "Use config/seed_k")
        runs = [root / "runs/spc" / relative / "train"]
    else:
        from audit_results import audit
        audit(root, "training")
        runs = sorted((root / "runs/spc").glob("*/seed_*/train"))
    records = []
    for run in runs:
        rows = validate_training(run)
        for name in ("best.pt", "last.pt"):
            source = run / "weights" / name
            archive = root / "runs/spc_recovery_originals" / run.relative_to(root / "runs/spc") / "weights" / name
            record = finalize_one(source, archive, rows, torch)
            records.append(record)
            print(record["status"], source, flush=True)
    output = root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "passed", "runs": len(runs), "checkpoints": records}, indent=2) + "\n")
    print("CHECKPOINT FINALIZATION PASSED", flush=True)
