"""EXP-6: one canonical split for every model and both detector families.

Input: the list files written by scripts/prepare_all.py (data/merged_txt/*_unified.txt).
Output (in --out):
  train_{rgb,ir}.txt      M3FD train minus the selection subset + all FLIR train
  sel_{rgb,ir}.txt        10% of M3FD train (seeded), used ONLY for best.pt selection
  test_{rgb,ir}.txt       the unchanged 840-image M3FD val split, reported as the test set
  flir_test_{rgb,ir}.txt  FLIR official test split (in-domain, FLIR train is in training)
  m3fd_test_stems.txt     stems of the M3FD test set, for the Faster R-CNN pipeline
  m3fd_sel_stems.txt      stems of the selection subset
  spc_train.yaml / spc_test.yaml / spc_flir_test.yaml   data configs for train.py / test.py
RGB and IR lists are paired after sorting (as utils/datasets.py does) and checked stem by stem.
"""
import argparse, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--merged", default="data/merged_txt", help="dir with *_unified.txt and *_flir_only.txt")
ap.add_argument("--out", default="data/spc_split")
ap.add_argument("--sel-ratio", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
md, out = Path(a.merged), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
NAMES = ['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck']

def read(p):
    return sorted(l.strip() for l in (md / p).read_text(encoding="utf-8").splitlines() if l.strip())

def pair(rgb, ir):
    assert len(rgb) == len(ir), (len(rgb), len(ir))
    for r, i in zip(rgb, ir):
        assert Path(r).stem == Path(i).stem, (r, i)
    return list(zip(rgb, ir))

def is_flir(p):
    return "flir" in p.lower()

train = pair(read("train_rgb_unified.txt"), read("train_ir_unified.txt"))
test = pair(read("val_rgb_unified.txt"), read("val_ir_unified.txt"))
assert not any(is_flir(r) for r, _ in test), "val_*_unified should be M3FD-only"
m3fd_train = [t for t in train if not is_flir(t[0])]
flir_train = [t for t in train if is_flir(t[0])]
rng = random.Random(a.seed)
stems = sorted({Path(r).stem for r, _ in m3fd_train})
rng.shuffle(stems)
sel_stems = set(stems[: round(len(stems) * a.sel_ratio)])
sel = [t for t in m3fd_train if Path(t[0]).stem in sel_stems]
tr = [t for t in m3fd_train if Path(t[0]).stem not in sel_stems] + flir_train
assert not ({Path(r).stem for r, _ in test} & {Path(r).stem for r, _ in m3fd_train}), "train/test overlap"

def write(name, pairs):
    (out / f"{name}_rgb.txt").write_text("\n".join(r for r, _ in pairs) + "\n", encoding="utf-8")
    (out / f"{name}_ir.txt").write_text("\n".join(i for _, i in pairs) + "\n", encoding="utf-8")

write("train", tr); write("sel", sel); write("test", test)
flir_test = None
if (md / "val_rgb_flir_only.txt").exists():
    flir_test = pair(read("val_rgb_flir_only.txt"), read("val_ir_flir_only.txt"))
    write("flir_test", flir_test)
(out / "m3fd_test_stems.txt").write_text("\n".join(sorted(Path(r).stem for r, _ in test)) + "\n")
(out / "m3fd_sel_stems.txt").write_text("\n".join(sorted(sel_stems)) + "\n")

o = out.resolve().as_posix()
def yaml(name, tr_name, va_name):
    (out / name).write_text(f"train_rgb: {o}/{tr_name}_rgb.txt\ntrain_ir: {o}/{tr_name}_ir.txt\n"
                            f"val_rgb: {o}/{va_name}_rgb.txt\nval_ir: {o}/{va_name}_ir.txt\nnc: 6\nnames: {NAMES}\n")
yaml("spc_train.yaml", "train", "sel")   # per-epoch validation / best.pt selection on sel only
yaml("spc_test.yaml", "train", "test")
if flir_test:
    yaml("spc_flir_test.yaml", "train", "flir_test")
print(f"train {len(tr)} (M3FD {len(m3fd_train) - len(sel)} + FLIR {len(flir_train)}) | sel {len(sel)} | "
      f"M3FD test {len(test)} | FLIR test {len(flir_test) if flir_test else 0}")
