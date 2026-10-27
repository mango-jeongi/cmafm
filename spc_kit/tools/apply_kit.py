"""Install the SPC experiment kit into a patched cft_engine (run after scripts/setup.sh).

1. copies models/cmafm_variants.py into cft_engine/models/
2. registers CMAFMVar in models/yolo_test.py parse_model (same channel rule as CMAFM_Fusion)
3. fixes train.py so --seed is actually used (it called init_seeds(2 + rank) and ignored --seed)
4. writes all model YAMLs (tools/make_configs.py)
Idempotent: running it twice changes nothing.
"""
import argparse, shutil, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="cft_engine")
ap.add_argument("--overlay", default="src/engine/cft_engine_patches")
a = ap.parse_args()
kit = Path(__file__).resolve().parent.parent
eng = Path(a.engine)

shutil.copy(kit / "models" / "cmafm_variants.py", eng / "models" / "cmafm_variants.py")

yt = eng / "models" / "yolo_test.py"
s = yt.read_text(encoding="utf-8")
if "from .cmafm_variants import CMAFMVar" not in s:
    s = "from .cmafm_variants import CMAFMVar\n" + s
    old = "        elif m is CMAFM_Fusion:\n            c2 = ch[f[0]]\n            args = [c2]\n"
    new = old + "        elif m is CMAFMVar:\n            c2 = ch[f[0]]\n            args = [c2, *args[1:]]\n"
    assert s.count(old) >= 1, "parse_model CMAFM_Fusion branch not found; check yolo_test.py"
    s = s.replace(old, new)
    yt.write_text(s, encoding="utf-8")

tr = eng / "train.py"
s = tr.read_text(encoding="utf-8")
n = s.count("init_seeds(2 + rank)")
if n:
    s = s.replace("init_seeds(2 + rank)", "init_seeds(opt.seed + max(rank, 0))")
    tr.write_text(s, encoding="utf-8")
s = s.replace("init_seeds(opt.seed + rank)", "init_seeds(opt.seed + max(rank, 0))")
assert "init_seeds(opt.seed + max(rank, 0))" in s, "seed initialization not found"
# test.py returns P, R, AP50, AP75, AP50:95, then three losses.
# The legacy four-metric fitness function must receive AP50:95, not AP75.
s = s.replace("fitness(np.array(results).reshape(1, -1))",
              "fitness(np.array(results).reshape(1, -1)[:, [0, 1, 2, 4]])")
assert s.count("fitness(np.array(results).reshape(1, -1)[:, [0, 1, 2, 4]])") == 2

tr.write_text(s, encoding="utf-8")
print(f"train.py: replaced {n} init_seeds call(s)")
# Both modality caches may discard invalid samples; never silently mispair them.
ds = eng / "utils/datasets.py"
d = ds.read_text(encoding="utf-8")
marker = "        self.img_files_ir = list(cache_ir.keys())  # update"
check = """
        if [Path(p).stem for p in self.img_files_rgb] != [Path(p).stem for p in self.img_files_ir]:
            raise ValueError("SPC RGB/IR cache filtering produced different image pairs")
"""
assert marker in d, "RGB/IR cache loading not found"
if "SPC RGB/IR cache filtering" not in d:
    d = d.replace(marker, marker + check)
    ds.write_text(d, encoding="utf-8")


subprocess.run([sys.executable, str(kit / "tools" / "make_configs.py"), "--engine", str(eng)], check=True)
# Persist engine modifications in the overlay consumed by scripts/setup.sh/ps1.
overlay = Path(a.overlay)
for rel in [Path("train.py"), Path("utils/datasets.py"), Path("models/yolo_test.py"), Path("models/cmafm_variants.py"),
            *[p.relative_to(eng) for p in (eng / "models/spc").glob("*.yaml")]]:
    dest = overlay / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(eng / rel, dest)
print("kit installed")
