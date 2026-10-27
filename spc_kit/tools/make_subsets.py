"""EXP-5: fixed evaluation subsets, built once and shared by every model.

Day/night: mean grayscale of the RGB frame, thresholds fixed before evaluating any model
(M3FD 60, FLIR 100 as in the WACV version; both are reported in the paper with the reason).
Scene subsets (optional): --scene-csv with lines "stem,scene" (e.g. from the M3FD scene labels:
Daytime / Night / Overcast / Challenge(fog, glare, backlight)); one subset per scene value.
Writes <out>/<name>_{rgb,ir}.txt and <out>/<name>.yaml, plus subsets.json with image counts.
"""
import argparse, csv, json
from pathlib import Path
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="data/spc_split")
ap.add_argument("--out", default="data/spc_subsets")
ap.add_argument("--scene-csv", default="")
ap.add_argument("--thr-m3fd", type=float, default=60)
ap.add_argument("--thr-flir", type=float, default=100)
a = ap.parse_args()
sp, out = Path(a.split), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
NAMES = ['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck']

def lines(p):
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def write(name, pairs, info):
    (out / f"{name}_rgb.txt").write_text("\n".join(r for r, _ in pairs) + "\n", encoding="utf-8")
    (out / f"{name}_ir.txt").write_text("\n".join(i for _, i in pairs) + "\n", encoding="utf-8")
    o = out.resolve().as_posix()
    (out / f"{name}.yaml").write_text(f"train_rgb: {o}/{name}_rgb.txt\ntrain_ir: {o}/{name}_ir.txt\n"
                                      f"val_rgb: {o}/{name}_rgb.txt\nval_ir: {o}/{name}_ir.txt\nnc: 6\nnames: {NAMES}\n")
    info[name] = len(pairs)

info, lum = {}, {}
for base, thr in (("test", a.thr_m3fd), ("flir_test", a.thr_flir)):
    if not (sp / f"{base}_rgb.txt").exists():
        continue
    pairs = list(zip(lines(sp / f"{base}_rgb.txt"), lines(sp / f"{base}_ir.txt")))
    tag = "m3fd" if base == "test" else "flir"
    write(f"{tag}_all", pairs, info)
    night, day = [], []
    for r, i in pairs:
        y = float(np.mean(cv2.imread(r, cv2.IMREAD_GRAYSCALE)))
        lum[Path(r).stem] = round(y, 2)
        (night if y < thr else day).append((r, i))
    write(f"{tag}_night", night, info)
    write(f"{tag}_day", day, info)
    if a.scene_csv and tag == "m3fd":
        scene = {row[0]: row[1] for row in csv.reader(open(a.scene_csv)) if len(row) >= 2}
        for sc in sorted(set(scene.values())):
            sub = [(r, i) for r, i in pairs if scene.get(Path(r).stem) == sc]
            if sub:
                write(f"m3fd_scene_{sc.lower().replace(' ', '_')}", sub, info)
json.dump({"counts": info, "thresholds": {"m3fd": a.thr_m3fd, "flir": a.thr_flir}, "luminance": lum},
          open(out / "subsets.json", "w"), indent=1)
print(json.dumps(info, indent=1))
