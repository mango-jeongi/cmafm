"""Write every model YAML of the SPC experiment matrix into cft_engine/models/spc/.

Usage: python tools/make_configs.py --engine cft_engine
All CMAFM-family configs are copies of yolov5l_cmafm_M3FD.yaml with only the fusion layer swapped,
so backbone, neck, head and the residual injection (Add2 into both streams) are identical.
"""
import argparse, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="cft_engine")
a = ap.parse_args()
eng = Path(a.engine)
out = eng / "models" / "spc"
out.mkdir(parents=True, exist_ok=True)

cmafm = (eng / "models" / "yolov5l_cmafm_M3FD.yaml").read_text()
add = (eng / "models" / "transformer" / "yolov5l_fusion_add_FLIR_aligned.yaml").read_text()
cft = (eng / "models" / "transformer" / "yolov5l_fusion_transformerx3_M3FD.yaml").read_text()
add = re.sub(r"^nc:\s*\d+", "nc: 6", add, flags=re.M)

def swap_fusion(mode):
    return re.sub(r"CMAFM_Fusion, \[(\d+)\]", lambda m: f"CMAFMVar, [{m.group(1)}, '{mode}']", cmafm)

configs = {"cft": cft, "dual_add": add, "cmafm_full": cmafm}
for mode in ("channel_only", "spatial_only", "no_alpha", "v_sigmoid", "se_self", "cbam_self", "concat_1x1"):
    configs[f"cmafm_{mode}" if mode not in ("se_self", "cbam_self", "concat_1x1") else f"ctrl_{mode}"] = swap_fusion(mode)

# single-modality baselines: stream one of the Dual+Add model + the same head.
# Thermal-only needs a placeholder layer 0 because parse_model cannot take from=-4 at index 0
# (yolo_test.forward_once feeds the IR image to any layer whose from is -4).
head = add[add.index("head:"):]
pre = add[:add.index("backbone:")]

def single(thermal):
    o = 1 if thermal else 0
    bb = ["[-1, 1, nn.Identity, []],  # placeholder (RGB input unused)"] if thermal else []
    bb += [f"[{-4 if thermal else -1}, 1, Focus, [64, 3]],", "[-1, 1, Conv, [128, 3, 2]],", "[-1, 3, C3, [128]],",
           "[-1, 1, Conv, [256, 3, 2]],", "[-1, 9, C3, [256]],", "[-1, 1, Conv, [512, 3, 2]],", "[-1, 9, C3, [512]],",
           "[-1, 1, Conv, [1024, 3, 2]],", "[-1, 1, SPP, [1024, [5, 9, 13]]],", "[-1, 3, C3, [1024, False]],"]
    p3, p4, h = 4 + o, 6 + o, 10 + o  # backbone P3/P4 outputs, first head layer
    hd = (head.replace("[[-1,21], 1, Concat", f"[[-1,{p4}], 1, Concat")
              .replace("[[-1,20], 1, Concat", f"[[-1,{p3}], 1, Concat")
              .replace("[[-1,27], 1, Concat", f"[[-1,{h + 4}], 1, Concat")
              .replace("[[-1,23], 1, Concat", f"[[-1,{h}], 1, Concat")
              .replace("[[30, 33, 36], 1, Detect", f"[[{h + 7}, {h + 10}, {h + 13}], 1, Detect"))
    return pre + "backbone:\n  [\n" + "".join(f"   {l}\n" for l in bb) + "  ]\n\n" + hd

configs["rgb_only"] = single(False)
configs["thermal_only"] = single(True)

for k, v in configs.items():
    (out / f"{k}.yaml").write_text("\n".join(line.rstrip() for line in v.splitlines()) + "\n")
    print("wrote", out / f"{k}.yaml")
