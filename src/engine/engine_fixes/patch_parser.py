import subprocess
import os

print("--- Starting Ultimate Total Repair (v8 - Final Cleanup & Plot Fix) ---")

# 1. RESET FILES
engine_dir = 'cft_engine'
files_to_fix = [
    'models/yolo_test.py', 
    'utils/datasets.py', 
    'utils/general.py', 
    'utils/loss.py', 
    'utils/plots.py',
    'test.py'
]

for f_path in files_to_fix:
    try:
        print(f"Resetting {f_path} to original DocF state...")
        subprocess.run(['git', 'checkout', f_path], cwd=engine_dir, check=True)
    except Exception as e:
        print(f"Warning: Git reset failed for {f_path}: {e}")

# 2. PATCH PARSER (yolo_test.py)
parser_path = os.path.join(engine_dir, 'models/yolo_test.py')
with open(parser_path, 'r') as f:
    content = f.read()

if 'from .cmafm import CMAFM_Fusion' not in content:
    content = 'from .cmafm import CMAFM_Fusion\n' + content

old_block = """        elif m is Concat:
            c2 = sum([ch[x] for x in f])"""

new_block = """        elif m is Concat:
            c2 = sum([ch[x] for x in f])
            args = [args[0] if args else 1]  # Fix: DocF engine Concat bug
        elif m is CMAFM_Fusion:
            c1 = [ch[x] for x in f]
            c2 = args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2]"""

if old_block in content:
    content = content.replace(old_block, new_block)
    with open(parser_path, 'w') as f:
        f.write(content)
    print("Successfully patched models/yolo_test.py")

# 3. PATCH UTILS (datasets.py and general.py)
for util_file in ['utils/datasets.py', 'utils/general.py']:
    u_path = os.path.join(engine_dir, util_file)
    with open(u_path, 'r') as f:
        u_content = f.read()

    if 'datasets.py' in util_file:
        old_func = """def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]"""

        new_func = """def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    # Fix for DocF: correctly handle both /images/ and /images_ir/ by looking for 'images' folder
    out = []
    for x in img_paths:
        x = x.replace('\\\\', '/')
        p = x.replace('/images_ir/', '/labels/').replace('/images/', '/labels/')
        p = os.path.splitext(p)[0] + '.txt'
        out.append(p)
    return out"""
        if old_func in u_content:
            u_content = u_content.replace(old_func, new_func)
            
        u_content = u_content.replace('torch.load(cache_rgb_path)', 'torch.load(cache_rgb_path, weights_only=False)')
        u_content = u_content.replace('torch.load(cache_ir_path)', 'torch.load(cache_ir_path, weights_only=False)')

    # FIX for strip_optimizer crash in general.py (PyTorch 2.6)
    if 'general.py' in util_file:
        u_content = u_content.replace("torch.load(f, map_location=torch.device('cpu'))", "torch.load(f, map_location=torch.device('cpu'), weights_only=False)")

    u_content = u_content.replace('.astype(np.int)', '.astype(int)')
    u_content = u_content.replace('.astype(np.float)', '.astype(float)')
    u_content = u_content.replace('.astype(np.bool)', '.astype(bool)')

    with open(u_path, 'w') as f:
        f.write(u_content)
    print(f"Successfully patched {util_file}")

# 4. PATCH LOSS (utils/loss.py)
loss_path = os.path.join(engine_dir, 'utils/loss.py')
with open(loss_path, 'r') as f:
    l_content = f.read()

old_indexing = "indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices"
new_indexing = "indices.append((b, a, gj.clamp_(0, (gain[3] - 1).long()), gi.clamp_(0, (gain[2] - 1).long())))  # image, anchor, grid indices"

if old_indexing in l_content:
    l_content = l_content.replace(old_indexing, new_indexing)
else:
    l_lines = l_content.splitlines()
    final_lines = []
    for line in l_lines:
        if 'indices.append((b, a, gj.clamp_' in line:
            indent = line[:line.find('indices.append')]
            line = indent + "indices.append((b, a, gj.clamp_(0, (gain[3] - 1).long()), gi.clamp_(0, (gain[2] - 1).long())))"
        final_lines.append(line)
    l_content = "\n".join(final_lines)

with open(loss_path, 'w') as f:
    f.write(l_content)
print("Successfully patched utils/loss.py")

# 5. PATCH TEST LOGGING (test.py)
test_path = os.path.join(engine_dir, 'test.py')
with open(test_path, 'r') as f:
    t_content = f.read()

old_wandb_log = "wandb_logger.wandb.Image(img[si]"
new_wandb_log = "wandb_logger.wandb.Image(img[si][:3]"

if old_wandb_log in t_content:
    t_content = t_content.replace(old_wandb_log, new_wandb_log)
    with open(test_path, 'w') as f:
        f.write(t_content)
    print("Successfully patched test.py")

# 6. PATCH MOSAIC PLOTTING (utils/plots.py) - Fixes "could not broadcast input array [6] into [3]"
plots_path = os.path.join(engine_dir, 'utils/plots.py')
with open(plots_path, 'r') as f:
    p_content = f.read()

old_mosaic_line = "mosaic[block_y:block_y + h, block_x:block_x + w, :] = img"
new_mosaic_line = "mosaic[block_y:block_y + h, block_x:block_x + w, :] = img[:, :, :3] if img.shape[2] == 6 else img"

if old_mosaic_line in p_content:
    p_content = p_content.replace(old_mosaic_line, new_mosaic_line)
    with open(plots_path, 'w') as f:
        f.write(p_content)
    print("Successfully patched utils/plots.py (6ch Plot Fix)")

print("--- Ultimate Repair Complete (v8) ---")

# 7. PATCH YAML DATASET CONFIGS (Replacing apply_local_paths.py)
from dotenv import load_dotenv
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")
DATA_DIR = os.environ.get("CMAFM_DATA_DIR")

if not DATA_DIR:
    print("[patch_parser] WARNING: CMAFM_DATA_DIR not set in .env — skipping YAML patch.")
else:
    DATA_DIR = Path(DATA_DIR).resolve()
    MERGED_TXT = DATA_DIR / "merged_txt"
    CFT_DATA_DIR = _REPO_ROOT / engine_dir / "data"

    SCENARIOS = {
        "M3FD_FLIR.yaml": {
            "train_rgb": str(MERGED_TXT / "train_rgb_unified.txt").replace('\\', '/'),
            "val_rgb":   str(MERGED_TXT / "val_rgb_unified.txt").replace('\\', '/'),
            "train_ir":  str(MERGED_TXT / "train_ir_unified.txt").replace('\\', '/'),
            "val_ir":    str(MERGED_TXT / "val_ir_unified.txt").replace('\\', '/'),
            "nc":        "6",
            "names":     "['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck']",
        },
        "M3FD_ONLY.yaml": {
            "train_rgb": str(MERGED_TXT / "train_rgb_m3fd_only.txt").replace('\\', '/'),
            "val_rgb":   str(MERGED_TXT / "val_rgb_m3fd_only.txt").replace('\\', '/'),
            "train_ir":  str(MERGED_TXT / "train_ir_m3fd_only.txt").replace('\\', '/'),
            "val_ir":    str(MERGED_TXT / "val_ir_m3fd_only.txt").replace('\\', '/'),
            "nc":        "6",
            "names":     "['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck']",
        },
        "mini.yaml": {
            "train_rgb": str(MERGED_TXT / "mini_train_rgb.txt").replace('\\', '/'),
            "val_rgb":   str(MERGED_TXT / "mini_val_rgb.txt").replace('\\', '/'),
            "train_ir":  str(MERGED_TXT / "mini_train_ir.txt").replace('\\', '/'),
            "val_ir":    str(MERGED_TXT / "mini_val_ir.txt").replace('\\', '/'),
            "nc":        "6",
            "names":     "['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck']",
        },
    }

    for fname, fields in SCENARIOS.items():
        yaml_path = CFT_DATA_DIR / fname
        if not yaml_path.exists():
            print(f"[patch_parser] Skipping {fname} (not found).")
            continue
        lines = []
        for key, val in fields.items():
            lines.append(f"{key}: {val}")
        yaml_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
        print(f"[patch_parser] Patched YAML data config {yaml_path}")
