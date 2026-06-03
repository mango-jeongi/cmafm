import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Path logic
import os
env_data_dir = os.environ.get("CMAFM_DATA_DIR")
user_home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
local_ssd_data = Path(user_home) / "bmvc_data" if user_home else None

if env_data_dir and Path(env_data_dir).exists():
    BASE = Path(env_data_dir)
elif local_ssd_data and local_ssd_data.exists():
    BASE = local_ssd_data
elif Path("../data").exists():
    BASE = Path("../data").resolve()
else:
    BASE = Path.cwd() / "data"

VAL_LIST = BASE / "merged_txt" / "val_rgb_flir_only.txt"

def analyze_brightness():
    if not VAL_LIST.exists():
        print(f"Error: {VAL_LIST} not found.")
        return

    print("Analyzing FLIR brightness distribution...")
    with open(VAL_LIST, 'r') as f:
        paths = [l.strip() for l in f if l.strip()]
    
    brightness_vals = []
    for p in tqdm(paths):
        img = cv2.imread(p)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(np.mean(gray))
    
    plt.hist(brightness_vals, bins=50)
    plt.title("FLIR Validation Set Brightness Distribution")
    plt.xlabel("Mean Brightness (0-255)")
    plt.ylabel("Frequency")
    plt.axvline(x=60, color='r', linestyle='--', label='Current Threshold (60)')
    plt.legend()
    plt.savefig("flir_brightness_hist.png")
    
    print(f"\nStats:")
    print(f"  Min:    {np.min(brightness_vals):.2f}")
    print(f"  Max:    {np.max(brightness_vals):.2f}")
    print(f"  Mean:   {np.mean(brightness_vals):.2f}")
    print(f"  Median: {np.median(brightness_vals):.2f}")
    
    # Propose new threshold (lower quartile or specific value)
    below_60 = sum(1 for v in brightness_vals if v < 60)
    below_40 = sum(1 for v in brightness_vals if v < 40)
    print(f"\nImages below 60: {below_60} (Too high?)")
    print(f"Images below 40: {below_40}")

if __name__ == "__main__":
    analyze_brightness()
