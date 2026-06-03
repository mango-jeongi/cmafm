import random
import matplotlib.pyplot as plt
import PIL.Image
from pathlib import Path

# --- SMART PATH LOGIC ---
if Path(r"../data").exists():
    BASE = Path(r"../data")
elif Path("../data").exists():
    BASE = Path("../data").resolve()
else:
    BASE = Path.cwd() / "data"

EXTRACTED_ROOT = BASE / "M3FD"

def verify_m3fd(num_samples=5):
    if not EXTRACTED_ROOT.exists():
        print(f"Extracted M3FD not found at {EXTRACTED_ROOT}. Please run scripts/prepare_all.py first.")
        return

    # Structure: M3FD / [Annotation, Ir, Vis]
    vis_dir = next(EXTRACTED_ROOT.glob("**/Vis"), None)
    ir_dir  = next(EXTRACTED_ROOT.glob("**/Ir"), None)

    if not vis_dir or not ir_dir:
        print("Error: Could not find Vis or Ir directories in M3FD.")
        return

    all_vis = list(vis_dir.glob("*.png"))
    samples = random.sample(all_vis, min(num_samples, len(all_vis)))

    for i, vis_path in enumerate(samples):
        stem = vis_path.stem
        ir_path = ir_dir / f"{stem}.png"
        
        if not ir_path.exists():
            print(f"  [Skip] Missing IR image for {stem}")
            continue

        img_rgb = PIL.Image.open(vis_path)
        img_ir  = PIL.Image.open(ir_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Sample {i+1} - M3FD RGB ({stem})")
        axes[0].axis('off')
        
        axes[1].imshow(img_ir, cmap='gray')
        axes[1].set_title(f"Sample {i+1} - M3FD Thermal ({stem})")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    verify_m3fd(5)
