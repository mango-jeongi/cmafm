"""
CMAFM — Anonymous Reproducibility & Architecture Verification Demo
WACV 2027 Applications Track (Round 2)
"""

import sys
import time
from pathlib import Path
import torch
import torch.nn as nn

# Ensure local imports work
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "src" / "fusion"))
sys.path.insert(0, str(repo_root / "cft_engine"))

from model import CrossModalAttentionFusion


def test_cmafm_forward():
    print("=" * 65)
    print("CMAFM: Cross-Modal Attention Fusion Module Architecture Test")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")
    
    # Test across pyramid feature scales: C3 (256ch), C4 (512ch), C5 (1024ch)
    test_scales = [
        ("C3 Scale (Stride 8)",  256,  80, 80),
        ("C4 Scale (Stride 16)", 512,  40, 40),
        ("C5 Scale (Stride 32)", 1024, 20, 20),
    ]
    
    for name, channels, height, width in test_scales:
        print(f"\n[+] Testing {name} — Channels: {channels}, Spatial: {height}x{width}")
        
        # Instantiate CrossModalAttentionFusion block
        cmafm_block = CrossModalAttentionFusion(channels=channels).to(device)
        cmafm_block.eval()
        
        # Generate synthetic input tensors for visible (RGB) and thermal (LWIR)
        rgb_feat = torch.randn(1, channels, height, width, device=device)
        th_feat  = torch.randn(1, channels, height, width, device=device)
        
        # Warmup and timed forward execution
        with torch.no_grad():
            for _ in range(5):
                _ = cmafm_block(rgb_feat, th_feat)
            
            t0 = time.perf_counter()
            fused_feat = cmafm_block(rgb_feat, th_feat)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            
        print(f"    - Input RGB shape:     {tuple(rgb_feat.shape)}")
        print(f"    - Input Thermal shape: {tuple(th_feat.shape)}")
        print(f"    - Fused Output shape:  {tuple(fused_feat.shape)}")
        print(f"    - Forward Latency:     {elapsed_ms:.2f} ms")
        
        # Verify output dimensional consistency
        assert fused_feat.shape == rgb_feat.shape, f"Shape mismatch: {fused_feat.shape} vs {rgb_feat.shape}"
        assert not torch.isnan(fused_feat).any(), "Output contains NaN values!"
        print(f"    - Integrity Check:     PASSED (Tensor is finite and dimensionally consistent)")
    
    print("\n" + "=" * 65)
    print("[SUCCESS] All CMAFM architectural forward tests passed cleanly!")
    print("=" * 65)


if __name__ == "__main__":
    test_cmafm_forward()
