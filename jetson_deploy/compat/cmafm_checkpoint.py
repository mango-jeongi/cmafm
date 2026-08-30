"""Historical CMAFM classes required by the supplied legacy best.pt checkpoint.

The checkpoint was serialized with ``models.common.CMAFM_Fusion`` and
``models.common._CMAFM``. This implementation is taken from this repository's
earlier CFT training integration and is installed only into the vendored engine.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CMAFM_Fusion(nn.Module):
    """Adapt CMAFM to the two-output interface consumed by legacy Add2 layers."""

    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.cmafm = _CMAFM(channels, num_heads, dropout)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        rgb_h, rgb_w = rgb_fea.shape[2:]
        ir_h, ir_w = ir_fea.shape[2:]
        fused = self.cmafm(rgb_fea, ir_fea)
        fused_rgb = (
            fused
            if fused.shape[2:] == (rgb_h, rgb_w)
            else F.interpolate(
                fused, size=(rgb_h, rgb_w), mode="bilinear", align_corners=False
            )
        )
        fused_ir = (
            fused
            if fused.shape[2:] == (ir_h, ir_w)
            else F.interpolate(
                fused, size=(ir_h, ir_w), mode="bilinear", align_corners=False
            )
        )
        return fused_rgb, fused_ir


class _CMAFM(nn.Module):
    """Efficient bidirectional channel/spatial cross-modal attention."""

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.channels = channels

        self.rgb_channel_q = nn.Linear(channels, channels)
        self.thermal_channel_kv = nn.Linear(channels, channels * 2)
        self.thermal_channel_q = nn.Linear(channels, channels)
        self.rgb_channel_kv = nn.Linear(channels, channels * 2)

        def spatial_branch(channel_count):
            return nn.Sequential(
                nn.Conv2d(
                    channel_count,
                    channel_count,
                    3,
                    padding=1,
                    groups=channel_count,
                    bias=False,
                ),
                nn.BatchNorm2d(channel_count),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel_count, channel_count, 1, bias=False),
            )

        self.rgb_spatial = spatial_branch(channels)
        self.thermal_spatial = spatial_branch(channels)
        self.rgb_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False), nn.Sigmoid()
        )
        self.thermal_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False), nn.Sigmoid()
        )
        self.norm_rgb = nn.BatchNorm2d(channels)
        self.norm_thermal = nn.BatchNorm2d(channels)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _channel_cross_attn(q_proj, kv_proj, feat_q, feat_kv):
        batch, channels, _, _ = feat_q.shape
        q_vec = feat_q.mean(dim=[2, 3])
        kv_vec = feat_kv.mean(dim=[2, 3])
        q = q_proj(q_vec)
        key, value = kv_proj(kv_vec).chunk(2, dim=-1)
        scale = (q * key).sigmoid()
        return (value * scale).view(batch, channels, 1, 1)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        if rgb.shape[2:] != thermal.shape[2:]:
            thermal = F.interpolate(
                thermal, size=rgb.shape[2:], mode="bilinear", align_corners=False
            )

        rgb_channel = self._channel_cross_attn(
            self.rgb_channel_q, self.thermal_channel_kv, rgb, thermal
        )
        thermal_channel = self._channel_cross_attn(
            self.thermal_channel_q, self.rgb_channel_kv, thermal, rgb
        )
        rgb_enhanced = rgb * rgb_channel
        thermal_enhanced = thermal * thermal_channel

        rgb_spatial = self.rgb_spatial(rgb_enhanced)
        thermal_spatial = self.thermal_spatial(thermal_enhanced)
        rgb_cross = (
            self.rgb_gate(torch.cat([rgb_spatial, thermal_spatial], dim=1))
            * rgb_enhanced
        )
        thermal_cross = (
            self.thermal_gate(torch.cat([thermal_spatial, rgb_spatial], dim=1))
            * thermal_enhanced
        )
        rgb_out = self.norm_rgb(rgb + rgb_cross)
        thermal_out = self.norm_thermal(thermal + thermal_cross)

        gate = self.fusion_gate(torch.cat([rgb_out, thermal_out], dim=1))
        fused = gate * rgb_out + (1 - gate) * thermal_out
        return self.out_proj(fused) + fused

