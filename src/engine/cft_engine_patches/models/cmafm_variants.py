"""CMAFM ablation variants and same-cost controls for the SPC revision.

All classes follow the two-output interface consumed by Add2 (returns (fused, fused)),
so they drop into the YAML at the same place as CMAFM_Fusion.
mode:
  full          - identical to the module serialized in best.pt (cmafm_checkpoint._CMAFM)
  channel_only  - Stage 1 only (no spatial cross-gating)
  spatial_only  - Stage 2 only (no channel cross-gating)
  no_alpha      - Stages 1-2, fixed 0.5/0.5 average instead of the learnable gate alpha
  v_sigmoid     - full, but the value vector v is bounded with a sigmoid
  se_self       - control: intra-modal SE on each stream, average fusion (no cross-modal terms)
  cbam_self     - control: intra-modal CBAM on each stream, average fusion (no cross-modal terms)
  concat_1x1    - control: Dual+Concat, fused = BN(Conv1x1([rgb; ir])) + same output refinement
"""
import torch
import torch.nn as nn

MODES = ("full", "channel_only", "spatial_only", "no_alpha", "v_sigmoid", "se_self", "cbam_self", "concat_1x1")


def _spatial_branch(c):
    return nn.Sequential(
        nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        nn.Conv2d(c, c, 1, bias=False),
    )


def _out_proj(c):
    return nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))


class _SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True), nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(dim=[2, 3]))[:, :, None, None]


class _CBAM(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True), nn.Linear(c // r, c))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean(dim=[2, 3])) + self.mlp(x.amax(dim=[2, 3])))
        x = x * ca[:, :, None, None]
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


class CMAFMVar(nn.Module):
    def __init__(self, channels, mode="full"):
        super().__init__()
        assert mode in MODES, mode
        self.mode = mode
        c = channels
        if mode == "concat_1x1":
            self.proj = nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c))
            self.out_proj = _out_proj(c)
            return
        if mode in ("se_self", "cbam_self"):
            blk = _SE if mode == "se_self" else _CBAM
            self.att_rgb, self.att_ir = blk(c), blk(c)
            self.out_proj = _out_proj(c)
            return
        if mode != "spatial_only":
            self.rgb_channel_q = nn.Linear(c, c)
            self.thermal_channel_kv = nn.Linear(c, 2 * c)
            self.thermal_channel_q = nn.Linear(c, c)
            self.rgb_channel_kv = nn.Linear(c, 2 * c)
        if mode != "channel_only":
            self.rgb_spatial = _spatial_branch(c)
            self.thermal_spatial = _spatial_branch(c)
            self.rgb_gate = nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.Sigmoid())
            self.thermal_gate = nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.Sigmoid())
        self.norm_rgb = nn.BatchNorm2d(c)
        self.norm_thermal = nn.BatchNorm2d(c)
        if mode != "no_alpha":
            self.fusion_gate = nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.Sigmoid())
        self.out_proj = _out_proj(c)
        self.last_alpha = None  # mean alpha of the last forward, read by tools/alpha_stats.py

    def _channel(self, q_proj, kv_proj, fq, fkv):
        b, c = fq.shape[:2]
        q = q_proj(fq.mean(dim=[2, 3]))
        k, v = kv_proj(fkv.mean(dim=[2, 3])).chunk(2, dim=-1)
        if self.mode == "v_sigmoid":
            v = v.sigmoid()
        return (v * (q * k).sigmoid()).view(b, c, 1, 1)

    def forward(self, x):
        rgb, ir = x[0], x[1]
        if self.mode == "concat_1x1":
            fused = self.proj(torch.cat([rgb, ir], 1))
            fused = self.out_proj(fused) + fused
            return fused, fused
        if self.mode in ("se_self", "cbam_self"):
            fused = 0.5 * (self.att_rgb(rgb) + self.att_ir(ir))
            fused = self.out_proj(fused) + fused
            return fused, fused
        if self.mode == "spatial_only":
            rgb_e, ir_e = rgb, ir
        else:
            rgb_e = rgb * self._channel(self.rgb_channel_q, self.thermal_channel_kv, rgb, ir)
            ir_e = ir * self._channel(self.thermal_channel_q, self.rgb_channel_kv, ir, rgb)
        if self.mode == "channel_only":
            rgb_c, ir_c = rgb_e, ir_e
        else:
            rs, ts = self.rgb_spatial(rgb_e), self.thermal_spatial(ir_e)
            rgb_c = self.rgb_gate(torch.cat([rs, ts], 1)) * rgb_e
            ir_c = self.thermal_gate(torch.cat([ts, rs], 1)) * ir_e
        rgb_o = self.norm_rgb(rgb + rgb_c)
        ir_o = self.norm_thermal(ir + ir_c)
        if self.mode == "no_alpha":
            fused = 0.5 * (rgb_o + ir_o)
        else:
            a = self.fusion_gate(torch.cat([rgb_o, ir_o], 1))
            self.last_alpha = a.detach().mean(dim=[1, 2, 3])  # per-image mean RGB weight
            fused = a * rgb_o + (1 - a) * ir_o
        fused = self.out_proj(fused) + fused
        return fused, fused
