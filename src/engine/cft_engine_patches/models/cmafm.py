import torch
import torch.nn as nn

class CMAFM_Fusion(nn.Module):
    """
    Cross-Modal Attention Fusion Module (CMAFM) for YOLOv5.
    Structural requirement for DocF: Must return a list of two enhanced tensors [rgb, ir].
    """
    def __init__(self, c1, c2):
        super().__init__()
        # c1 is a list [ch_rgb, ch_ir], c2 is the output channel count
        self.channels = c2 
        
        # 1. Channel Cross-Attention (Global)
        self.rgb_channel_q = nn.Linear(c2, c2)
        self.thermal_channel_kv = nn.Linear(c2, c2 * 2)
        self.thermal_channel_q = nn.Linear(c2, c2)
        self.rgb_channel_kv = nn.Linear(c2, c2 * 2)

        # 2. Spatial Cross-Attention (Local)
        self.rgb_spatial = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 1, bias=False),
        )
        self.thermal_spatial = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 1, bias=False),
        )

        # 3. Bi-directional cross-gating
        self.rgb_gate = nn.Sequential(nn.Conv2d(c2 * 2, c2, 1, bias=False), nn.Sigmoid())
        self.thermal_gate = nn.Sequential(nn.Conv2d(c2 * 2, c2, 1, bias=False), nn.Sigmoid())

        self.out_proj_rgb = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.out_proj_ir = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

    def _channel_attn(self, q_proj, kv_proj, feat_q, feat_kv):
        B, C, _, _ = feat_q.shape
        q = q_proj(feat_q.mean(dim=[2, 3]))
        k, v = kv_proj(feat_kv.mean(dim=[2, 3])).chunk(2, dim=-1)
        scale = (q * k).sigmoid()
        return (v * scale).view(B, C, 1, 1)

    def forward(self, x):
        # x is [feat_rgb, feat_ir] from the YAML indices
        rgb_feat, thermal_feat = x[0], x[1]
        
        # Channel-wise global exchange
        rgb_enhanced = rgb_feat * self._channel_attn(self.rgb_channel_q, self.thermal_channel_kv, rgb_feat, thermal_feat)
        thermal_enhanced = thermal_feat * self._channel_attn(self.thermal_channel_q, self.rgb_channel_kv, thermal_feat, rgb_feat)

        # Spatial local exchange
        rgb_sp = self.rgb_spatial(rgb_enhanced)
        thermal_sp = self.thermal_spatial(thermal_enhanced)
        
        # Bi-directional cross-attention
        rgb_cross = self.rgb_gate(torch.cat([rgb_sp, thermal_sp], dim=1)) * rgb_enhanced
        thermal_cross = self.thermal_gate(torch.cat([thermal_sp, rgb_sp], dim=1)) * thermal_enhanced

        # Enhanced streams
        rgb_out = self.out_proj_rgb(rgb_feat + rgb_cross)
        thermal_out = self.out_proj_ir(thermal_feat + thermal_cross)
        
        # RETURN DUAL STREAMS: This satisfies the DocF engine's Add2 modules
        return [rgb_out, thermal_out]
