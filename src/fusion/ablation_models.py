"""
Ablation model variants for comparison study.

Variants:
  1. SingleModalDetector (RGB-only or Thermal-only)
  2. EarlyFusionDetector (6ch input, single backbone)
  3. DualNoCrossAttnDetector (dual backbone, concat fusion, no attention)
  4. Full model (already in model.py)
"""

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from config import ModelConfig


# ---------------------------------------------------------------------------
# 1. Single-Modal Detector (RGB-only or Thermal-only)
# ---------------------------------------------------------------------------

class SingleBackboneWithFPN(nn.Module):
    """Standard single ResNet-50 + FPN backbone."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.out_channels = cfg.fpn_out_channels

        weights = ResNet50_Weights.DEFAULT if cfg.pretrained else None
        resnet = resnet50(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        in_channels_list = [512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=cfg.fpn_out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        fpn_input = OrderedDict([("0", c3), ("1", c4), ("2", c5)])
        return self.fpn(fpn_input)


class SingleModalDetector(nn.Module):
    """
    Ablation 1 & 2: RGB-only or Thermal-only detector.
    Standard Faster R-CNN with single ResNet-50 backbone.
    """

    def __init__(self, cfg: ModelConfig, num_classes: int, modality: str = "rgb"):
        super().__init__()
        self.modality = modality  # "rgb" or "thermal"

        backbone = SingleBackboneWithFPN(cfg)

        anchor_sizes = cfg.rpn_anchor_sizes
        aspect_ratios = cfg.rpn_aspect_ratios
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        rpn_head = RPNHead(cfg.fpn_out_channels, anchor_generator.num_anchors_per_location()[0])

        self.detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes + 1,
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            box_score_thresh=cfg.roi_score_thresh,
            box_nms_thresh=cfg.roi_nms_thresh,
            box_detections_per_img=cfg.roi_detections_per_img,
        )

    def forward(self, rgb, thermal, targets=None):
        if self.modality == "rgb":
            images = [rgb[i] for i in range(rgb.shape[0])]
        else:
            images = [thermal[i] for i in range(thermal.shape[0])]

        if targets is not None:
            return self.detector(images, targets)
        return self.detector(images)


# ---------------------------------------------------------------------------
# 2. Early Fusion Detector (6ch input, single backbone)
# ---------------------------------------------------------------------------

class EarlyFusionBackbone(nn.Module):
    """Single backbone that takes 6ch (RGB+Thermal) input."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.out_channels = cfg.fpn_out_channels

        weights = ResNet50_Weights.DEFAULT if cfg.pretrained else None
        resnet = resnet50(weights=weights)

        # Replace first conv: 3ch → 6ch
        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize: copy pretrained weights for first 3ch, duplicate for last 3ch
        with torch.no_grad():
            self.conv1.weight[:, :3] = old_conv.weight
            self.conv1.weight[:, 3:] = old_conv.weight

        self.stem_rest = nn.Sequential(resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        in_channels_list = [512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=cfg.fpn_out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.stem_rest(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        fpn_input = OrderedDict([("0", c3), ("1", c4), ("2", c5)])
        return self.fpn(fpn_input)


class EarlyFusionDetector(nn.Module):
    """
    Ablation 3: Concatenate RGB+Thermal at input → single backbone.
    """

    def __init__(self, cfg: ModelConfig, num_classes: int):
        super().__init__()
        backbone = EarlyFusionBackbone(cfg)

        anchor_sizes = cfg.rpn_anchor_sizes
        aspect_ratios = cfg.rpn_aspect_ratios
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        rpn_head = RPNHead(cfg.fpn_out_channels, anchor_generator.num_anchors_per_location()[0])

        # 6ch normalization
        image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]

        self.detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes + 1,
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            box_score_thresh=cfg.roi_score_thresh,
            box_nms_thresh=cfg.roi_nms_thresh,
            box_detections_per_img=cfg.roi_detections_per_img,
            image_mean=image_mean,
            image_std=image_std,
        )

    def forward(self, rgb, thermal, targets=None):
        combined = torch.cat([rgb, thermal], dim=1)  # 6ch
        images = [combined[i] for i in range(combined.shape[0])]
        if targets is not None:
            return self.detector(images, targets)
        return self.detector(images)


# ---------------------------------------------------------------------------
# 3. Dual Backbone WITHOUT Cross-Modal Attention (concat fusion)
# ---------------------------------------------------------------------------

class DualConcatFPN(nn.Module):
    """Two backbones, features concatenated (no attention), 1x1 conv to reduce."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.out_channels = cfg.fpn_out_channels

        weights = ResNet50_Weights.DEFAULT if cfg.pretrained else None

        def _make_stages(resnet):
            return nn.ModuleDict({
                "stem": nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
                "layer1": resnet.layer1,
                "layer2": resnet.layer2,
                "layer3": resnet.layer3,
                "layer4": resnet.layer4,
            })

        self.rgb_stages = _make_stages(resnet50(weights=weights))
        self.thermal_stages = _make_stages(resnet50(weights=weights))

        # Simple concat + 1x1 conv (no attention)
        c3_ch, c4_ch, c5_ch = 512, 1024, 2048
        self.reduce_c3 = nn.Sequential(
            nn.Conv2d(c3_ch * 2, c3_ch, 1, bias=False),
            nn.GroupNorm(32, c3_ch), nn.ReLU(inplace=True))
        self.reduce_c4 = nn.Sequential(
            nn.Conv2d(c4_ch * 2, c4_ch, 1, bias=False),
            nn.GroupNorm(32, c4_ch), nn.ReLU(inplace=True))
        self.reduce_c5 = nn.Sequential(
            nn.Conv2d(c5_ch * 2, c5_ch, 1, bias=False),
            nn.GroupNorm(32, c5_ch), nn.ReLU(inplace=True))

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[c3_ch, c4_ch, c5_ch],
            out_channels=cfg.fpn_out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def _forward_backbone(self, stages, x):
        x = stages["stem"](x)
        c2 = stages["layer1"](x)
        c3 = stages["layer2"](c2)
        c4 = stages["layer3"](c3)
        c5 = stages["layer4"](c4)
        return c3, c4, c5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb = x[:, :3]
        thermal = x[:, 3:]

        r_c3, r_c4, r_c5 = self._forward_backbone(self.rgb_stages, rgb)
        t_c3, t_c4, t_c5 = self._forward_backbone(self.thermal_stages, thermal)

        # Simple concatenation + reduction (NO attention)
        fused_c3 = self.reduce_c3(torch.cat([r_c3, t_c3], dim=1))
        fused_c4 = self.reduce_c4(torch.cat([r_c4, t_c4], dim=1))
        fused_c5 = self.reduce_c5(torch.cat([r_c5, t_c5], dim=1))

        fpn_input = OrderedDict([("0", fused_c3), ("1", fused_c4), ("2", fused_c5)])
        return self.fpn(fpn_input)


class DualNoCrossAttnDetector(nn.Module):
    """
    Ablation 4: Dual backbone + simple concat (no cross-modal attention).
    """

    def __init__(self, cfg: ModelConfig, num_classes: int):
        super().__init__()
        backbone = DualConcatFPN(cfg)

        anchor_sizes = cfg.rpn_anchor_sizes
        aspect_ratios = cfg.rpn_aspect_ratios
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        rpn_head = RPNHead(cfg.fpn_out_channels, anchor_generator.num_anchors_per_location()[0])

        image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]

        self.detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes + 1,
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            box_score_thresh=cfg.roi_score_thresh,
            box_nms_thresh=cfg.roi_nms_thresh,
            box_detections_per_img=cfg.roi_detections_per_img,
            image_mean=image_mean,
            image_std=image_std,
        )

    def forward(self, rgb, thermal, targets=None):
        combined = torch.cat([rgb, thermal], dim=1)
        images = [combined[i] for i in range(combined.shape[0])]
        if targets is not None:
            return self.detector(images, targets)
        return self.detector(images)



# ---------------------------------------------------------------------------
# 4. Simplified CMAFM (Mid-Level Fusion)
# ---------------------------------------------------------------------------

class SimplifiedCMAFMFusion(nn.Module):
    """
    Simplified mid-level fusion mechanism inspired by Cross-Modal Attention.
    Uses Global Average Pooling (GAP) to dynamically weigh the importance of
    RGB vs. Thermal feature maps based on the overall lighting/channel context.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # Dynamic weighting network (joint modality representation to weights)
        self.weight_net = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels * 2),
        )

    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = rgb_feat.shape

        # 1. Global Average Pooling (GAP) to obtain illumination / global context
        rgb_gap = rgb_feat.mean(dim=[2, 3])       # Shape: (B, C)
        thermal_gap = thermal_feat.mean(dim=[2, 3]) # Shape: (B, C)

        # 2. Concatenate pooled features
        gap_concat = torch.cat([rgb_gap, thermal_gap], dim=-1) # Shape: (B, 2C)

        # 3. Predict channel-wise weights
        weights = self.weight_net(gap_concat).sigmoid()        # Shape: (B, 2C)
        rgb_w, thermal_w = weights.chunk(2, dim=1)             # Shape: (B, C) each

        # 4. Reshape for broadcasting
        rgb_w = rgb_w.view(B, C, 1, 1)
        thermal_w = thermal_w.view(B, C, 1, 1)

        # 5. Weighted summation
        fused = rgb_feat * rgb_w + thermal_feat * thermal_w
        return fused


class SimplifiedDualBackboneFPN(nn.Module):
    """Dual backbones (RGB + Thermal) with Simplified CMAFM Fusion and FPN."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.out_channels = cfg.fpn_out_channels

        weights = ResNet50_Weights.DEFAULT if cfg.pretrained else None

        def _make_stages(resnet):
            return nn.ModuleDict({
                "stem": nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
                "layer1": resnet.layer1,
                "layer2": resnet.layer2,
                "layer3": resnet.layer3,
                "layer4": resnet.layer4,
            })

        self.rgb_stages = _make_stages(resnet50(weights=weights))
        self.thermal_stages = _make_stages(resnet50(weights=weights))

        # ResNet-50 channel sizes: C3=512, C4=1024, C5=2048
        c3_ch, c4_ch, c5_ch = cfg.backbone_channels[1], cfg.backbone_channels[2], cfg.backbone_channels[3]

        self.fusion_c3 = SimplifiedCMAFMFusion(c3_ch)
        self.fusion_c4 = SimplifiedCMAFMFusion(c4_ch)
        self.fusion_c5 = SimplifiedCMAFMFusion(c5_ch)

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[c3_ch, c4_ch, c5_ch],
            out_channels=cfg.fpn_out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def _forward_backbone(self, stages, x):
        x = stages["stem"](x)
        c2 = stages["layer1"](x)
        c3 = stages["layer2"](c2)
        c4 = stages["layer3"](c3)
        c5 = stages["layer4"](c4)
        return c3, c4, c5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb = x[:, :3]
        thermal = x[:, 3:]

        r_c3, r_c4, r_c5 = self._forward_backbone(self.rgb_stages, rgb)
        t_c3, t_c4, t_c5 = self._forward_backbone(self.thermal_stages, thermal)

        # Apply simplified CMAFM fusion at C3, C4, C5 feature scales
        fused_c3 = self.fusion_c3(r_c3, t_c3)
        fused_c4 = self.fusion_c4(r_c4, t_c4)
        fused_c5 = self.fusion_c5(r_c5, t_c5)

        fpn_input = OrderedDict([("0", fused_c3), ("1", fused_c4), ("2", fused_c5)])
        return self.fpn(fpn_input)


class SimplifiedMidLevelFusionDetector(nn.Module):
    """
    Ablation 5: Dual backbone with Simplified CMAFM (dynamic mid-level fusion).
    """

    def __init__(self, cfg: ModelConfig, num_classes: int):
        super().__init__()
        backbone = SimplifiedDualBackboneFPN(cfg)

        anchor_sizes = cfg.rpn_anchor_sizes
        aspect_ratios = cfg.rpn_aspect_ratios
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        rpn_head = RPNHead(cfg.fpn_out_channels, anchor_generator.num_anchors_per_location()[0])

        image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]

        self.detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes + 1,
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            box_score_thresh=cfg.roi_score_thresh,
            box_nms_thresh=cfg.roi_nms_thresh,
            box_detections_per_img=cfg.roi_detections_per_img,
            image_mean=image_mean,
            image_std=image_std,
        )

    def forward(self, rgb, thermal, targets=None):
        combined = torch.cat([rgb, thermal], dim=1)
        images = [combined[i] for i in range(combined.shape[0])]
        if targets is not None:
            return self.detector(images, targets)
        return self.detector(images)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_ablation_model(variant: str, cfg: ModelConfig, num_classes: int):
    """
    Build ablation model variant.

    Args:
        variant: one of "rgb_only", "thermal_only", "early_fusion",
                 "dual_no_attn", "simplified_cmafm", "full" (uses model.py)
    """
    if variant == "rgb_only":
        model = SingleModalDetector(cfg, num_classes, modality="rgb")
    elif variant == "thermal_only":
        model = SingleModalDetector(cfg, num_classes, modality="thermal")
    elif variant == "early_fusion":
        model = EarlyFusionDetector(cfg, num_classes)
    elif variant == "dual_no_attn":
        model = DualNoCrossAttnDetector(cfg, num_classes)
    elif variant == "simplified_cmafm":
        model = SimplifiedMidLevelFusionDetector(cfg, num_classes)
    elif variant == "full":
        from model import build_model
        return build_model(cfg, num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    total = sum(p.numel() for p in model.parameters())
    print(f"[{variant}] {total / 1e6:.1f}M params")
    return model
