"""Swin-Large backbone + ArcFace head
File: models/swin_arcface.py
This module provides a helper to build a Swin-Large feature extractor and an ArcFace head (ArcMarginProduct)
Requires: timm, torch
"""
from typing import Tuple
import math
import torch
import torch.nn as nn
import timm


class ArcMarginProduct(nn.Module):
    """Implement of ArcFace (ArcMarginProduct)
    Reference: https://arxiv.org/abs/1801.07698
    This implementation follows the common practical formulation using cosine and a fixed margin m.
    Forward when training expects (features, labels) and returns logits scaled by s.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.5, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        # features: (N, F)
        # normalize
        normalized_features = nn.functional.normalize(features)
        normalized_weight = nn.functional.normalize(self.weight)
        cosine = nn.functional.linear(normalized_features, normalized_weight)  # (N, out_features)

        if labels is None:
            # inference mode: return cosine scaled
            return cosine * self.s

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits


def get_swin_large_backbone(pretrained: bool = True, img_size: int = 384) -> Tuple[nn.Module, int]:
    """Create a Swin-Large backbone from timm and return (backbone, feat_dim).
    The returned backbone produces a 1-D pooled feature vector per image.
    """
    # model name: swin_large_patch4_window12_384 is common; timm must be recent
    model_name = f"swin_large_patch4_window12_{img_size}"
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
    feat_dim = backbone.num_features
    return backbone, feat_dim


def build_model(num_classes: int, pretrained: bool = True, img_size: int = 384, s: float = 64.0, m: float = 0.5) -> Tuple[nn.Module, nn.Module]:
    """Return (backbone, arcface_head).
    backbone: nn.Module that maps (N,3,H,W) -> (N, feat_dim)
    arcface_head: ArcMarginProduct which expects (features, labels) during training and returns logits.
    """
    backbone, feat_dim = get_swin_large_backbone(pretrained=pretrained, img_size=img_size)
    arc_head = ArcMarginProduct(in_features=feat_dim, out_features=num_classes, s=s, m=m)
    return backbone, arc_head


if __name__ == '__main__':
    # small smoke test
    b, d = get_swin_large_backbone(pretrained=False, img_size=384)
    print('backbone feature dim =', d)
    dummy = torch.randn(2, 3, 384, 384)
    feats = b(dummy)
    print('feats.shape =', feats.shape)
    head = ArcMarginProduct(in_features=d, out_features=1000)
    labels = torch.tensor([1, 2])
    logits = head(feats, labels)
    print('logits.shape =', logits.shape)
