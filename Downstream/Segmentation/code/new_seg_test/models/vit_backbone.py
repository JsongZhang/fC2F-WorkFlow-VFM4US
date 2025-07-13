# models/vit_backbone.py

import torch
import torch.nn as nn
from torchvision.models import vision_transformer as vit

class ViTBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ViTBackbone, self).__init__()
        # 加载预训练权重的ViT模型
        self.vit = vit.vit_b_16(pretrained=pretrained)
        # 移除分类头（我们会使用自己的解码头）
        self.vit.heads = nn.Identity()

    def forward(self, x):
        # 通过ViT backbone提取特征
        return self.vit(x)
