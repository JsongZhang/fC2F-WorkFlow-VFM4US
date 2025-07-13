# models/segmentation_model.py

import torch
import torch.nn as nn
from .vit_backbone import ViTBackbone
from .upernet_head import UPerNetHead


class ViT_UperNet_Segmentation(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ViT_UperNet_Segmentation, self).__init__()
        # ViT作为backbone
        self.backbone = ViTBackbone(pretrained=pretrained)

        # 使用UperNet作为解码头
        self.decode_head = UPerNetHead(in_channels=768, num_classes=num_classes)

    def forward(self, x):
        # 提取特征
        features = self.backbone(x)

        # 通过解码头进行分割
        segmentation_map = self.decode_head(features)

        return segmentation_map
