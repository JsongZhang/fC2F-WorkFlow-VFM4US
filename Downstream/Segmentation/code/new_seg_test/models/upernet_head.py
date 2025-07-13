# models/upernet_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class UPerNetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UPerNetHead, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x
