# 适用于PyTorch的通用数据集类
import os
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class TN3KSegmentationDataset(Dataset):
    def __init__(self, root_dir, mode, fold=0, transform=None):
        """
        root_dir: 数据集的根目录
        mode: 'train'，'val' 或 'test'
        fold: 使用的数据fold编号
        transform: 应用于样本的转换
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        # 加载分割文件
        if mode in ['train', 'val']:
            with open(os.path.join(root_dir, f'tn3k-trainval-fold{fold}.json'), 'r') as file:
                trainval = json.load(file)
            if mode == 'train':
                self.ids = trainval['train']
            else:
                self.ids = trainval['val']
        else:
            self.ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root_dir, 'test-image'))]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_path = os.path.join(self.root_dir, f'{self.mode}-image', f'{image_id}.png')
        label_path = os.path.join(self.root_dir, f'{self.mode}-mask', f'{image_id}.png')

        # 加载图像和标签
        image = Image.open(image_path).convert('RGB')  # 确保为RGB格式，因为ViT通常处理彩色图像
        label = Image.open(label_path).convert('L')  # 加载标签掩码为灰度图像

        # 应用转换
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


# 转换函数，包括调整图像尺寸和转换为张量
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像尺寸为256x256
    transforms.ToTensor(),  # 将图像和标签转换为张量
])

# 实例化数据集
dataset = TN3KSegmentationDataset(root_dir='/path/to/tn3k', mode='train', fold=0, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 使用loader在训练循环中
for images, labels in loader:
    # 在这里加入模型训练代码
    pass
