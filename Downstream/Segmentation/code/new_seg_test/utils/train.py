# utils/train.py

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models.segmentation_model import ViT_UperNet_Segmentation
from mmcv.runner import build_optimizer, load_checkpoint


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    return loss.item()


def main():
    # 初始化模型
    model = ViT_UperNet_Segmentation(num_classes=19, pretrained=True)
    model = model.to('cuda')

    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练过程
    for epoch in range(50):
        loss = train(model, train_loader, optimizer, criterion, 'cuda')
        print(f"Epoch {epoch}, Loss: {loss}")


if __name__ == "__main__":
    main()
