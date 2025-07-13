# utils/test.py

import torch
from models.segmentation_model import ViT_UperNet_Segmentation
from torch.utils.data import DataLoader


def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            # 可以进行进一步的评估，如计算mIoU等

    return outputs


def main():
    model = ViT_UperNet_Segmentation(num_classes=19, pretrained=True)
    model = model.to('cuda')

    # 测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 评估
    outputs = test(model, test_loader, 'cuda')
    print(outputs)


if __name__ == "__main__":
    main()
