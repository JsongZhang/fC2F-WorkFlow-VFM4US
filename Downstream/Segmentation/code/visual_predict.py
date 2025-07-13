import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dataloaders.Fetal_AP import MyDataset
from dataloaders.c_tus import CTrusDataset
from dataloaders.Nerve_12k import Ne12K
from dataloaders.utils import get_iou, get_dice, cal_HD_2, get_prec_recall, get_mae
import segmentation_models_pytorch as smp
from dataloaders import custom_transforms as trforms
import torch.nn as nn
import math
#
# def visualize_predictions(model, dataloader, device, save_dir, num_classes=5):
#     model.eval()
#     os.makedirs(save_dir, exist_ok=True)
#
#     jac, dsc, HD_2, total_mae = 0, 0, 0, 0
#     prec_lists, recall_lists = [], []
#     count, cnt = 0, 0
#
#     with torch.no_grad():
#         for i, sample_batched in enumerate(dataloader):
#             inputs, labels = sample_batched['image'].to(device), sample_batched['label'].to(device)
#
#             outputs = model(inputs)
#             outputs = F.interpolate(outputs, size=labels.shape[2:], mode='bilinear', align_corners=False)
#
#             # One-hot encode labels to match the model output's shape
#             # labels_one_hot = F.one_hot(labels.squeeze(1).long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
#             # One-hot encode labels to match the model output's shape
#             labels_one_hot = F.one_hot(labels.squeeze(1).long(), num_classes=2).permute(0, 3, 1, 2).float()
#
#             labels_one_hot = F.interpolate(labels_one_hot, size=outputs.shape[2:], mode='nearest')
#
#
#             # Compute metrics
#             predictions = torch.softmax(outputs, dim=1)
#             predictions_single_channel = torch.argmax(predictions, dim=1, keepdim=True).float()
#             labels_single_channel = torch.argmax(labels_one_hot, dim=1, keepdim=True).float()
#             # predictions_single_channel = predictions_single_channel / (num_classes - 1)
#             # labels_single_channel = labels_single_channel / (num_classes - 1)
#
#             print("Predictions shape:", predictions.shape)
#             print("Labels one hot shape:", labels_one_hot.shape)
#
#             # Compute IOU after flattening
#             jac += get_iou(predictions_single_channel, labels_single_channel)
#             # jac += get_iou(predictions, labels_one_hot)  resnet18可用
#             count += 1
#             total_mae += get_mae(predictions, labels_one_hot) * predictions.size(0)
#             prec_list, recall_list = get_prec_recall(predictions_single_channel, labels_single_channel)
#             prec_lists.extend(prec_list)
#             recall_lists.extend(recall_list)
#             cnt += predictions.size(0)
#             dsc += get_dice(predictions, labels_one_hot)
#             HD_2 += cal_HD_2(predictions, labels_one_hot)
#
#             # Visualize and save predictions
#             pred_vis = predictions_single_channel[0].cpu().numpy().squeeze() * 255  # Scale to [0, 255]
#             label_vis = labels_single_channel[0].cpu().numpy().squeeze() * 255  # Scale to [0, 255]
#             input_image = inputs[0].cpu().numpy().transpose(1, 2, 0)  # Transpose for matplotlib (H, W, C)
#
#             fig, ax = plt.subplots(1, 3, figsize=(12, 4))
#             ax[0].imshow(input_image.astype(np.uint8))
#             ax[0].set_title("Input Image")
#             ax[1].imshow(label_vis, cmap='gray')
#             ax[1].set_title("Ground Truth")
#             ax[2].imshow(pred_vis, cmap='gray')
#             ax[2].set_title("Prediction")
#             plt.suptitle(f"Sample {i}")
#             plt.savefig(os.path.join(save_dir, f"prediction_{i}.png"))
#             plt.close(fig)
#
#     # Calculate and print metrics
#     mean_iou = jac / count
#     mean_dice = dsc / count
#     mean_hd2 = HD_2 / count
#     mean_mae = total_mae / cnt
#     mean_prec = sum(prec_lists) / len(prec_lists) if len(prec_lists) > 0 else 0
#     mean_recall = sum(recall_lists) / len(recall_lists) if len(recall_lists) > 0 else 0
#
#     print(f"Mean IoU: {mean_iou:.4f}")
#     print(f"Mean Dice: {mean_dice:.4f}")
#     print(f"Mean HD_2: {mean_hd2:.4f}")
#     print(f"Mean MAE: {mean_mae:.4f}")
#     print(f"Mean Precision: {mean_prec:.4f}")
#     print(f"Mean Recall: {mean_recall:.4f}")

def visualize_predictions(model, dataloader, device, save_dir, num_classes=2):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    jac, dsc, HD_2, total_mae = 0, 0, 0, 0
    prec_lists, recall_lists = [], []
    count, cnt = 0, 0

    with torch.no_grad():
        for i, sample_batched in enumerate(dataloader):
            inputs, labels = sample_batched['image'].to(device), sample_batched['label'].to(device)

            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=labels.shape[2:], mode='bilinear', align_corners=False)

            # One-hot encode labels to match the model output's shape
            labels_one_hot = F.one_hot(labels.squeeze(1).long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
            labels_one_hot = F.interpolate(labels_one_hot, size=outputs.shape[2:], mode='nearest')

            # Compute predictions
            predictions = torch.softmax(outputs, dim=1)
            predictions_single_channel = torch.argmax(predictions, dim=1, keepdim=True).float()
            labels_single_channel = torch.argmax(labels_one_hot, dim=1, keepdim=True).float()

            # Compute metrics only for foreground
            foreground_predictions = (predictions_single_channel == 1).float()
            foreground_labels = (labels_single_channel == 1).float()

            jac += get_iou(foreground_predictions, foreground_labels)
            dsc += get_dice(foreground_predictions, foreground_labels)
            count += 1

            total_mae += get_mae(foreground_predictions, foreground_labels) * predictions.size(0)
            prec_list, recall_list = get_prec_recall(foreground_predictions, foreground_labels)
            prec_lists.extend(prec_list)
            recall_lists.extend(recall_list)
            cnt += predictions.size(0)

            # Visualize and save predictions
            pred_vis = predictions_single_channel[0].cpu().numpy().squeeze() * 255  # Scale to [0, 255]
            label_vis = labels_single_channel[0].cpu().numpy().squeeze() * 255  # Scale to [0, 255]
            input_image = inputs[0].cpu().numpy().transpose(1, 2, 0)  # Transpose for matplotlib (H, W, C)

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(input_image.astype(np.uint8))
            ax[0].set_title("Input Image")
            ax[1].imshow(label_vis, cmap='gray')
            ax[1].set_title("Ground Truth")
            ax[2].imshow(pred_vis, cmap='gray')
            ax[2].set_title("Prediction")
            plt.suptitle(f"Sample {i}")
            plt.savefig(os.path.join(save_dir, f"prediction_{i}.png"))
            plt.close(fig)

    # Calculate and print metrics
    mean_iou = jac / count
    mean_dice = dsc / count
    mean_mae = total_mae / cnt
    mean_prec = sum(prec_lists) / len(prec_lists) if len(prec_lists) > 0 else 0
    mean_recall = sum(recall_lists) / len(recall_lists) if len(recall_lists) > 0 else 0

    print(f"Mean IoU (Foreground): {mean_iou:.4f}")
    print(f"Mean Dice (Foreground): {mean_dice:.4f}")
    print(f"Mean MAE: {mean_mae:.4f}")
    print(f"Mean Precision: {mean_prec:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")

# -------------------- Modified UNet Decoder -------------------- #
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.upconv(x)  # Upsample
        if skip is not None:
            # Match spatial dimensions and concatenate
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((x, skip), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# -------------------- Modified ViT-UNet -------------------- #
class UNetViT(nn.Module):
    def __init__(self, encoder, target_layers, img_size=224, patch_size=16, num_classes=1):
        super(UNetViT, self).__init__()
        self.encoder = encoder
        self.target_layers = target_layers
        self.hooks = []
        self.features = [None] * len(target_layers)

        # 注册 forward hooks
        for i, layer_idx in enumerate(target_layers):
            self.hooks.append(self.encoder.blocks[layer_idx].register_forward_hook(self._hook_fn(i)))

        # 解码器层（skip_channels 是跳跃连接的通道数）
        self.decoder4 = UNetDecoder(768, 512, 768)  # f11 + f7 (768 skip channels)
        self.decoder3 = UNetDecoder(512, 256, 768)  # d4 + f5 (768 skip channels)
        self.decoder2 = UNetDecoder(256, 128, 768)  # d3 + f3 (768 skip channels)
        self.decoder1 = UNetDecoder(128, 64, 0)     # d2 (no skip)

        # 最终的分割头，仅输出前景概率
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _hook_fn(self, idx):
        def hook(module, input, output):
            self.features[idx] = output
        return hook

    def forward(self, x):
        _ = self.encoder(x)  # 编码器的前向传播

        # 提取目标层的特征
        processed_features = []
        for i, feat in enumerate(self.features):
            if feat is not None:
                num_patches = feat.size(1) - 1  # 去掉 class token
                spatial_dim = int(math.sqrt(num_patches))
                if spatial_dim * spatial_dim != num_patches:
                    raise ValueError(f"Feature {i} has invalid shape: {feat.shape}")
                processed_features.append(feat[:, 1:].transpose(1, 2).reshape(-1, 768, spatial_dim, spatial_dim))
            else:
                raise ValueError(f"Feature {i} is None. Check your forward hook registration.")

        f3, f5, f7, f11 = processed_features

        # 解码器与跳跃连接
        d4 = self.decoder4(f11, f7)  # Decoder 4: f11 (upsampled) + f7 (skip connection)
        d3 = self.decoder3(d4, f5)  # Decoder 3: d4 (upsampled) + f5 (skip connection)
        d2 = self.decoder2(d3, f3)  # Decoder 2: d3 (upsampled) + f3 (skip connection)
        d1 = self.decoder1(d2, None)  # Decoder 1: d2 (upsampled, no skip)

        # 返回前景概率
        return self.final_conv(d1)

import timm
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    # model = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=5)
    # 创建 ViT 编码器时移除分类头
    encoder = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=0,  # 禁用分类头
        global_pool=''  # 禁用全局池化
    )
    model = UNetViT(encoder, target_layers=[3, 5, 7, 11], img_size=224, patch_size=16, num_classes=2)
    model.load_state_dict(torch.load("/Volumes/T7/jiansongzhang/SZU_DATA/us-vfm/downstream_task/seg/code/run/run_10/use_ctus-usfm-vit-unet357_11_best.pth"))  # Replace with the actual path
    model.to(device)

    # Dataloader
    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(224, 224)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])
    # dataset = CTrusDataset(root_dir='/SZU_DATA/us-vfm/downstream_task/seg/data/c-trus-main', mode='val', transform=composed_transforms_ts)  # Replace with actual path
    dataset = Ne12K(root_dir='/Volumes/T7/jiansongzhang/SZU_DATA/us-vfm/downstream_task/seg/data/Nerve/ultrasound-nerve-segmentation/train', mode='val', transform=composed_transforms_ts)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Directory to save visualizations
    save_dir = "/Volumes/T7/jiansongzhang/SZU_DATA/us-vfm/downstream_task/seg/code/20250619_seg_sample/Nerve/USFM"  # Replace with actual path

    # Run visualization
    visualize_predictions(model, dataloader, device, save_dir)


if __name__ == "__main__":
    main()
