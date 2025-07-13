import argparse
import glob
import os
import random
import socket
import time
from datetime import datetime
import math
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
# PyTorch includes
import torch
import torch.optim as optim
# Tensorboard include
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
# Dataloaders includes
from dataloaders import tn3k, tg3k, tatn, tn3k_point, Fetal_AP, Nerve_12k, busi, c_tus
from dataloaders import custom_transforms as trforms
from dataloaders import utils
# zhi biao
from dataloaders.utils import get_dice
from dataloaders.utils import cal_HD_2
import torch.nn.functional as F
import timm
from tqdm import tqdm
from torch import nn
from sklearn.metrics import jaccard_score
from utils import soft_dice
#在这个代码中 只计算前景对应分割的贡献（smp.Unet也是这么计算的，所以要统一，背景为0 计算也没有意义）
#但是在这篇代码的定义中背景被考虑进预测中了，因此传递类别个数时，要+1
def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.6,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="mean")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-model_name', type=str, default='use_ctus-usfm-vit-unet357_11')
    parser.add_argument('-criterion', type=str, default='Dice')
    parser.add_argument('-pretrained', type=str, default='/SZU_DATA/us-vfm/upstream_task/mae/mae/mae-main/us_120k_380k_25per_output_dir/checkpoint-199.pth')
    parser.add_argument('-num_classes', type=int, default=2)  # Updated for foreground and background
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-output_stride', type=int, default=16)
    parser.add_argument('-dataset', type=str, default='ctus')
    parser.add_argument('-fold', type=str, default='0')
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-nepochs', type=int, default=200)
    parser.add_argument('-resume_epoch', type=int, default=0)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--warm-up-epochs", default=5, type=int)
    parser.add_argument('-save_every', type=int, default=10)
    parser.add_argument('-log_every', type=int, default=40)
    parser.add_argument('-load_path', type=str, default='')
    parser.add_argument('-run_id', type=int, default=-1)
    parser.add_argument('-use_eval', type=int, default=1)
    parser.add_argument('-use_test', type=int, default=1)
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1234)

Pretrained = True

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.upconv(x)
        if skip is not None:
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((x, skip), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UNetViT(nn.Module):
    def __init__(self, encoder, target_layers, img_size=224, patch_size=16, num_classes=2):
        super(UNetViT, self).__init__()
        self.encoder = encoder
        self.target_layers = target_layers
        self.hooks = []
        self.features = [None] * len(target_layers)

        for i, layer_idx in enumerate(target_layers):
            self.hooks.append(self.encoder.blocks[layer_idx].register_forward_hook(self._hook_fn(i)))

        self.decoder4 = UNetDecoder(768, 512, 768)
        self.decoder3 = UNetDecoder(512, 256, 768)
        self.decoder2 = UNetDecoder(256, 128, 768)
        self.decoder1 = UNetDecoder(128, 64, 0)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _hook_fn(self, idx):
        def hook(module, input, output):
            self.features[idx] = output
        return hook

    def forward(self, x):
        _ = self.encoder(x)
        processed_features = []
        for i, feat in enumerate(self.features):
            if feat is not None:
                num_patches = feat.size(1) - 1
                spatial_dim = int(math.sqrt(num_patches))
                if spatial_dim * spatial_dim != num_patches:
                    raise ValueError(f"Feature {i} has invalid shape: {feat.shape}")
                processed_features.append(feat[:, 1:].transpose(1, 2).reshape(-1, 768, spatial_dim, spatial_dim))
            else:
                raise ValueError(f"Feature {i} is None. Check your forward hook registration.")

        f3, f5, f7, f11 = processed_features
        d4 = self.decoder4(f11, f7)
        d3 = self.decoder3(d4, f5)
        d2 = self.decoder2(d3, f3)
        d1 = self.decoder1(d2, None)
        return self.final_conv(d1)

def compute_foreground_loss_and_metrics(predictions, labels_one_hot, criterion):
    """
    Compute loss and metrics for the foreground only (class 1).
    """
    # Get foreground predictions and labels
    foreground_predictions = predictions[:, 1:, :, :]
    foreground_labels = labels_one_hot[:, 1:, :, :]

    # Compute loss
    loss = criterion(foreground_predictions, foreground_labels)

    # Flatten tensors for IOU computation
    pred_flat = (foreground_predictions > 0.5).long().flatten()
    label_flat = foreground_labels.long().flatten()

    # Compute Dice and IOU
    dice = get_dice(foreground_predictions, foreground_labels)
    iou = jaccard_score(label_flat.cpu().numpy(), pred_flat.cpu().numpy(), average='binary')

    return loss, dice, iou


save_dir = './CTUS_weight'
model_name = 'MAE25per'
save_path = os.path.join(save_dir, model_name + '_best' + '.pth')
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    encoder = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=0,
        global_pool=''
    )

    net = UNetViT(encoder, target_layers=[3, 5, 7, 11], img_size=args.input_size, patch_size=16, num_classes=args.num_classes)

    if args.pretrained and os.path.isfile(args.pretrained) and Pretrained:
        print(f"=> loading checkpoint '{args.pretrained}'")
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        state_dict = checkpoint.get('model', checkpoint)
        new_state_dict = {}

        # for k in list(state_dict.keys()):
        #     if k.startswith('module.base_encoder'):
        #         new_key = k.replace('module.base_encoder.', '')
        #         new_state_dict[new_key] = state_dict[k]

        net.encoder.load_state_dict(new_state_dict, strict=False)
        print('Loaded pretrained(encoder) ----->', args.pretrained)
    else:
        print('There is no pretrained for encoder')

    net.cuda()

    criterion = soft_dice
    optimizer = optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    composed_transforms_tr = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.RandomHorizontalFlip(),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()
    ])

    # train_data = busi.BBU(root_dir='/SZU_DATA/us-vfm/downstream_task/seg/data/BUSI', mode='train', transform=composed_transforms_tr)
    # val_data = busi.BBU(root_dir='/SZU_DATA/us-vfm/downstream_task/seg/data/BUSI', mode='val', transform=composed_transforms_ts)
    # train_data = Fetal_AP.MyDataset(root_dir='/SZU_DATA/us-vfm/downstream_task/seg/data/Fetal/Fetal_Ab/Featl_Ab',
    #                                 mode='train', transform=composed_transforms_tr)
    # val_data = Fetal_AP.MyDataset(root_dir='/SZU_DATA/us-vfm/downstream_task/seg/data/Fetal/Fetal_Ab/Featl_Ab',
    #                               mode='val', transform=composed_transforms_ts)
    train_data = c_tus.CTrusDataset(
        root_dir='/SZU_DATA/us-vfm/downstream_task/seg/data/c-trus-main',
        mode='train', transform=composed_transforms_tr)
    val_data = c_tus.CTrusDataset(
        root_dir='/SZU_DATA/us-vfm/downstream_task/seg/data/c-trus-main',
        mode='val', transform=composed_transforms_ts)

    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=2)
    best_dice = 0.0
    for epoch in range(args.nepochs):
        net.train()
        for sample_batched in tqdm(trainloader):
            inputs, labels = sample_batched['image'].cuda(), sample_batched['label'].cuda()
            outputs = net(inputs)
            outputs = F.interpolate(outputs, size=labels.shape[2:], mode='bilinear', align_corners=False)

            labels_one_hot = F.one_hot(labels.squeeze(1).long(), args.num_classes).permute(0, 3, 1, 2).float()

            loss, dice, iou = compute_foreground_loss_and_metrics(outputs, labels_one_hot, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Dice={dice:.4f}, IoU={iou:.4f}")

        # Evaluate on validation set
        net.eval()
        with torch.no_grad():

            total_loss, total_dice, total_iou, count = 0.0, 0.0, 0.0, 0
            for sample_batched in tqdm(testloader):
                inputs, labels = sample_batched['image'].cuda(), sample_batched['label'].cuda()
                outputs = net(inputs)
                outputs = F.interpolate(outputs, size=labels.shape[2:], mode='bilinear', align_corners=False)

                labels_one_hot = F.one_hot(labels.squeeze(1).long(), args.num_classes).permute(0, 3, 1, 2).float()

                loss, dice, iou = compute_foreground_loss_and_metrics(outputs, labels_one_hot, criterion)

                total_loss += loss.item()
                total_dice += dice
                total_iou += iou
                count += 1
            if total_dice/count > best_dice:
                best_dice = total_dice/count
                torch.save(net.state_dict(), save_path)
            print(f"Best Dice is:{best_dice:.4f}")
            print(f"Validation - Epoch {epoch}: Loss={total_loss / count:.4f}, Dice={total_dice / count:.4f}, IoU={total_iou / count:.4f}")

if __name__ == "__main__":
    args = get_arguments()
    main(args)
