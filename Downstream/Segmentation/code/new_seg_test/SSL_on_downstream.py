import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from tn3k_dataset import TN3KDataset  # Assuming TN3KDataset is defined in a separate dataset.py file

# U-Net Decoder
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UNetViT(nn.Module):
    def __init__(self, encoder, img_size=224, patch_size=16, num_channels=768):
        super(UNetViT, self).__init__()
        self.encoder = encoder
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        # Calculate spatial dimensions
        self.spatial_dim = img_size // patch_size

        # Projection from ViT output to spatial feature map
        self.proj = nn.Linear(self.num_channels, self.num_channels)
        self.reshape_to_map = nn.Conv2d(self.num_channels, 512, kernel_size=1)

        # Decoders
        self.decoder1 = UNetDecoder(512, 256)
        self.decoder2 = UNetDecoder(256, 128)
        self.decoder3 = UNetDecoder(128, 64)
        self.decoder4 = UNetDecoder(64, 32)

        # Final segmentation head
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)  # Binary segmentation

    def forward(self, x):
        # ViT encoding
        x = self.encoder(x)
        x = x[:, 1:, :]  # Remove class token
        x = self.proj(x)  # Linear projection
        x = x.transpose(1, 2).reshape(-1, self.num_channels, self.spatial_dim, self.spatial_dim)

        # Convert to feature map
        x = self.reshape_to_map(x)

        # Decoding
        x = self.decoder1(x, None)  # No skip connection for simplicity
        x = self.decoder2(x, None)
        x = self.decoder3(x, None)
        x = self.decoder4(x, None)

        x = self.final_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, encoder):
        super(UNet, self).__init__()
        self.encoder = encoder

        self.decoder1 = UNetDecoder(512, 256)
        self.decoder2 = UNetDecoder(256, 128)
        self.decoder3 = UNetDecoder(128, 64)
        self.decoder4 = UNetDecoder(64, 32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)  # Binary segmentation

    def forward(self, x):
        enc_features = []
        for layer in self.encoder.children():
            x = layer(x)
            enc_features.append(x)

        x = enc_features[-1]
        x = self.decoder1(x, enc_features[-2])
        x = self.decoder2(x, enc_features[-3])
        x = self.decoder3(x, enc_features[-4])
        x = self.decoder4(x, enc_features[-5])

        x = self.final_conv(x)
        return x

# Define classification and segmentation models
def get_model(backbone_name, task, pretrained_weights):
    if backbone_name == 'resnet18':
        backbone = timm.create_model('resnet18', pretrained=False)
    elif backbone_name == 'vit-b':
        backbone = timm.create_model('vit_base_patch16_224', pretrained=False)
    else:
        raise ValueError("Unsupported backbone")

    # Load pretrained weights
    if pretrained_weights:
        state_dict = torch.load(pretrained_weights)
        backbone.load_state_dict(state_dict, strict=False)

    if task == 'classification':
        model = nn.Sequential(
            backbone,
            nn.Linear(backbone.get_classifier().in_features, num_classes)
        )
    elif task == 'segmentation' :
        if backbone_name == 'resnet18':
            model = UNet(backbone)
        else:
            model = UNetViT(backbone)
    else:
        raise ValueError("Unsupported task")

    return model

# Define loss functions
classification_criterion = nn.CrossEntropyLoss()

# Example Dice Loss Implementation
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
        return 1 - dice

segmentation_criterion = DiceLoss()

# Training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Configuration
num_classes = 2  # Example for classification
num_epochs = 10
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create data loaders
root_dir = "/SZU_DATA/us-vfm/downstream_task/seg/data/tn3k"
train_dataset = TN3KDataset(mode='train', transform=data_transforms)
val_dataset = TN3KDataset(mode='val', transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)



# Choose backbone, task, and pretrained weights
backbone_name = 'vit-b'
task = 'segmentation'  # or 'classification'
Pretrained = True
pretrained_weights = '/SZU_DATA/us-vfm/upstream_task/USFM-MIA2024/USFM_latest.pth'

model = get_model(backbone_name, task, pretrained_weights).to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)

# Training loop
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, segmentation_criterion if task == 'segmentation' else classification_criterion, device)
    val_loss = validate(model, val_loader, segmentation_criterion if task == 'segmentation' else classification_criterion, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
