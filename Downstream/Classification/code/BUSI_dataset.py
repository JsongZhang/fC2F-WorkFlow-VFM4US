import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        自定义数据集类。
        Args:
            image_paths (list): 图像文件的路径列表。
            labels (list): 与图像路径对应的标签列表。
            transform (callable, optional): 应用于样本的图像变换。
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')  # 确保图像为RGB格式
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, torch.tensor(label)


# 获取图像路径和标签
root_dir = '/SZU_DATA/us-vfm/downstream_task/clc/data/BUSI_clc'  # 数据集根目录
all_image_paths = []
all_labels = []

# 获取类别信息及标签映射
classes = sorted(os.listdir(root_dir))
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

for cls_name in classes:
    cls_folder = os.path.join(root_dir, cls_name)
    for img_name in os.listdir(cls_folder):
        if img_name.endswith(('.jpg', '.png', '.bmp')):
            all_image_paths.append(os.path.join(cls_folder, img_name))
            all_labels.append(class_to_idx[cls_name])

# 使用 train_test_split 分割数据集
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths, all_labels, test_size=0.15, stratify=all_labels, random_state=142
)

# 定义数据变换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 实例化数据集
busi_train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
busi_val_dataset = ImageDataset(val_paths, val_labels, transform=val_transform)

# 设置DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 检查数据集划分结果
# print(f"Total dataset size: {len(all_image_paths)}")
# print(f"Training dataset size: {len(busi_train_dataset)}")
# print(f"Validation dataset size: {len(busi_val_dataset)}")