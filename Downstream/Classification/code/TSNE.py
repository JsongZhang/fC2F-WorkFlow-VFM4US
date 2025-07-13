import os
import random
import shutil
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 配置路径和参数
coarse_dir = '/SZU_DATA/us-vfm/data/US_510K/US-RIM380K/us_380k_train/data'
fine_dir = '/SZU_DATA/us-vfm/data/US_510K/US-MTD120K/us_120k_train/data'
output_dir = './sampled_data'
coarse_output_dir = os.path.join(output_dir, 'coarse')
fine_output_dir = os.path.join(output_dir, 'fine')
sample_per_class = 150  # 每类采样图像数量
img_size = 128  # 调整图像大小便于计算

# 创建目标文件夹
os.makedirs(coarse_output_dir, exist_ok=True)
os.makedirs(fine_output_dir, exist_ok=True)

# 图像预处理函数
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# 函数：从文件夹采样图像并拷贝到指定目录
def sample_images(src_dir, dst_dir, sample_count, start_label=0):
    features = []
    labels = []
    label_map = {}
    os.makedirs(dst_dir, exist_ok=True)
    current_label = start_label
    for class_name in os.listdir(src_dir):
        class_dir = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_output_dir = os.path.join(dst_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('png', 'jpg', 'jpeg', 'bmp'))]
        sampled_images = random.sample(images, min(sample_count, len(images)))

        for img_path in sampled_images:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            features.append(img_tensor.flatten().numpy())
            labels.append(current_label)
            label_map[current_label] = class_name

            # 拷贝到目标目录
            shutil.copy(img_path, class_output_dir)

        current_label += 1

    return np.array(features), np.array(labels), label_map, current_label

# 粗粒度采样
print("Processing coarse-grained data...")
coarse_features, coarse_labels, coarse_label_map, next_label = sample_images(coarse_dir, coarse_output_dir, sample_per_class)

# 细粒度采样
print("Processing fine-grained data...")
fine_features, fine_labels, fine_label_map, _ = sample_images(fine_dir, fine_output_dir, sample_per_class, start_label=next_label)

# 合并粗粒度和细粒度特征
features = np.vstack([coarse_features, fine_features])
labels = np.hstack([coarse_labels, fine_labels])
label_map = {**coarse_label_map, **fine_label_map}

# T-SNE 可视化
print("Running T-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(features)

# 可视化
plt.figure(figsize=(12, 8))
unique_labels = np.unique(labels)
colors_fine = plt.cm.rainbow(np.linspace(0, 1, len(np.unique(unique_labels))))

for i,label in unique_labels:
    indices = labels == label

    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label_map[label], alpha=0.6, c=[colors_fine[i]], marker='o')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='medium')
plt.title("T-SNE Visualization of Sampled Data")
plt.xlabel("TSNE Dimension 1")
plt.ylabel("TSNE Dimension 2")
plt.tight_layout()  # 去掉网格线
plt.savefig('./tsne.png', dpi=300)
plt.show()