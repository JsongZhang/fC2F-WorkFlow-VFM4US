import torch
import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np
from sklearn.model_selection import train_test_split


def make_dataset(image_dir, mask_dir, indices):
    imgs = []
    img_names = os.listdir(image_dir)
    # 修正排序方式，假设文件名格式为 P0123_IMG4.jpg
    img_names = sorted(img_names, key=lambda i: (int(i.split('_')[0][1:]), int(i.split('_')[1][3:].split('.')[0])))

    for i in indices:
        img_name = img_names[i]
        img = os.path.join(image_dir, img_name)
        mask_name = img_name.split('.')[0] + '.npy'  # Assuming the mask file has the same base name but .npy extension
        mask = os.path.join(mask_dir, mask_name)
        imgs.append((img, mask, 0))
    return imgs



class MyDataset(data.Dataset):
    def __init__(self, root_dir, mode, transform=None, return_size=False, visual=False):
        self.transform = transform
        self.return_size = return_size
        self.visual = visual
        image_dir = os.path.join(root_dir, 'IMAGES/')
        mask_dir = os.path.join(root_dir, 'ARRAY_FORMAT/')

        # Load all indices and split
        total_indices = range(len(os.listdir(image_dir)))
        train_indices, val_indices = train_test_split(list(total_indices), test_size=0.2, random_state=22)
        short_train_indices, non_sense_train_indices = train_test_split(list(train_indices), test_size=0.2, random_state=22)
        if mode == 'train':
            self.imgs = make_dataset(image_dir, mask_dir, train_indices)
        elif mode == 'val':
            self.imgs = make_dataset(image_dir, mask_dir, val_indices)

    def __getitem__(self, item):
        image_path, mask_path, _ = self.imgs[item]  # 这里使用 _ 忽略第三个元素

        image = Image.open(image_path).convert('RGB')
        mask_data = np.load(mask_path, allow_pickle=True).item()
        masks = mask_data['structures']

        # Initialize a mask array with zeros where each pixel's value will represent a structure
        full_mask = np.zeros_like(list(masks.values())[0], dtype=np.uint8)

        # Assign each structure a unique label
        structures = ['artery', 'liver', 'stomach', 'vein']
        for idx, structure in enumerate(structures):
            full_mask[masks[structure] > 0] = idx + 1  # Assuming non-zero values are the actual mask

        full_mask = Image.fromarray(full_mask)



        sample = {'image': image, 'label': full_mask}

        if self.transform and self.visual:
            image = self.transform(image)
            label = torch.tensor(np.array(full_mask), dtype=torch.long)
            label = self.transform(label)
            sample = {'image': image, 'label': label}
            return sample

        else:
            sample = self.transform(sample)

        if self.return_size:
            w, h = image.size
            sample['size'] = torch.tensor((h, w))

        return sample

    def __len__(self):
        return len(self.imgs)

# 使用示例
# dataset = MyDataset(root_dir='path/to/dataset', mode='train')
# loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
