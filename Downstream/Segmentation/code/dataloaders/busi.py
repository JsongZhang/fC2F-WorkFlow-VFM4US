import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class BBU(Dataset):
    def __init__(self, root_dir, mode="train", val_split=0.15, random_seed=142, transform=None):
        """
        Custom Dataset for segmentation, handling one-to-many mask relationships.

        Args:
            root_dir (str): Root directory containing 'images' and 'masks' folders.
            mode (str): 'train' or 'val'.
            val_split (float): Proportion of data for validation.
            random_seed (int): Seed for reproducibility.
            transform (callable, optional): Transformations for image/mask.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        # Prepare paths for images and masks
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")

        # Collect image-mask pairs
        self.valid_images, self.valid_masks = self._collect_image_mask_pairs()

        # Split data into train and validation sets
        np.random.seed(random_seed)
        indices = np.arange(len(self.valid_images))
        train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=random_seed)

        if mode == "train":
            self.image_files = [self.valid_images[i] for i in train_indices]
            self.mask_files = [self.valid_masks[i] for i in train_indices]
        elif mode == "val":
            self.image_files = [self.valid_images[i] for i in val_indices]
            self.mask_files = [self.valid_masks[i] for i in val_indices]
        else:
            raise ValueError("Mode must be 'train' or 'val'")

    def _collect_image_mask_pairs(self):
        """
        Collect valid image-mask pairs based on the folder structure and filenames.

        Returns:
            valid_images: List of image file paths.
            valid_masks: List of corresponding mask file paths (as lists for one-to-many relationship).
        """
        valid_images = []
        valid_masks = []

        # Iterate through `benign` and `malignant` categories
        for category in ["benign", "malignant"]:
            category_image_dir = os.path.join(self.image_dir, category)
            category_mask_dir = os.path.join(self.mask_dir, category)

            images = sorted([f for f in os.listdir(category_image_dir) if f.endswith(".png")])
            masks = sorted([f for f in os.listdir(category_mask_dir) if f.endswith(".png")])

            for img in images:
                img_name = img.split(".")[0]  # Extract the base name (e.g., "1" from "1.png")
                mask_files = [os.path.join(category_mask_dir, m) for m in masks if m.startswith(img_name + "_")]

                # Only include images with at least one mask
                if len(mask_files) > 0:
                    valid_images.append(os.path.join(category_image_dir, img))
                    valid_masks.append(mask_files)

        return valid_images, valid_masks

    def __len__(self):
        return len(self.image_files)

    # def __getitem__(self, idx):
    #     """
    #     Returns a single sample as a dictionary with 'image' and 'label' keys.
    #
    #     Args:
    #         idx (int): Index of the sample.
    #
    #     Returns:
    #         dict: A dictionary with 'image' (PIL.Image) and 'label' (PIL.Image).
    #     """
    #     # Load image
    #     img_path = self.image_files[idx]
    #     image = Image.open(img_path).convert("RGB")
    #
    #     # Combine masks into a single mask
    #     mask_paths = self.mask_files[idx]
    #     combined_mask = None
    #     for mask_path in mask_paths:
    #         mask = np.array(Image.open(mask_path).convert("L"))
    #         if combined_mask is None:
    #             combined_mask = mask
    #         else:
    #             combined_mask = np.maximum(combined_mask, mask)  # Combine masks by taking the maximum
    #
    #     combined_mask = Image.fromarray(combined_mask)
    #
    #     sample = {"image": image, "label": combined_mask}
    #
    #     # Apply transformations, if specified
    #     if self.transform:
    #         sample = self.transform(sample)
    #
    #     return sample
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        # Combine masks into a single mask with category distinction
        mask_paths = self.mask_files[idx]
        combined_mask = None
        for mask_path in mask_paths:
            mask = np.array(Image.open(mask_path).convert("L"))

            # 根据文件夹名称区分类别，假设良性=1，恶性=2
            if "benign" in mask_path:
                mask = (mask > 0) * 1  # 将良性掩码像素值设置为1
            elif "malignant" in mask_path:
                mask = (mask > 0) * 2  # 将恶性掩码像素值设置为2

            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = np.maximum(combined_mask, mask)  # 合并掩码

        combined_mask = Image.fromarray(combined_mask.astype(np.uint8))


        sample = {"image": image, "label": combined_mask}

        # Apply transformations, if specified
        if self.transform:
            sample = self.transform(sample)

        return sample

