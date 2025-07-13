# import os
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from PIL import Image
#
#
# import os
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
# from PIL import Image
#
# class Ne12K(Dataset):
#     def __init__(self, root_dir, mode="train", val_split=0.15, random_seed=142, transform=None):
#         """
#         Custom Dataset for segmentation.
#
#         Args:
#             root_dir (str): Root directory containing images and masks.
#             mode (str): 'train' or 'val'.
#             val_split (float): Proportion of data for validation.
#             random_seed (int): Seed for reproducibility.
#             transform (callable, optional): Transformations for image/mask.
#         """
#         self.root_dir = root_dir
#         self.mode = mode
#         self.transform = transform
#
#         # Get all file names and pair images with their masks
#         all_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".tif")])
#         self.image_files = [f for f in all_files if not f.endswith("_mask.tif")]
#         self.mask_files = [f.replace(".tif", "_mask.tif") for f in self.image_files]
#
#         # Ensure data integrity: every image has a corresponding mask
#         assert all(os.path.exists(os.path.join(root_dir, mask)) for mask in self.mask_files), \
#             "Some masks are missing for the given images."
#
#         # Split data into train and validation sets
#         np.random.seed(random_seed)
#         indices = np.arange(len(self.image_files))
#         train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=random_seed)
#
#         if mode == "train":
#             self.image_files = [self.image_files[i] for i in train_indices]
#             self.mask_files = [self.mask_files[i] for i in train_indices]
#         elif mode == "val":
#             self.image_files = [self.image_files[i] for i in val_indices]
#             self.mask_files = [self.mask_files[i] for i in val_indices]
#         else:
#             raise ValueError("Mode must be 'train' or 'val'")
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         """
#         Returns a single sample as a dictionary with 'image' and 'label' keys.
#
#         Args:
#             idx (int): Index of the sample.
#
#         Returns:
#             dict: A dictionary with 'image' (PIL.Image) and 'label' (PIL.Image).
#         """
#         # Load image and mask
#         img_path = os.path.join(self.root_dir, self.image_files[idx])
#         mask_path = os.path.join(self.root_dir, self.mask_files[idx])
#         image = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")
#
#         sample = {"image": image, "label": mask}
#
#         # Apply transformations, if specified
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
#
#
# # # Example usage
# # if __name__ == "__main__":
# #     dataset = Ne12K(root_dir="/path/to/data", mode="train")
# #     print(f"Number of samples: {len(dataset)}")
# #     sample = dataset[0]
# #     print(f"Image shape: {sample['image'].size if isinstance(sample['image'], Image.Image) else sample['image'].shape}")
# #     print(f"Mask shape: {sample['mask'].shape}")
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class Ne12K(Dataset):
    def __init__(self, root_dir, mode="train", val_split=0.15, random_seed=142, transform=None):
        """
        Custom Dataset for segmentation, loading only images with non-empty masks.

        Args:
            root_dir (str): Root directory containing images and masks.
            mode (str): 'train' or 'val'.
            val_split (float): Proportion of data for validation.
            random_seed (int): Seed for reproducibility.
            transform (callable, optional): Transformations for image/mask.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        # Get all file names and pair images with their masks
        all_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".tif")])
        image_files = [f for f in all_files if not f.endswith("_mask.tif")]
        mask_files = [f.replace(".tif", "_mask.tif") for f in image_files]

        # Filter only valid samples (non-empty masks)
        valid_images, valid_masks = [], []
        for img, mask in zip(image_files, mask_files):
            mask_path = os.path.join(root_dir, mask)
            if os.path.exists(mask_path):
                mask_image = np.array(Image.open(mask_path).convert("L"))
                if np.any(mask_image > 0):  # Check if mask has non-zero pixels
                    valid_images.append(img)
                    valid_masks.append(mask)

        # Split data into train and validation sets
        np.random.seed(random_seed)
        indices = np.arange(len(valid_images))
        train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=random_seed)

        if mode == "train":
            self.image_files = [valid_images[i] for i in train_indices]
            self.mask_files = [valid_masks[i] for i in train_indices]
        elif mode == "val":
            self.image_files = [valid_images[i] for i in val_indices]
            self.mask_files = [valid_masks[i] for i in val_indices]
        else:
            raise ValueError("Mode must be 'train' or 'val'")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns a single sample as a dictionary with 'image' and 'label' keys.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with 'image' (PIL.Image) and 'label' (PIL.Image).
        """
        # Load image and mask
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        mask_path = os.path.join(self.root_dir, self.mask_files[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        sample = {"image": image, "label": mask}

        # Apply transformations, if specified
        if self.transform:
            sample = self.transform(sample)

        return sample
