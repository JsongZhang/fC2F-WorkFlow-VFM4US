import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class CTrusDataset(Dataset):
    def __init__(self, root_dir, mode="train", val_split=0.15, random_seed=142, transform=None):
        """
        Custom Dataset for segmentation, ensuring only images with corresponding valid masks are loaded.

        Args:
            root_dir (str): Root directory containing 'original' (images) and 'labels' (masks).
            mode (str): 'train' or 'val'.
            val_split (float): Proportion of data for validation.
            random_seed (int): Seed for reproducibility.
            transform (callable, optional): Transformations for image/mask.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        # Define image and mask directories
        image_dir = os.path.join(root_dir, "original")
        mask_dir = os.path.join(root_dir, "labels")

        # Collect all images and their corresponding masks
        all_images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        all_masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(".jpg")])

        # Match images with masks and filter invalid masks
        image_files, mask_files = [], []
        for img in all_images:
            mask = img  # Assume the mask filename matches the image filename
            mask_path = os.path.join(mask_dir, mask)
            if os.path.exists(mask_path):
                mask_array = np.array(Image.open(mask_path).convert("L"))
                if np.any(mask_array > 0):  # Check if the mask has non-zero pixels (valid foreground)
                    image_files.append(img)
                    mask_files.append(mask)

        # Split into train and validation sets
        np.random.seed(random_seed)
        indices = np.arange(len(image_files))
        train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=random_seed)

        if mode == "train":
            self.image_files = [image_files[i] for i in train_indices]
            self.mask_files = [mask_files[i] for i in train_indices]
        elif mode == "val":
            self.image_files = [image_files[i] for i in val_indices]
            self.mask_files = [mask_files[i] for i in val_indices]
        else:
            raise ValueError("Mode must be 'train' or 'val'")

        self.image_dir = image_dir
        self.mask_dir = mask_dir

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
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming masks are grayscale

        # Process mask: convert all non-background (values > 0) to 1
        mask = np.array(mask)  # Convert mask to numpy array
        mask[mask <255] = 0  # Set all non-background pixels to 1
        mask = Image.fromarray(mask.astype(np.uint8))  # Convert back to PIL.Image

        sample = {"image": image, "label": mask}

        # Apply transformations, if specified
        if self.transform:
            sample = self.transform(sample)

        return sample

