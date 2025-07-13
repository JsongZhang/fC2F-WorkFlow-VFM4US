import os
from typing import Callable, Optional, Tuple, List
from PIL import Image
from .extended import ExtendedVisionDataset

class us380k(ExtendedVisionDataset):
    """
    Dataset loader for US-VFM classification task.
    Assumes images are organized in subdirectories, with each subdirectory representing a class.
    """
    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        """
        Args:
            root (str): Root directory containing class subdirectories.
            transform (callable, optional): Transform to be applied to images.
            target_transform (callable, optional): Transform to be applied to labels.
        """
        # Pass both transforms and target_transform as a single `transforms` argument
        super().__init__(root=root, transforms=(transform, target_transform))
        self.image_paths, self.labels = self._load_data(self.root)
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _load_data(self, root_dir: str) -> Tuple[List[str], List[str]]:
        """
        Load image paths and labels from the dataset directory.

        Args:
            root_dir (str): Root directory containing class subdirectories.

        Returns:
            Tuple[List[str], List[str]]: Image paths and corresponding class labels.
        """
        image_paths = []
        labels = []
        for label in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".tif", "bmp")):
                        image_paths.append(os.path.join(class_dir, img_file))
                        labels.append(label)
        return image_paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with keys "global_crops", "local_crops", and "label".
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        label_idx = self.class_to_idx[label]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transforms:
            transform, target_transform = self.transforms
            augmented = transform(image) if transform else {"global_crops": image, "local_crops": []}
            global_crops = augmented["global_crops"]
            local_crops = augmented.get("local_crops", [])
            if target_transform:
                label_idx = target_transform(label_idx)
            return {"global_crops": global_crops, "local_crops": local_crops, "label": label_idx}
        else:
            raise ValueError("Transformations are required for DINOv2 training")
