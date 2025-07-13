# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Callable, Optional
import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")
_Target = int

class us120k(ExtendedVisionDataset):
    """
    Custom dataset for US-VFM classification task.
    Each subdirectory in the dataset root represents a class with labeled images inside.
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            root_dir (str): Root directory containing class subdirectories.
            transform (callable, optional): Transform to be applied to images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._load_data(root_dir)
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _load_data(self, root_dir: str):
        """
        Load image paths and labels from the dataset directory.

        Args:
            root_dir (str): Root directory containing class subdirectories.

        Returns:
            tuple: A list of image paths and corresponding class labels.
        """
        image_paths = []
        labels = []
        for label in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                        image_paths.append(os.path.join(class_dir, img_file))
                        labels.append(label)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with keys "image" and "label".
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        label_idx = self.class_to_idx[label]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label_idx}
