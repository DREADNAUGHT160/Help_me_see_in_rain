import os
from pathlib import Path
from typing import Optional, Callable, Tuple, Any

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from src.config import IMG_SIZE

# Computed from GTSRB training set (approximate values)
GTSRB_MEAN = [0.3337, 0.3064, 0.3171]
GTSRB_STD = [0.2672, 0.2564, 0.2629]

class GTSRBDataset(ImageFolder):
    """
    Wrapper around ImageFolder for GTSRB dataset.
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )

def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Returns standard transforms for GTSRB.
    
    Args:
        split: 'train' or 'val'/'test'. 
               Currently we use same resize/norm for all, 
               but augmentation could be added to train.
    """
    transform_list = [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD)
    ]
    
    return transforms.Compose(transform_list)

def get_dataloader(
    root_dir: str, 
    batch_size: int = 32, 
    shuffle: bool = True, 
    num_workers: int = 4,
    split: str = "train"
):
    """
    Helper to create a DataLoader for GTSRB.
    """
    dataset = GTSRBDataset(
        root=root_dir,
        transform=get_transforms(split=split)
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
