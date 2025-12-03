import pytest
import torch
import os
from src.data.gtsrb_dataset import GTSRBDataset, get_transforms
from src.data.rain_augmentation import add_rain_to_image
import numpy as np

def test_transforms_output_shape():
    transform = get_transforms(split="train")
    # Create dummy image (H, W, C) -> PIL or Tensor?
    # transforms.Resize expects PIL or Tensor. 
    # But ToTensor expects PIL or numpy.ndarray (H, W, C) in [0, 255]
    
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # ToTensor converts to (C, H, W) and scales to [0, 1]
    # Resize resizes to (64, 64)
    # Normalize subtracts mean/std
    
    # Note: Compose applies in order. 
    # If Resize is first, it expects PIL or Tensor. 
    # If we pass numpy, we might need ToPILImage first if Resize doesn't support numpy.
    # torchvision.transforms.Resize supports Tensor.
    
    # Let's convert to Tensor first manually to check pipeline or just use PIL
    from PIL import Image
    pil_img = Image.fromarray(dummy_img)
    
    out = transform(pil_img)
    
    assert out.shape == (3, 64, 64)
    assert isinstance(out, torch.Tensor)

def test_rain_augmentation():
    # (H, W, C)
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    rainy = add_rain_to_image(dummy_img)
    
    assert rainy.shape == dummy_img.shape
    assert rainy.dtype == dummy_img.dtype
    # Check that image is modified (probabilistic, but p=1.0 in our config)
    assert not np.array_equal(dummy_img, rainy)
