import os
import cv2
import albumentations as A
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import shutil

# Add project root to path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.config import CLEAR_DATA_DIR, RAINY_DATA_DIR, RAIN_CONFIG

def generate_rain():
    """
    Generates a rainy version of the dataset using Albumentations.
    """
    print(f"Generating rainy dataset from {CLEAR_DATA_DIR} to {RAINY_DATA_DIR}...")
    
    if RAINY_DATA_DIR.exists():
        print(f"Removing existing {RAINY_DATA_DIR}...")
        try:
            shutil.rmtree(RAINY_DATA_DIR)
        except Exception as e:
            print(f"Error removing directory: {e}")
            return

    # Define the rain transformation
    # We use p=1.0 to ensure rain is applied to every image in this parallel dataset
    transform = A.Compose([
        A.RandomRain(
            brightness_coefficient=RAIN_CONFIG["brightness_coefficient"],
            drop_length=RAIN_CONFIG["drop_length"],
            drop_width=RAIN_CONFIG["drop_width"],
            blur_value=RAIN_CONFIG["blur_value"],
            p=1.0
        ),
    ])

    # Walk through the clear data directory
    # Structure: data/clear/[train|val]/[class_id]/[image_name]
    
    # Get all image files
    image_files = list(CLEAR_DATA_DIR.rglob("*.ppm"))
    
    if not image_files:
        print(f"No images found in {CLEAR_DATA_DIR}. Please run prepare_gtsrb.py first.")
        return

    for img_path in tqdm(image_files, desc="Adding rain"):
        # Calculate relative path to mirror structure
        rel_path = img_path.relative_to(CLEAR_DATA_DIR)
        target_path = RAINY_DATA_DIR / rel_path
        
        # Create parent directory
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        # Apply rain
        # Albumentations expects RGB but cv2 reads BGR. 
        # However, for just rain noise it might not matter much, 
        # but technically we should convert to RGB, apply, then back to BGR for saving.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image)["image"]
        transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
        
        # Save
        cv2.imwrite(str(target_path), transformed)
        
    print(f"Rain generation complete! Saved to {RAINY_DATA_DIR}")

if __name__ == "__main__":
    generate_rain()
