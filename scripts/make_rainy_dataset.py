import os
import shutil
from pathlib import Path
from tqdm import tqdm
from src.config import CLEAR_DATA_DIR, RAINY_DATA_DIR, RAIN_CONFIG
from src.data.rain_augmentation import load_and_add_rain

def make_rainy_dataset():
    """
    Generates rainy version of the clear dataset.
    """
    print("Generating rainy dataset...")
    
    if not CLEAR_DATA_DIR.exists():
        print(f"Error: Clear data directory {CLEAR_DATA_DIR} does not exist. Run prepare_gtsrb.py first.")
        return

    # Clean up if exists
    if RAINY_DATA_DIR.exists():
        shutil.rmtree(RAINY_DATA_DIR)
    
    # Mirror the structure
    # We want to process both train and val folders if they exist
    for split in ["train", "val"]:
        src_split_dir = CLEAR_DATA_DIR / split
        dst_split_dir = RAINY_DATA_DIR / split
        
        if not src_split_dir.exists():
            continue
            
        print(f"Processing {split} set...")
        
        # Iterate over classes
        class_folders = [d for d in src_split_dir.iterdir() if d.is_dir()]
        
        for class_folder in tqdm(class_folders, desc=f"Classes in {split}"):
            class_name = class_folder.name
            dst_class_dir = dst_split_dir / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            
            images = list(class_folder.glob("*")) # match all files
            
            for img_path in images:
                if img_path.suffix.lower() not in ['.ppm', '.png', '.jpg', '.jpeg']:
                    continue
                    
                dst_path = dst_class_dir / img_path.name
                
                try:
                    load_and_add_rain(img_path, dst_path, config=RAIN_CONFIG)
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")
                    
    print(f"Done! Rainy data generated in {RAINY_DATA_DIR}")

if __name__ == "__main__":
    make_rainy_dataset()
