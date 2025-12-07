import os
import shutil
import random
from pathlib import Path
import torchvision
from tqdm import tqdm
import sys
# Add project root to path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR, CLEAR_DATA_DIR

def prepare_gtsrb():
    """
    Downloads GTSRB and prepares train/val split in ImageFolder format.
    """
    print("Downloading GTSRB dataset...")
    # We use torchvision to download into raw folder
    # This will create RAW_DATA_DIR/gtsrb/GTSRB/Final_Training/Images
    dataset = torchvision.datasets.GTSRB(
        root=str(RAW_DATA_DIR),
        split="train",
        download=True
    )
    
    # Path to the downloaded images
    # torchvision downloads to <root>/gtsrb/GTSRB/Final_Training/Images
    source_dir = RAW_DATA_DIR / "gtsrb" / "GTSRB" / "Final_Training" / "Images"
    
    if not source_dir.exists():
        print(f"Error: Expected source directory {source_dir} does not exist.")
        return

    print("Creating train/val split...")
    train_dir = CLEAR_DATA_DIR / "train"
    val_dir = CLEAR_DATA_DIR / "val"
    
    # Clean up if exists
    if train_dir.exists(): shutil.rmtree(train_dir)
    if val_dir.exists(): shutil.rmtree(val_dir)
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Split ratio
    val_ratio = 0.2
    
    # Iterate over classes
    class_folders = [d for d in source_dir.iterdir() if d.is_dir()]
    
    for class_folder in tqdm(class_folders, desc="Processing classes"):
        class_name = class_folder.name
        images = list(class_folder.glob("*.ppm"))
        
        # Shuffle
        random.shuffle(images)
        
        # Split
        num_val = int(len(images) * val_ratio)
        val_images = images[:num_val]
        train_images = images[num_val:]
        
        # Create class subdirs
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        
        # Copy files
        for img_path in train_images:
            shutil.copy(img_path, train_dir / class_name / img_path.name)
            
        for img_path in val_images:
            shutil.copy(img_path, val_dir / class_name / img_path.name)
            
    print(f"Done! Data prepared in {CLEAR_DATA_DIR}")

if __name__ == "__main__":
    prepare_gtsrb()
