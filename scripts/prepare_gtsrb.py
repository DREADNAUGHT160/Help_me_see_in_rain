import os
import shutil
import random
import time
from pathlib import Path
import torchvision
from tqdm import tqdm
import sys

# Add project root to path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR, CLEAR_DATA_DIR

def fast_scandir(dirname):
    """
    Scanning directory significantly faster than os.listdir
    """
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.
    If the error is because the file is being used by another process, it waits and retries.
    """
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, 0o777)
        try:
            func(path)
            return
        except Exception:
            pass
    
    # Check if it is a "file in use" error (WinError 145 or 32)
    # WinError 145: The directory is not empty (often happens if file inside is locked)
    # WinError 32: The process cannot access the file because it is being used by another process
    print(f"  Warning: Could not remove {path}. Retrying in 1 second...")
    time.sleep(1)
    try:
        func(path)
    except Exception as e:
        print(f"  Failed to remove {path}: {e}")
        print("  Please close any programs (like VS Code or Explorer) that might use this specific file/folder.")

def prepare_gtsrb():
    """
    Downloads GTSRB and prepares train/val split in ImageFolder format.
    """
    print("Downloading GTSRB dataset...")
    # We use torchvision to download into raw folder
    dataset = torchvision.datasets.GTSRB(
        root=str(RAW_DATA_DIR),
        split="train",
        download=True
    )
    
    # Path to the downloaded images
    # Actual structure is <root>/gtsrb/GTSRB/Training/<class_id>
    # We'll check both expected and actual locations to be safe
    base_gtsrb = RAW_DATA_DIR / "gtsrb" / "GTSRB"
    possible_sources = [
        base_gtsrb / "Training",                   # Common structure
        base_gtsrb / "Final_Training" / "Images"   # Old structure
    ]
    
    source_dir = None
    for p in possible_sources:
        if p.exists():
            source_dir = p
            break
            
    if not source_dir:
        print(f"Error: Could not find GTSRB data in expected locations: {[str(p) for p in possible_sources]}")
        print(f"Contents of {base_gtsrb}: {[p.name for p in base_gtsrb.iterdir()] if base_gtsrb.exists() else 'Directory not found'}")
        return

    print(f"Found source data at: {source_dir}")

    print("Creating train/val split...")
    train_dir = CLEAR_DATA_DIR / "train"
    val_dir = CLEAR_DATA_DIR / "val"
    
    # Clean up if exists with robust error handling
    if train_dir.exists():
        print(f"Removing existing {train_dir}...")
        shutil.rmtree(train_dir, onerror=on_rm_error)
            
    if val_dir.exists(): 
        print(f"Removing existing {val_dir}...")
        shutil.rmtree(val_dir, onerror=on_rm_error)
    
    # Double check if removal worked, if not we might be merging (which is okay-ish but warn)
    if train_dir.exists() or val_dir.exists():
        print("Warning: Could not strictly clean output directories. New data will be merged/overwritten.")

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Split ratio
    val_ratio = 0.2
    
    # Iterate over classes
    try:
        class_folders = [d for d in source_dir.iterdir() if d.is_dir()]
    except Exception as e:
        print(f"Error reading source directory: {e}")
        return
    
    if not class_folders:
        print("Error: No class folders found in source directory.")
        return

    for class_folder in tqdm(class_folders, desc="Processing classes"):
        class_name = class_folder.name
        images = list(class_folder.glob("*.ppm"))
        
        if not images:
            continue

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
