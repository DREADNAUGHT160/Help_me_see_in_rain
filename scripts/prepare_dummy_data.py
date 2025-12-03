import os
import numpy as np
import cv2
from pathlib import Path
from src.config import CLEAR_DATA_DIR, NUM_CLASSES

def create_dummy_data():
    print("Creating dummy dataset for testing...")
    
    # Create train and val directories
    for split in ["train", "val"]:
        split_dir = CLEAR_DATA_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a few classes
        for class_id in range(5): # Just 5 classes for testing
            class_dir = split_dir / str(class_id)
            class_dir.mkdir(exist_ok=True)
            
            # Create 10 images per class
            for i in range(10):
                # Random image
                img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                
                # Draw class ID on it so it's somewhat learnable (maybe)
                cv2.putText(img, str(class_id), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imwrite(str(class_dir / f"img_{i}.png"), img)
                
    print(f"Dummy data created in {CLEAR_DATA_DIR}")

if __name__ == "__main__":
    create_dummy_data()
