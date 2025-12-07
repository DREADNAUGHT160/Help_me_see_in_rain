import torch
import torch.nn.functional as F
import numpy as np
import os
import shutil
from pathlib import Path
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.config import CLEAR_DATA_DIR, RAINY_DATA_DIR, UNCERTAIN_SAMPLES_DIR, NUM_CLASSES, IMG_SIZE, DEVICE
from train import build_model
from utils import entropy

class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(img_path)

def select_uncertain(model_path, data_source_dir, output_dir, topk=20, limit=None):
    """
    Selects the top-k most uncertain samples from data_source_dir 
    and copies them to output_dir for labeling.
    """
    print(f"Running Active Learning Selection...")
    print(f"Source: {data_source_dir}")
    print(f"Output: {output_dir}")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
        return

    # Load Model
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Prepare Data
    # Recursively find all images
    image_paths = list(Path(data_source_dir).rglob("*.ppm"))
    if not image_paths:
        print("No images found to process.")
        return

    if limit:
        print(f"Limiting to {limit} samples for speed...")
        image_paths = image_paths[:limit]

    print(f"Found {len(image_paths)} candidates.")

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SimpleImageDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_entropies = []
    all_paths = []

    with torch.no_grad():
        for images, paths in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            # Calculate entropy
            batch_entropies = entropy(probs.cpu().numpy())
            
            all_entropies.extend(batch_entropies)
            all_paths.extend(paths)

    # Sort by entropy (descending)
    sorted_indices = np.argsort(all_entropies)[::-1]
    
    # Select top k
    selected_indices = sorted_indices[:topk]
    
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for idx in selected_indices:
        src_path = all_paths[idx]
        
        # We rename the file to avoid collisions and keep extension
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, filename)
        
        # If file already exists in uncertain (e.g. from previous run), skip or overwrite
        shutil.copy2(src_path, dst_path)
        count += 1

    print(f"Selected {count} uncertain samples and moved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "model.pth"
    # Select from Rainy data (unlabeled pool)
    SOURCE_POOL = RAINY_DATA_DIR 
    
    # Run slightly faster for user demo
    select_uncertain(MODEL_PATH, SOURCE_POOL, UNCERTAIN_SAMPLES_DIR, topk=20, limit=500)