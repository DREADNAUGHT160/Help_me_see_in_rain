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

def select_uncertain(model_path, data_source_dir, output_dir, topk=20, limit=None, strategy_name="entropy"):
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
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
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

    all_scores = []
    all_paths = []
    all_preds = []
    all_confidences = []

    with torch.no_grad():
        for images, paths in loader:
            images = images.to(DEVICE)
            inputs_size = images.size(0)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            # Calculate scores based on strategy
            if strategy_name == "entropy":
                batch_scores = entropy(probs.cpu().numpy())
            elif strategy_name == "least_confidence":
                # LC = 1 - max_prob
                max_probs, _ = torch.max(probs, dim=1)
                batch_scores = 1.0 - max_probs.cpu().numpy()
            elif strategy_name == "margin":
                # Margin = top1_prob - top2_prob
                # We want smallest margin, so use negative or invert. 
                # Converting to "uncertainty score": 1 - margin (since small margin = uncertain)
                top2, _ = torch.topk(probs, k=2, dim=1)
                margin = top2[:, 0] - top2[:, 1]
                batch_scores = 1.0 - margin.cpu().numpy()
            elif strategy_name == "random":
                # Random score
                batch_scores = np.random.rand(inputs_size)
            else:
                batch_scores = entropy(probs.cpu().numpy())
            
            # Store paths and scores
            all_scores.extend(batch_scores)
            all_paths.extend(paths)
            
            # Store predictions and confidences for "prediction" display
            # Get max prob and class index
            max_probs, preds = torch.max(probs, dim=1)
            
            all_confidences.extend(max_probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Sort by score (descending)
    sorted_indices = np.argsort(all_scores)[::-1]
    
    # Select top k
    selected_indices = sorted_indices[:topk]
    
    # Clear directory if it exists to clean up old run samples
    if os.path.exists(output_dir):
        # We want to keep metrics if we are merging, BUT if we are merging metrics
        # we still probably want to clear the folder of IMAGES to avoid confusion.
        # Actually, if we merge metrics, we might want to keep images?
        # No, "Find Uncertain Samples" implies a new batch. 
        # Best approach: Clear everything.
        shutil.rmtree(output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store metrics for the selected images
    selected_metrics = {}
    
    count = 0
    for idx in selected_indices:
        src_path = all_paths[idx]
        score = all_scores[idx]
        pred_class = int(all_preds[idx])
        confidence = float(all_confidences[idx])
        
        # We rename the file to avoid collisions and keep extension
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, filename)
        
        # Copy file
        shutil.copy2(src_path, dst_path)
        
        # Store metric
        selected_metrics[filename] = {
            "score": float(score),
            "strategy": strategy_name,
            "predicted_class": pred_class,
            "confidence": confidence
        }
        
        count += 1
        
    # Save metrics to JSON
    import json
    # Save metrics to JSON
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(selected_metrics, f, indent=4)
        
    print(f"Selected {count} samples using '{strategy_name}' strategy.")
    print(f"Metrics saved to {metrics_path}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="entropy", 
                        choices=["entropy", "least_confidence", "margin", "random"],
                        help="Active Learning Strategy")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    args = parser.parse_args()

    # Example usage
    MODEL_PATH = "model.pth"
    # Select from Rainy data (unlabeled pool)
    SOURCE_POOL = RAINY_DATA_DIR 
    
    # Run slightly faster for user demo
    select_uncertain(MODEL_PATH, SOURCE_POOL, UNCERTAIN_SAMPLES_DIR, topk=20, limit=args.limit, strategy_name=args.strategy)