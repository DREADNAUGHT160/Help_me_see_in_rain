import torch
import os
import shutil
import numpy as np
from pathlib import Path
from src.config import (
    CLEAR_DATA_DIR, RAINY_DATA_DIR, HITS_DIR, UNCERTAIN_SAMPLES_DIR, 
    ANNOTATIONS_FILE, DEVICE, NUM_CLASSES, PROJECT_ROOT, AL_BATCH_SIZE
)
from src.data.gtsrb_dataset import get_dataloader
from src.models.resnet_model import get_model, ResNet18Classifier
from src.models.utils import load_checkpoint
from src.training.train import train_model
from src.training.active_learning import compute_probabilities, select_topk_uncertain
from src.hitl.annotation_store import AnnotationStore
from src.training.tensorboard_utils import TensorBoardLogger

def prepare_pool():
    """
    Prepares the unlabeled pool from rainy data.
    For simulation, we use the rainy train set.
    """
    pool_dir = RAINY_DATA_DIR / "train"
    if not pool_dir.exists():
        raise FileNotFoundError(f"Rainy train data not found at {pool_dir}")
    return pool_dir

def run_active_learning_loop():
    print("Starting Active Learning Loop...")
    
    # 1. Load Baseline Model
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "baseline", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print("Baseline checkpoint not found. Run run_baseline.py first.")
        return

    model = get_model(device=DEVICE, num_classes=NUM_CLASSES)
    load_checkpoint(model, checkpoint_path, device=DEVICE)
    print("Loaded baseline model.")
    
    # 2. Prepare Unlabeled Pool
    pool_dir = prepare_pool()
    pool_loader = get_dataloader(str(pool_dir), batch_size=64, split="train", shuffle=False)
    
    # 3. Compute Uncertainties
    print("Computing uncertainties on pool...")
    probs, paths = compute_probabilities(model, pool_loader, device=DEVICE)
    
    # 4. Select Top-K
    print(f"Selecting top {AL_BATCH_SIZE} uncertain samples...")
    indices, selected_paths, entropies = select_topk_uncertain(probs, k=AL_BATCH_SIZE, paths=paths)
    
    # 5. Export for Labeling
    UNCERTAIN_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear previous samples
    for f in UNCERTAIN_SAMPLES_DIR.glob("*"):
        f.unlink()
        
    print(f"Copying {len(selected_paths)} samples to {UNCERTAIN_SAMPLES_DIR}...")
    for p in selected_paths:
        shutil.copy(p, UNCERTAIN_SAMPLES_DIR / Path(p).name)
        
    print("Ready for Human Labeling! Run `streamlit run src/hitl/streamlit_app.py`")
    
    # 6. (Optional) Check if annotations exist and retrain
    # This part would be a separate step or loop in a real system
    store = AnnotationStore()
    if store.get_count() > 0:
        print(f"Found {store.get_count()} annotations. Proceeding to fine-tuning...")
        # Implement fine-tuning logic here:
        # 1. Create a new dataset merging Clear + Annotated Rainy
        # 2. Fine-tune model
        # 3. Evaluate
    else:
        print("No annotations found yet. Please label the data first.")

if __name__ == "__main__":
    run_active_learning_loop()
