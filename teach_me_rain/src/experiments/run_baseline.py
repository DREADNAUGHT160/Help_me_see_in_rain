import torch
import os
from src.config import CLEAR_DATA_DIR, RAINY_DATA_DIR, DEVICE, NUM_CLASSES, PROJECT_ROOT
from src.data.gtsrb_dataset import get_dataloader
from src.models.resnet_model import get_model
from src.training.train import train_model
from src.training.eval import evaluate_model
from src.training.tensorboard_utils import TensorBoardLogger

def run_baseline():
    print("Running Baseline Experiment...")
    
    # Config
    config = {
        "lr": 0.001,
        "epochs": 5,
        "device": DEVICE
    }
    
    # Data
    train_loader = get_dataloader(str(CLEAR_DATA_DIR / "train"), batch_size=64, split="train")
    val_loader = get_dataloader(str(CLEAR_DATA_DIR / "val"), batch_size=64, split="val")
    
    # Model
    model = get_model(device=DEVICE, num_classes=NUM_CLASSES)
    
    # Logger
    log_dir = os.path.join(PROJECT_ROOT, "logs", "baseline")
    logger = TensorBoardLogger(log_dir)
    
    # Train
    print("Training on Clear Data...")
    model = train_model(model, train_loader, val_loader, config, logger, save_dir=os.path.join(PROJECT_ROOT, "checkpoints", "baseline"))
    
    # Evaluate on Clear Test (Val)
    print("Evaluating on Clear Data...")
    clear_metrics = evaluate_model(model, val_loader, device=DEVICE)
    print(f"Clear Accuracy: {clear_metrics['accuracy']:.4f}")
    logger.log_scalar("Eval/ClearAcc", clear_metrics['accuracy'], 0)
    
    # Evaluate on Rainy Data (if exists)
    rainy_val_dir = RAINY_DATA_DIR / "val"
    if rainy_val_dir.exists():
        print("Evaluating on Rainy Data...")
        rainy_loader = get_dataloader(str(rainy_val_dir), batch_size=64, split="val")
        rainy_metrics = evaluate_model(model, rainy_loader, device=DEVICE)
        print(f"Rainy Accuracy: {rainy_metrics['accuracy']:.4f}")
        logger.log_scalar("Eval/RainyAcc", rainy_metrics['accuracy'], 0)
    else:
        print("Rainy data not found, skipping evaluation.")
        
    logger.close()

if __name__ == "__main__":
    run_baseline()
