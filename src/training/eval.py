import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from src.config import DEVICE, NUM_CLASSES
from src.training.metrics import compute_accuracy, compute_per_class_accuracy

def evaluate_model(model, dataloader, device=DEVICE):
    """
    Evaluates the model on a dataset and returns metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.append(preds.cpu())
            all_targets.append(targets)
            
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    acc = (all_preds == all_targets).float().mean().item()
    
    # Per class accuracy
    # We can use sklearn or manual
    from sklearn.metrics import classification_report
    report = classification_report(all_targets.numpy(), all_preds.numpy(), output_dict=True, zero_division=0)
    
    return {
        "accuracy": acc,
        "report": report
    }
