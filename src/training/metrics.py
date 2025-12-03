import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

def compute_accuracy(outputs, targets):
    """
    Computes top-1 accuracy.
    """
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == targets).item() / len(targets)

def compute_per_class_accuracy(outputs, targets, num_classes):
    """
    Computes accuracy per class.
    """
    _, preds = torch.max(outputs, 1)
    cm = confusion_matrix(targets.cpu().numpy(), preds.cpu().numpy(), labels=range(num_classes))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    # Replace NaNs (for classes with no samples) with 0
    per_class_acc = np.nan_to_num(per_class_acc)
    return per_class_acc

def compute_entropy(probs):
    """
    Computes entropy of probabilities.
    probs: (N, C) tensor or numpy array
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
        
    # Add epsilon to avoid log(0)
    epsilon = 1e-10
    entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
    return entropy
