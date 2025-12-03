import pytest
import torch
import numpy as np
from src.training.metrics import compute_accuracy, compute_per_class_accuracy

def test_accuracy():
    outputs = torch.tensor([
        [0.1, 0.9], # Class 1
        [0.8, 0.2], # Class 0
        [0.4, 0.6]  # Class 1
    ])
    targets = torch.tensor([1, 0, 1])
    
    acc = compute_accuracy(outputs, targets)
    assert acc == 1.0
    
    targets_wrong = torch.tensor([0, 1, 0])
    acc_wrong = compute_accuracy(outputs, targets_wrong)
    assert acc_wrong == 0.0

def test_per_class_accuracy():
    # 3 classes
    outputs = torch.tensor([
        [0.9, 0.1, 0.0], # Pred 0, True 0
        [0.1, 0.9, 0.0], # Pred 1, True 1
        [0.2, 0.3, 0.5], # Pred 2, True 2
        [0.8, 0.1, 0.1]  # Pred 0, True 2 (Error)
    ])
    targets = torch.tensor([0, 1, 2, 2])
    
    # Class 0: 1 sample, 1 correct -> 1.0
    # Class 1: 1 sample, 1 correct -> 1.0
    # Class 2: 2 samples, 1 correct -> 0.5
    
    per_class = compute_per_class_accuracy(outputs, targets, num_classes=3)
    
    assert per_class[0] == 1.0
    assert per_class[1] == 1.0
    assert per_class[2] == 0.5
