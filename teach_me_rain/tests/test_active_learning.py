import pytest
import numpy as np
import torch
from src.training.active_learning import select_topk_uncertain
from src.training.metrics import compute_entropy

def test_entropy_computation():
    # 2 samples, 3 classes
    # Sample 1: High entropy (uniform)
    # Sample 2: Low entropy (confident)
    probs = np.array([
        [0.33, 0.33, 0.34],
        [0.90, 0.05, 0.05]
    ])
    
    entropy = compute_entropy(probs)
    
    assert entropy[0] > entropy[1]

def test_selection_logic():
    # 3 samples
    # 0: High entropy
    # 1: Medium
    # 2: Low
    probs = np.array([
        [0.33, 0.33, 0.34], # High
        [0.50, 0.40, 0.10], # Medium
        [0.99, 0.01, 0.00]  # Low
    ])
    
    indices, _, _ = select_topk_uncertain(probs, k=2)
    
    # Should select 0 and 1
    assert 0 in indices
    assert 1 in indices
    assert 2 not in indices
    assert len(indices) == 2
