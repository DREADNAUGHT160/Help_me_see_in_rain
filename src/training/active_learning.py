import torch
import numpy as np
from tqdm import tqdm
from src.training.metrics import compute_entropy

def compute_probabilities(model, dataloader, device):
    """
    Computes probabilities for all samples in the dataloader.
    Returns:
        probs: (N, C) numpy array
        paths: list of file paths (if available in dataset)
    """
    model.eval()
    all_probs = []
    all_paths = []
    
    # Check if dataset has samples attribute (ImageFolder)
    has_paths = hasattr(dataloader.dataset, 'samples')
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(dataloader, desc="Computing probabilities")):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            
            if has_paths:
                # Calculate indices for this batch
                start_idx = i * dataloader.batch_size
                end_idx = start_idx + inputs.size(0)
                # This assumes dataloader is not shuffled or we can access samples by index
                # If shuffled, we can't easily map back unless we return indices from dataset
                # For AL on pool, usually we don't shuffle
                batch_paths = [s[0] for s in dataloader.dataset.samples[start_idx:end_idx]]
                all_paths.extend(batch_paths)
                
    return np.concatenate(all_probs), all_paths

def select_topk_uncertain(probs, k, paths=None):
    """
    Selects top-k samples with highest entropy.
    """
    entropy = compute_entropy(probs)
    # argsort returns indices that sort the array
    # we want descending order (highest entropy)
    sorted_indices = np.argsort(entropy)[::-1]
    
    top_k_indices = sorted_indices[:k]
    
    if paths:
        selected_paths = [paths[i] for i in top_k_indices]
        return top_k_indices, selected_paths, entropy[top_k_indices]
        
    return top_k_indices, None, entropy[top_k_indices]
