import torch
import os

def save_checkpoint(model, optimizer, epoch, metrics, path):
    """
    Saves a model checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, path, optimizer=None, device="cpu"):
    """
    Loads a checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})

def count_parameters(model):
    """
    Returns the number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
