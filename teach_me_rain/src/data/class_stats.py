import os
import numpy as np
from collections import Counter
from pathlib import Path

def get_class_counts(root_dir: str) -> dict[str, int]:
    """
    Computes the number of samples per class in a directory structure
    where each subdirectory represents a class.
    
    Args:
        root_dir: Path to the dataset root (e.g., data/clear/train)
        
    Returns:
        Dictionary mapping class names (folder names) to counts.
    """
    root = Path(root_dir)
    if not root.exists():
        return {}
        
    counts = {}
    for class_dir in root.iterdir():
        if class_dir.is_dir():
            # Count files in the directory
            num_files = len([f for f in class_dir.iterdir() if f.is_file()])
            counts[class_dir.name] = num_files
    return counts

def compute_imbalance_stats(counts: dict[str, int]) -> dict:
    """
    Computes mean, std, and imbalance ratios from class counts.
    
    Args:
        counts: Dictionary of class counts.
        
    Returns:
        Dictionary with 'mean', 'std', 'min', 'max', and 'imbalance_ratios'.
    """
    if not counts:
        return {}
        
    values = list(counts.values())
    mean_count = np.mean(values)
    std_count = np.std(values)
    
    imbalance_ratios = {k: v / mean_count for k, v in counts.items()}
    
    return {
        "mean": mean_count,
        "std": std_count,
        "min": np.min(values),
        "max": np.max(values),
        "imbalance_ratios": imbalance_ratios
    }

def print_stats(stats: dict):
    """Pretty prints the statistics."""
    print(f"Class Count Stats:")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Std : {stats['std']:.2f}")
    print(f"  Min : {stats['min']}")
    print(f"  Max : {stats['max']}")
    print("  Imbalance Ratios (Top 5 lowest):")
    sorted_ratios = sorted(stats['imbalance_ratios'].items(), key=lambda x: x[1])
    for cls, ratio in sorted_ratios[:5]:
        print(f"    {cls}: {ratio:.2f}")
    print("  Imbalance Ratios (Top 5 highest):")
    for cls, ratio in sorted_ratios[-5:]:
        print(f"    {cls}: {ratio:.2f}")
