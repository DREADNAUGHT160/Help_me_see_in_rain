import matplotlib.pyplot as plt
import seaborn as sns
from src.config import CLEAR_DATA_DIR
from src.data.class_stats import get_class_counts, compute_imbalance_stats

def plot_distribution():
    train_dir = CLEAR_DATA_DIR / "train"
    if not train_dir.exists():
        print("Train directory not found.")
        return

    counts = get_class_counts(str(train_dir))
    stats = compute_imbalance_stats(counts)
    
    print("Statistics:")
    print(f"Total images: {sum(counts.values())}")
    print(f"Mean per class: {stats['mean']:.2f}")
    
    # Sort by class ID (assuming numeric names)
    try:
        sorted_keys = sorted(counts.keys(), key=lambda x: int(x))
    except:
        sorted_keys = sorted(counts.keys())
        
    sorted_counts = [counts[k] for k in sorted_keys]
    
    plt.figure(figsize=(15, 6))
    sns.barplot(x=sorted_keys, y=sorted_counts)
    plt.title("GTSRB Class Distribution (Train)")
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_distribution()
