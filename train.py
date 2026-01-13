import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.config import PROJECT_ROOT, CLEAR_DATA_DIR, NUM_CLASSES, IMG_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS, DEVICE

def get_loader(train_path, batch_size, shuffle=True, num_workers=0):
    """
    Creates a DataLoader for the given path.
    """
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # Normalize is good practice, adding standard ImageNet means
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if directory exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Data directory not found: {train_path}")

    dataset = datasets.ImageFolder(root=train_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, len(dataset.classes)

def build_model(num_classes):
    """
    Builds a ResNet18 model adapted for the specific number of classes.
    """
    # Use weights parameter instead of pretrained=True (deprecated)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_model(model, loader, num_epochs, learning_rate, device):
    """
    Training loop.
    """
    print(f"Training on {device}...")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for e in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {e+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    save_path = "model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    return model, epoch_loss, epoch_acc

def evaluate_model(model, loader, device):
    """Evaluates the model on the validation set."""
    print("Evaluating on validation set...")
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_loss = running_loss / len(loader)
    val_acc = 100 * correct / total
    print(f"Validation - Loss: {val_loss:.4f} - Acc: {val_acc:.2f}%")
    return val_loss, val_acc

def log_training_run(run_id, epochs, train_loss, train_acc, val_loss, val_acc):
    """Logs training metrics to a CSV file."""
    log_file = PROJECT_ROOT / "logs" / "training_history.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize file with header if it doesn't exist
    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write("Timestamp,Run_ID,Epochs,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")
            
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"{timestamp},{run_id},{epochs},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f}\n")

import argparse
from datetime import datetime
import uuid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    args = parser.parse_args()

    # Path to training data
    train_data_path = CLEAR_DATA_DIR / "train"
    val_data_path = CLEAR_DATA_DIR / "val"
    
    if not train_data_path.exists():
        print(f"Training data not found at {train_data_path}. Please run scripts/prepare_gtsrb.py first.")
        sys.exit(1)
        
    print(f"Loading data from {train_data_path}...")
    train_loader, num_classes_found = get_loader(str(train_data_path), BATCH_SIZE)
    
    val_loader = None
    if val_data_path.exists():
        print(f"Loading validation data from {val_data_path}...")
        val_loader, _ = get_loader(str(val_data_path), BATCH_SIZE, shuffle=False)
    else:
        print("Warning: No validation set found. Metrics will be N/A.")
    
    print(f"Detected {num_classes_found} classes. (Config expects {NUM_CLASSES})")
    
    model = build_model(NUM_CLASSES)
    
    # Generate a unique Run ID for this session
    run_id = str(uuid.uuid4())[:8]
    print(f"Starting Run ID: {run_id} for {args.epochs} epochs")
    
    # Train
    model, final_train_loss, final_train_acc = train_model(model, train_loader, args.epochs, LEARNING_RATE, DEVICE)
    
    # Evaluate
    final_val_loss, final_val_acc = 0.0, 0.0
    if val_loader:
        final_val_loss, final_val_acc = evaluate_model(model, val_loader, DEVICE)
        
    # Log to CSV at the END of the run
    log_training_run(run_id, args.epochs, final_train_loss, final_train_acc, final_val_loss, final_val_acc)
