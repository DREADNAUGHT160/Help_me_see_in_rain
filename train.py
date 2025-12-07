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
from src.config import CLEAR_DATA_DIR, NUM_CLASSES, IMG_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS, DEVICE

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
    return model

if __name__ == "__main__":
    # Path to training data
    train_data_path = CLEAR_DATA_DIR / "train"
    
    if not train_data_path.exists():
        print(f"Training data not found at {train_data_path}. Please run scripts/prepare_gtsrb.py first.")
        sys.exit(1)
        
    print(f"Loading data from {train_data_path}...")
    train_loader, num_classes_found = get_loader(str(train_data_path), BATCH_SIZE)
    
    print(f"Detected {num_classes_found} classes. (Config expects {NUM_CLASSES})")
    
    model = build_model(NUM_CLASSES)
    
    train_model(model, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
