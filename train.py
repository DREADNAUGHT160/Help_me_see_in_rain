import torch 
import torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader
import os

# make the datolader.
def get_loader(train_path, batch_size, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=train_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, len(dataset.classes)




def build_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(model,loader, num_epochs, learning_rate, device):
    model = model.to(device)
    opt = optim.Adam(model.parameter(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    for e in range(num_epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_function(model(x), y)
            loss.backward(); opt.step()
        print(f"Epochs {e+1}/{num_epochs}done.")
    torch.save(model.state_dict(), "model.pth")
    print("Saved model.pth")
    return model
