import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from src.config import DEVICE, NUM_CLASSES
from src.models.utils import save_checkpoint
from src.training.metrics import compute_accuracy

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Metrics
        batch_size = inputs.size(0)
        acc = compute_accuracy(outputs, targets)
        
        running_loss += loss.item() * batch_size
        running_acc += acc * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({'loss': loss.item(), 'acc': acc})
        
        if logger and i % 10 == 0:
            step = (epoch - 1) * len(dataloader) + i
            logger.log_scalar("Train/BatchLoss", loss.item(), step)
            logger.log_scalar("Train/BatchAcc", acc, step)
            
    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, epoch, logger=None):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            batch_size = inputs.size(0)
            acc = compute_accuracy(outputs, targets)
            
            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            total_samples += batch_size
            
    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, config, logger=None, save_dir="checkpoints"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    num_epochs = config.get("epochs", 10)
    device = config.get("device", DEVICE)
    
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, logger)
        
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        if logger:
            logger.log_scalar("Train/EpochLoss", train_loss, epoch)
            logger.log_scalar("Train/EpochAcc", train_acc, epoch)
            logger.log_scalar("Val/EpochLoss", val_loss, epoch)
            logger.log_scalar("Val/EpochAcc", val_acc, epoch)
            logger.log_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
            
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, {'val_acc': val_acc}, os.path.join(save_dir, "best_model.pth"))
            
        save_checkpoint(model, optimizer, epoch, {'val_acc': val_acc}, os.path.join(save_dir, "last_model.pth"))
        
    return model
