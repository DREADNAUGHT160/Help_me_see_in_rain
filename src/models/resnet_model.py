import torch
import torch.nn as nn
import torchvision.models as models
from src.config import NUM_CLASSES

class ResNet18Classifier(nn.Module):
    """
    ResNet-18 based classifier for GTSRB.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = False):
        super(ResNet18Classifier, self).__init__()
        
        # Load ResNet-18
        # weights='DEFAULT' is equivalent to pretrained=True in newer torchvision
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        # ResNet18's fc input features is 512
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def get_model(device: str = "cpu", **kwargs):
    """
    Factory function to get the model and move it to device.
    """
    model = ResNet18Classifier(**kwargs)
    model = model.to(device)
    return model
