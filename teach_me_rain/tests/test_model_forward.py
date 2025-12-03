import pytest
import torch
from src.models.resnet_model import ResNet18Classifier
from src.config import NUM_CLASSES

def test_model_forward():
    model = ResNet18Classifier(num_classes=NUM_CLASSES, pretrained=False)
    
    # Batch of 2 images, 3 channels, 64x64
    dummy_input = torch.randn(2, 3, 64, 64)
    
    output = model(dummy_input)
    
    assert output.shape == (2, NUM_CLASSES)
    
def test_feature_extraction_freeze():
    model = ResNet18Classifier(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
    
    # Check if backbone params have requires_grad=False
    for param in model.backbone.conv1.parameters():
        assert param.requires_grad == False
        
    # Check if fc layer has requires_grad=True
    for param in model.backbone.fc.parameters():
        assert param.requires_grad == True
