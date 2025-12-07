import torch, torch.nn.functional as F, numpy as np, os, shutil
from torchvision import datasets, transforms
from utils import entropy


def select_uncertain(model_path, data_path, topk=20):
    dev = 'cuda' if torch.cuda.is_available() else "cpu"
    tf = transfor