import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAR_DATA_DIR = DATA_DIR / "clear"
RAINY_DATA_DIR = DATA_DIR / "rainy"
HITS_DIR = DATA_DIR / "hits"
UNCERTAIN_SAMPLES_DIR = HITS_DIR / "uncertain_samples"
ANNOTATIONS_FILE = HITS_DIR / "annotations.json"

# Model Config
NUM_CLASSES = 43  # GTSRB has 43 classes
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Rain Generation Config
RAIN_CONFIG = {
    "brightness_coefficient": 0.9,
    "drop_length": 20,
    "drop_width": 1,
    "blur_value": 5,
    "rain_type": "default"  # 'drizzle', 'heavy', 'torrential', or 'default'
}

# Active Learning Config
AL_BATCH_SIZE = 50  # Number of samples to query per round
