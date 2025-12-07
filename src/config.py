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

# GTSRB Class Names
GTSRB_CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)', 
    2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 
    4: 'Speed limit (70km/h)', 
    5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 
    7: 'Speed limit (100km/h)', 
    8: 'Speed limit (120km/h)', 
    9: 'No passing', 
    10: 'No passing veh over 3.5 tons', 
    11: 'Right-of-way at intersection', 
    12: 'Priority road', 
    13: 'Yield', 
    14: 'Stop', 
    15: 'No vehicles', 
    16: 'Veh > 3.5 tons prohibited', 
    17: 'No entry', 
    18: 'General caution', 
    19: 'Dangerous curve left', 
    20: 'Dangerous curve right', 
    21: 'Double curve', 
    22: 'Bumpy road', 
    23: 'Slippery road', 
    24: 'Road narrows on the right', 
    25: 'Road work', 
    26: 'Traffic signals', 
    27: 'Pedestrians', 
    28: 'Children crossing', 
    29: 'Bicycles crossing', 
    30: 'Beware of ice/snow', 
    31: 'Wild animals crossing', 
    32: 'End speed + passing limits', 
    33: 'Turn right ahead', 
    34: 'Turn left ahead', 
    35: 'Ahead only', 
    36: 'Go straight or right', 
    37: 'Go straight or left', 
    38: 'Keep right', 
    39: 'Keep left', 
    40: 'Roundabout mandatory', 
    41: 'End of no passing', 
    42: 'End no passing veh > 3.5 tons'
}

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
