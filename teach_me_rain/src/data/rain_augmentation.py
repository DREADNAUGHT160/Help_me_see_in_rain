import albumentations as A
import cv2
import numpy as np
from src.config import RAIN_CONFIG

def get_rain_transform(config=None):
    """
    Returns an Albumentations composition for rain generation.
    """
    if config is None:
        config = RAIN_CONFIG
        
    # Map rain_type string to Albumentations parameters if needed, 
    # but RandomRain handles most logic via brightness/blur/etc.
    # We use the config directly.
    
    return A.Compose([
        A.RandomRain(
            brightness_coefficient=config.get("brightness_coefficient", 0.9),
            drop_length=config.get("drop_length", 20),
            drop_width=config.get("drop_width", 1),
            blur_value=config.get("blur_value", 5),
            rain_type=config.get("rain_type", None),
            p=1.0 # Always apply rain when this transform is called
        )
    ])

def add_rain_to_image(image: np.ndarray, config=None) -> np.ndarray:
    """
    Applies rain to a single image (H, W, C) numpy array.
    """
    transform = get_rain_transform(config)
    augmented = transform(image=image)
    return augmented["image"]

def load_and_add_rain(image_path: str, save_path: str, config=None):
    """
    Loads an image, adds rain, and saves it to the destination.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Albumentations expects RGB, cv2 loads BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    rainy_image = add_rain_to_image(image, config)
    
    # Save as BGR
    rainy_image_bgr = cv2.cvtColor(rainy_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), rainy_image_bgr)
