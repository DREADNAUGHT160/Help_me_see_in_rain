import albumentations as A
from PIL import Image
import numpy as np
import os



rain_aug = A.RandomRain(
    brightness_coefficient=0.9,
    drop_length=20,
    drop_width=1,
    blur_value=3,
    rain_type="heavy",
    # always_apply=True, # Deprecated/Invalid
    p=1.0, # Use p=1.0 instead for always apply
)


def add_rain_to_image(image_path, output_path):
    """Adds rain effect to an image and saves the result.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image with rain effect.
    """
    os.makedirs(output_path, exist_ok=True) # creating a directory to save rainy images and also checking is the directory is already present.
    for f in os.listdir(image_path):# going through all the files in the input directory.
        # Load image
        image = np.array(Image.open(os.path.join(image_path, f)).convert("RGB"))

        # Apply rain augmentation
        augmented = rain_aug(image=image)
        rainy_image = augmented["image"]
        Image.fromarray(rainy_image).save(os.path.join(output_path,f))

def entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10), axis=-1)