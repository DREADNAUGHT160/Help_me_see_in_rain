import streamlit as st
import os
import shutil
from pathlib import Path
from PIL import Image
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import UNCERTAIN_SAMPLES_DIR, CLEAR_DATA_DIR, NUM_CLASSES

def main():
    st.title("Human-in-the-Loop Labeling")
    st.markdown("Label the uncertain samples to improve the model.")

    if not UNCERTAIN_SAMPLES_DIR.exists():
        st.warning(f"No uncertain samples directory found at {UNCERTAIN_SAMPLES_DIR}")
        return

    # Get list of images
    images = list(UNCERTAIN_SAMPLES_DIR.glob("*.ppm"))
    
    if not images:
        st.success("No images left to label! Great job.")
        if st.button("Refresh"):
            st.rerun()
        return

    # Show progress
    st.sidebar.write(f"Remaining images: {len(images)}")

    # Pick the first image
    img_path = images[0]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(str(img_path), caption=img_path.name, width=300)

    with col2:
        st.write("### Select Class")
        
        # Grid layout for buttons
        cols = st.columns(4)
        for class_id in range(NUM_CLASSES):
            with cols[class_id % 4]:
                if st.button(f"{class_id}", key=f"btn_{class_id}"):
                    move_image(img_path, class_id)
                    st.rerun()

    if st.button("Skip"):
        # Move to end of list or just ignore for now (implementation detail: shuffling or similar)
        # For simple file list, maybe just rename it to .skipped or move to another folder
        # Here we just pass, it will show up again next refresh unless we act.
        # Let's move it to a 'skipped' folder
        skipped_dir = UNCERTAIN_SAMPLES_DIR.parent / "skipped"
        skipped_dir.mkdir(exist_ok=True)
        shutil.move(str(img_path), str(skipped_dir / img_path.name))
        st.rerun()

def move_image(img_path, class_id):
    """Moves image to the training directory of the selected class."""
    target_dir = CLEAR_DATA_DIR / "train" / f"{class_id:05d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = target_dir / img_path.name
    
    # Move file
    shutil.move(str(img_path), str(target_path))
    st.toast(f"Labeled {img_path.name} as Class {class_id}")

if __name__ == "__main__":
    main()
