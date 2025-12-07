import streamlit as st
import os
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    PROJECT_ROOT, 
    CLEAR_DATA_DIR, 
    RAINY_DATA_DIR, 
    UNCERTAIN_SAMPLES_DIR, 
    NUM_CLASSES,
    EPOCHS,
    GTSRB_CLASSES
)

st.set_page_config(layout="wide", page_title="Teach Me to See in Rain")

def run_command(command, desc):
    """Runs a shell command and streams output to streamlit."""
    st.info(f"Running: {desc}...")
    
    # We use a placeholder to update output
    out_placeholder = st.empty()
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=PROJECT_ROOT,
        shell=True
    )
    
    output_log = []
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            output_log.append(line.strip())
            # Keep only last 10 lines for cleaner UI
            out_placeholder.code("\n".join(output_log[-10:]))
            
    if process.returncode == 0:
        st.success(f"✅ {desc} Completed Successfully!")
    else:
        stderr = process.stderr.read()
        st.error(f"❌ {desc} Failed!\nError: {stderr}")

def get_stats():
    """Calculates dataset statistics."""
    stats = {}
    
    # Clear Data Stats
    if CLEAR_DATA_DIR.exists():
        clear_count = len(list(CLEAR_DATA_DIR.rglob("*.ppm")))
        stats['Clear Images'] = clear_count
    else:
        stats['Clear Images'] = 0
        
    # Rainy Data Stats
    if RAINY_DATA_DIR.exists():
        rainy_count = len(list(RAINY_DATA_DIR.rglob("*.ppm")))
        stats['Rainy Pool'] = rainy_count
    else:
        stats['Rainy Pool'] = 0
        
    # Uncertain Samples
    if UNCERTAIN_SAMPLES_DIR.exists():
        uncertain_count = len(list(UNCERTAIN_SAMPLES_DIR.glob("*.ppm")))
        stats['To Label'] = uncertain_count
    else:
        stats['To Label'] = 0
        
    return stats

def get_reference_image(class_id):
    """Finds a clear reference image for the given class."""
    class_dir = CLEAR_DATA_DIR / "train" / f"{class_id:05d}"
    if not class_dir.exists():
        return None
    
    images = list(class_dir.glob("*.ppm"))
    if not images:
        return None
        
    # Pick a random one or the first one
    return random.choice(images)

def main():
    st.sidebar.title("🌧️ Rain Command Center")
    
    stats = get_stats()
    st.sidebar.markdown("### Status")
    for k, v in stats.items():
        st.sidebar.metric(k, v)
        
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "⚙️ Pipeline", "🏷️ Labeling"])
    
    # --- TAB 1: DASHBOARD ---
    with tab1:
        st.markdown("### Project Overview")
        st.markdown(f"""
        This dashboard allows you to manage the entire lifecycle of the 'See in Rain' project.
        - **Total Classes**: {NUM_CLASSES}
        - **Current Configured Epochs**: {EPOCHS}
        """)
        
        # Simple chart if data exists
        if stats['Clear Images'] > 0:
            chart_data = pd.DataFrame({
                'Dataset': ['Clear', 'Rainy Pool', 'Waiting Label'],
                'Count': [stats['Clear Images'], stats['Rainy Pool'], stats['To Label']]
            })
            st.bar_chart(chart_data.set_index('Dataset'))

    # --- TAB 2: PIPELINE ---
    with tab2:
        st.header("Pipeline Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1. Rain Generation")
            st.markdown("Generate synthetic rainy data from the clear dataset.")
            if st.button("Generate Rain 🌧️"):
                run_command("python scripts/generate_rain.py", "Rain Generation")
                st.rerun()

        with col2:
            st.subheader("2. Train Model")
            st.markdown(f"Train ResNet18 on the clear dataset (Epochs: {EPOCHS}).")
            if st.button("Run Training 🚀"):
                run_command("python train.py", "Model Training")

        with col3:
            st.subheader("3. Active Learning")
            st.markdown("Identify the most uncertain rainy samples.")
            if st.button("Find Uncertain Samples 🔍"):
                run_command("python active_learning.py", "Uncertainty Sampling")
                st.rerun()

    # --- TAB 3: LABELING ---
    with tab3:
        st.header("Human-in-the-Loop Labeling")
        
        if not UNCERTAIN_SAMPLES_DIR.exists():
            st.warning("No uncertain samples directory found.")
            return

        images = list(UNCERTAIN_SAMPLES_DIR.glob("*.ppm"))
        
        if not images:
            st.success("🎉 No images left to label! You've cleared the queue.")
            if st.button("Refresh Queue"):
                st.rerun()
            return

        # Labeling Interface
        img_path = images[0]
        
        # Main Layout: Image Left, Controls Right
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.image(str(img_path), caption=f"Uncertain: {img_path.name}", width=400)
            
        with c2:
            st.info("👇 Select the correct class for the image.")
            
            # Reference Viewer
            with st.expander("Show Reference Class Helper", expanded=True):
                # Using names in selectbox
                options = [f"{i}: {GTSRB_CLASSES.get(i, 'Unknown')}" for i in range(NUM_CLASSES)]
                selected_option = st.selectbox("Preview Reference Class", options)
                
                # Extract ID from string "0: Name"
                preview_class = int(selected_option.split(":")[0])
                
                ref_img = get_reference_image(preview_class)
                if ref_img:
                    st.image(str(ref_img), caption=f"Reference: {GTSRB_CLASSES[preview_class]}", width=150)
                else:
                    st.warning("No reference image found.")

            st.write("### Assign Label")
            
            # Grid of buttons with names
            # 3 columns to fit names better
            btn_cols = st.columns(3)
            for i in range(NUM_CLASSES):
                class_name = GTSRB_CLASSES.get(i, f"Class {i}")
                with btn_cols[i % 3]:
                    # Button label: "ID: Name"
                    if st.button(f"{i}: {class_name}", key=f"cls_{i}", use_container_width=True):
                        move_image(img_path, i)
                        st.rerun()
            
            st.divider()
            if st.button("Skip / Not Sure"):
                skipped_dir = UNCERTAIN_SAMPLES_DIR.parent / "skipped"
                skipped_dir.mkdir(exist_ok=True)
                shutil.move(str(img_path), str(skipped_dir / img_path.name))
                st.rerun()

def move_image(img_path, class_id):
    """Moves image to the training directory of the selected class."""
    target_dir = CLEAR_DATA_DIR / "train" / f"{class_id:05d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / img_path.name
    shutil.move(str(img_path), str(target_path))
    class_name = GTSRB_CLASSES.get(class_id, f"Class {class_id}")
    st.toast(f"✅ Labeled as: {class_name}")

if __name__ == "__main__":
    main()

