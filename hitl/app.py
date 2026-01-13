import streamlit as st
import os
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import random
import json
import time

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
    
    with st.status(f"Running {desc}...", expanded=True) as status:
        st.write("Initializing...")
        
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
            status.update(label=f"✅ {desc} Completed!", state="complete", expanded=False)
            st.success(f"{desc} Finished Successfully.")
            return True
        else:
            stderr = process.stderr.read()
            status.update(label=f"❌ {desc} Failed!", state="error", expanded=True)
            st.error(f"Error: {stderr}")
            return False

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
        
        # Top level metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Classes", NUM_CLASSES)
        m2.metric("Clear Images", stats['Clear Images'])
        m3.metric("Rainy Pool", stats['Rainy Pool'])

        st.divider()
        
        # Training History
        log_file = PROJECT_ROOT / "logs" / "training_history.csv"
        if log_file.exists():
            df = pd.read_csv(log_file)
            
            st.subheader("📈 Training Progress")
            
            # Key Stats
            best_acc = df['Val_Acc'].max() if 'Val_Acc' in df.columns else (df['Accuracy'].max() if 'Accuracy' in df.columns else 0)
            latest_loss = df['Val_Loss'].iloc[-1] if 'Val_Loss' in df.columns else (df['Loss'].iloc[-1] if 'Loss' in df.columns else 0)
            total_runs = len(df)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Best Val Accuracy", f"{best_acc:.2f}%")
            k2.metric("Latest Val Loss", f"{latest_loss:.4f}")
            k3.metric("Total Runs", total_runs)
            
            # Charts
            # Check if we have new format or old format
            if 'Val_Acc' in df.columns:
                 # Comparison Chart
                chart_df = df[['Run_ID', 'Train_Acc', 'Val_Acc']].copy()
                chart_df.set_index('Run_ID', inplace=True)
                st.bar_chart(chart_df)
            else:
                st.line_chart(df.set_index('Epoch')[['Accuracy', 'Loss']])
            
            with st.expander("View Raw Logs"):
                st.dataframe(df.sort_values(by=['Timestamp'], ascending=False), use_container_width=True)
        else:
            st.info("No training logs found yet. Run training to see metrics here.")

    # --- TAB 2: PIPELINE ---
    with tab2:
        st.header("Pipeline Controls")
        
        # Configuration
        with st.expander("⚙️ Configuration", expanded=True):
            st.write("Customize your pipeline parameters.")
            c1, c2 = st.columns(2)
            with c1:
                strategy = st.selectbox(
                    "Active Learning Strategy", 
                    ["entropy", "least_confidence", "margin", "random"],
                    index=0,
                    help="Criteria for selecting uncertain samples."
                )
            with c2:
                epochs = st.slider("Training Epochs", min_value=1, max_value=50, value=EPOCHS)
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1. Rain Generation")
            st.markdown("Generate synthetic rainy data.")
            if st.button("Generate Rain 🌧️", use_container_width=True):
                if run_command("python scripts/generate_rain.py", "Rain Generation"):
                    time.sleep(1)
                    st.rerun()

        with col2:
            st.subheader("2. Train Model")
            st.markdown(f"Train ResNet18 on clear data.")
            if st.button("Run Training 🚀", use_container_width=True):
                run_command(f"python train.py --epochs {epochs}", "Model Training")

        with col3:
            st.subheader("3. Active Learning")
            st.markdown("Identify uncertain samples.")
            if st.button("Find Uncertain Samples 🔍", use_container_width=True):
                cmd = f"python active_learning.py --strategy {strategy}"
                if run_command(cmd, "Uncertainty Sampling"):
                    time.sleep(1)
                    st.rerun()

        st.divider()
        
        # Danger Zone
        with st.expander("🚨 Danger Zone", expanded=False):
            if st.button("Reset Project (Clear Logs & Models)", type="primary"):
                # Delete logs
                if (PROJECT_ROOT / "logs").exists():
                    shutil.rmtree(PROJECT_ROOT / "logs")
                # Delete model
                if (PROJECT_ROOT / "model.pth").exists():
                    (PROJECT_ROOT / "model.pth").unlink()
                # Delete metrics
                if (UNCERTAIN_SAMPLES_DIR / "metrics.json").exists():
                    (UNCERTAIN_SAMPLES_DIR / "metrics.json").unlink()
                    
                st.warning("Project reset! Please train a new model.")
                time.sleep(2)
                st.rerun()

    # --- TAB 3: LABELING ---
    with tab3:
        st.header("Human-in-the-Loop Labeling")
        
        # Check if we have samples
        images = list(UNCERTAIN_SAMPLES_DIR.glob("*.ppm"))
        metrics_file = UNCERTAIN_SAMPLES_DIR / "metrics.json"
        
        if not images or not metrics_file.exists():
            st.info("👋 No uncertain samples found waiting for review.")
            st.markdown("""
            **How to get started:**
            1. Go to the **Pipeline** tab.
            2. Train a model (if not already done).
            3. Click **Find Uncertain Samples**.
            """)
            return
        else:
            # Load first image
            img_path = images[0]
        
        # Load Metrics if available
        metrics = {}
        metrics_file = UNCERTAIN_SAMPLES_DIR / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        
        img_metric = metrics.get(img_path.name, {})
        score = img_metric.get("score", "N/A")
        strategy_used = img_metric.get("strategy", "Unknown")
        pred_class_id = img_metric.get("predicted_class", None)
        confidence = img_metric.get("confidence", None)
        
        pred_text = "N/A"
        if pred_class_id is not None:
             class_name = GTSRB_CLASSES.get(pred_class_id, f"Class {pred_class_id}")
             pred_text = f"**{class_name}** ({confidence*100:.1f}%)"

        # Main Layout: Image Left, Controls Right
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.image(str(img_path), width=400)
            st.caption(f"**File:** {img_path.name}")
            
            # Metric Card
            st.info(f"""
            **Uncertainty Score**: {score}  
            *(Strategy: {strategy_used})*  
            
            **Model Prediction**:  
            {pred_text}
            """)
            
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

