import streamlit as st
import os
import torch
from pathlib import Path
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import UNCERTAIN_SAMPLES_DIR, NUM_CLASSES, PROJECT_ROOT, DEVICE
from src.hitl.annotation_store import AnnotationStore
from src.models.utils import load_checkpoint
from src.models.resnet_model import get_model

# Map class ID to name
CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

def show_labeling_interface(store):
    st.markdown("### Human-in-the-Loop Annotation Interface")
    
    # Load images
    if not UNCERTAIN_SAMPLES_DIR.exists():
        st.warning(f"No uncertain samples found in {UNCERTAIN_SAMPLES_DIR}")
        return
        
    images = list(UNCERTAIN_SAMPLES_DIR.glob("*"))
    images = [img for img in images if img.suffix.lower() in ['.png', '.jpg', '.jpeg', '.ppm']]
    
    # Filter out already annotated
    images_to_label = [img for img in images if not store.is_annotated(img.name)]
    
    st.sidebar.metric("Total Images", len(images))
    st.sidebar.metric("Annotated", store.get_count())
    st.sidebar.metric("Remaining", len(images_to_label))
    
    if not images_to_label:
        st.success("All images annotated! 🎉")
        return
        
    # Pick the first one
    current_image_path = images_to_label[0]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(str(current_image_path), width=300, caption=current_image_path.name)
        
    with col2:
        st.write("### Select Label")
        
        # Dropdown for class selection
        options = [f"{k}: {v}" for k, v in CLASS_NAMES.items()]
        selected_option = st.selectbox("Class", options)
        
        selected_id = int(selected_option.split(":")[0])
        selected_name = selected_option.split(":")[1].strip()
        
        if st.button("Submit Label", type="primary"):
            store.add_annotation(current_image_path.name, selected_id, selected_name)
            st.rerun()
            
        if st.button("Skip"):
            store.add_annotation(current_image_path.name, -1, "Skipped")
            st.rerun()

@st.cache_resource
def load_model_and_data():
    # Load Model
    model = get_model(device=DEVICE, num_classes=NUM_CLASSES)
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "baseline", "best_model.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        saved_metrics = checkpoint.get('metrics', {})
    else:
        epoch = 0
        saved_metrics = {}
        
    model.eval()
    
    # Load Validation Data (Clear)
    from src.data.gtsrb_dataset import get_dataloader
    from src.config import CLEAR_DATA_DIR
    val_loader = get_dataloader(str(CLEAR_DATA_DIR / "val"), batch_size=32, split="val", shuffle=True)
    
    return model, val_loader, epoch, saved_metrics

@st.cache_data
def load_tensorboard_logs():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    log_dir = os.path.join(PROJECT_ROOT, "logs", "baseline")
    if not os.path.exists(log_dir):
        return None
        
    # Find the latest run (or just load all)
    # Assuming one run for now or just taking the dir
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Extract scalars
    tags = event_acc.Tags()['scalars']
    
    data = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {"steps": steps, "values": values}
        
    return data

def show_metrics_interface(store):
    st.markdown("### Model Performance & Predictions")
    
    # --- TensorBoard Integration ---
    st.subheader("Training History (TensorBoard)")
    
    tb_data = load_tensorboard_logs()
    
    if tb_data:
        # Plot Loss
        fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
        if 'Train/EpochLoss' in tb_data:
            sns.lineplot(x=tb_data['Train/EpochLoss']['steps'], y=tb_data['Train/EpochLoss']['values'], label='Train Loss', ax=ax_loss)
        if 'Val/EpochLoss' in tb_data:
            sns.lineplot(x=tb_data['Val/EpochLoss']['steps'], y=tb_data['Val/EpochLoss']['values'], label='Val Loss', ax=ax_loss)
        ax_loss.set_title("Loss Curves")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        st.pyplot(fig_loss)
        
        # Plot Accuracy
        fig_acc, ax_acc = plt.subplots(figsize=(10, 4))
        if 'Train/EpochAcc' in tb_data:
            sns.lineplot(x=tb_data['Train/EpochAcc']['steps'], y=tb_data['Train/EpochAcc']['values'], label='Train Acc', ax=ax_acc)
        if 'Val/EpochAcc' in tb_data:
            sns.lineplot(x=tb_data['Val/EpochAcc']['steps'], y=tb_data['Val/EpochAcc']['values'], label='Val Acc', ax=ax_acc)
        ax_acc.set_title("Accuracy Curves")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        st.pyplot(fig_acc)
        
        # Advanced Metrics (LR, etc.)
        with st.expander("Advanced Metrics"):
            if 'LR' in tb_data:
                st.line_chart(data={'LR': tb_data['LR']['values']})
                
    else:
        st.info("No TensorBoard logs found.")
        
    st.divider()

    model, val_loader, epoch, saved_metrics = load_model_and_data()
    
    if epoch == 0:
        st.warning("No trained model found.")
        return

    # 1. Metrics
    st.subheader(f"Metrics (Epoch {epoch})")
    
    # Compute metrics on a small subset for display if not fully available or to show current state
    # We'll run one pass over a few batches to get a quick estimate and samples
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    samples_to_show = []
    
    with torch.no_grad():
        # Just check 2 batches for speed in UI
        for i, (inputs, targets) in enumerate(val_loader):
            if i >= 2: break
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            _, preds = torch.max(outputs, 1)
            
            total_loss += loss.item() * inputs.size(0)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)
            
            # Collect samples from first batch
            if i == 0:
                # Move to cpu for plotting
                imgs = inputs.cpu()
                lbls = targets.cpu()
                prds = preds.cpu()
                
                for j in range(min(16, len(imgs))):
                    samples_to_show.append({
                        "img": imgs[j],
                        "true": lbls[j].item(),
                        "pred": prds[j].item()
                    })

    val_loss = total_loss / total
    val_acc = correct / total
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Validation Accuracy (Est.)", f"{val_acc:.4f}")
    with m_col2:
        st.metric("Validation Loss (Est.)", f"{val_loss:.4f}")
    with m_col3:
        st.metric("Saved Best Acc", f"{saved_metrics.get('val_acc', 0):.4f}")

    st.divider()

    # 2. Sample Predictions
    st.subheader("Sample Predictions (Validation Set)")
    
    # Display grid
    cols = st.columns(4)
    for idx, sample in enumerate(samples_to_show):
        col = cols[idx % 4]
        
        # Un-normalize image for display
        # GTSRB mean/std from config
        mean = torch.tensor([0.3337, 0.3064, 0.3171]).view(3, 1, 1)
        std = torch.tensor([0.2672, 0.2564, 0.2629]).view(3, 1, 1)
        img_tensor = sample["img"] * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Convert to PIL
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        true_name = CLASS_NAMES.get(sample['true'], str(sample['true']))
        pred_name = CLASS_NAMES.get(sample['pred'], str(sample['pred']))
        
        color = "green" if sample['true'] == sample['pred'] else "red"
        
        with col:
            st.image(img_np, clamp=True)
            st.markdown(f"**True**: {true_name}")
            st.markdown(f":{color}[**Pred**: {pred_name}]")

    st.divider()
    
    # 3. Annotation Statistics
    st.subheader("Annotation Progress")
    
    annotations = store.annotations
    if not annotations:
        st.write("No annotations yet.")
    else:
        # Count labels
        label_counts = {}
        for ann in annotations:
            label = ann.get('label_name', 'Unknown')
            label_counts[label] = label_counts.get(label, 0) + 1
            
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=list(label_counts.values()), y=list(label_counts.keys()), ax=ax)
        ax.set_title("Annotated Class Distribution")
        ax.set_xlabel("Count")
        st.pyplot(fig)

def main():
    st.set_page_config(page_title="Teach Me to See in the Rain", layout="wide")
    st.title("Teach Me to See in the Rain 🌧️")
    
    store = AnnotationStore()
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Labeling", "Metrics"])
    
    if page == "Labeling":
        show_labeling_interface(store)
    elif page == "Metrics":
        show_metrics_interface(store)

if __name__ == "__main__":
    main()
