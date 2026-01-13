# Teach Me to See in the Rain 🌧️

A Human-in-the-Loop (HITL) Active Learning system for traffic sign recognition under rainy conditions.

## Project Goal
Improve the robustness of a traffic sign classifier (ResNet-18) on rainy images by iteratively selecting the most uncertain samples for human annotation.

## Setup

### One-Click Setup (Recommended)
**Windows**:
Double-click `setup.bat` or run:
```bash
setup.bat
```

**Linux/Mac**:
```bash
chmod +x setup.sh
./setup.sh
```

This will automatically:
1. Install all Python dependencies.
2. Download and prepare the GTSRB dataset.
3. Generate the synthetic rainy dataset.

---

### Manual Setup (Optional)
If you prefer to run steps manually:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare Data**:
   ```bash
   python scripts/prepare_gtsrb.py
   ```
3. **Generate Rain**:
   ```bash
   python scripts/generate_rain.py
   ```

## Usage

### 1. Train Baseline
Train the model on clear data only:
```bash
python train.py
```
This will save the trained model to `model.pth`.

### 2. Active Learning Loop
Run the active learning loop to select uncertain rainy images:
```bash
python active_learning.py
```
This will:
- Load the baseline model (`model.pth`).
- Predict on the rainy pool (`data/rainy`).
- Select top-k uncertain samples.
- Save them to `data/hits/uncertain_samples`.

### 3. Human-in-the-Loop Labeling
Launch the Streamlit UI to label the selected images:
```bash
streamlit run hitl/app.py
```
- Label images via the UI.
- Labeled images are moved to `data/clear/train` to improve the dataset.

### 4. Retrain
After labeling, re-run training to improve the model:
```bash
python train.py
```
