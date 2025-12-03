# Teach Me to See in the Rain 🌧️

A Human-in-the-Loop (HITL) Active Learning system for traffic sign recognition under rainy conditions.

## Project Goal
Improve the robustness of a traffic sign classifier (ResNet-18) on rainy images by iteratively selecting the most uncertain samples for human annotation.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```
   > [!NOTE]
   > This project supports CUDA (NVIDIA GPUs) and MPS (Mac Silicon). The device is automatically detected in `src/config.py`.

2. **Prepare Data**:
   Download GTSRB and create the clear dataset:
   ```bash
   python scripts/prepare_gtsrb.py
   ```
   Generate synthetic rainy data (unlabeled pool):
   ```bash
   python scripts/make_rainy_dataset.py
   ```

## Usage

### 1. Train Baseline
Train the model on clear data only:
```bash
python src/experiments/run_baseline.py
```
This will save checkpoints to `checkpoints/baseline/` and logs to `logs/baseline/`.

### 2. Active Learning Loop
Run the active learning loop to select uncertain rainy images:
```bash
python src/experiments/run_active_loop.py
```
This will:
- Load the baseline model.
- Predict on the rainy pool.
- Select top-k uncertain samples.
- Save them to `data/hits/uncertain_samples`.

### 3. Human-in-the-Loop Labeling
Launch the Streamlit UI to label the selected images:
```bash
streamlit run src/hitl/streamlit_app.py
```
- Label images via the UI.
- Annotations are saved to `data/hits/annotations.json`.

### 4. Retrain (Future Work)
After labeling, you can merge the annotations and fine-tune the model (logic to be implemented in `run_active_loop.py` or a new script).

## Testing
Run unit tests:
```bash
pytest tests/
```

## Docker
Build and run with Docker:
```bash
docker build -t teach_me_rain -f docker/Dockerfile.cpu .
docker run -it teach_me_rain
```
