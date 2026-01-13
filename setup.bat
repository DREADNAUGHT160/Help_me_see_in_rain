@echo off
echo ==========================================
echo Setting up Teach Me to See in Rain Project
echo ==========================================

echo [1/3] Installing dependencies...
pip install -r requirements.txt

echo [2/3] Preparing GTSRB Dataset (Clear Data)...
python scripts/prepare_gtsrb.py

echo [3/3] Generating Rainy Dataset (Unlabeled Pool)...
python scripts/generate_rain.py

echo ==========================================
echo Setup Complete!
echo You can now launch the app with:
echo streamlit run hitl/app.py
echo ==========================================
pause
