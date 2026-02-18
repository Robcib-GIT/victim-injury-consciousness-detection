conda create -n triage_env python=3.10 -y
conda activate triage_env
python -m pip install --upgrade pip

pip install opencv-python numpy requests websocket-client ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install pyttsx3