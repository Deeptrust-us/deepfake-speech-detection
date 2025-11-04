#!/bin/bash
# Quick script to start training with your MultilingualDataset

echo "=========================================="
echo "Starting HM-Conformer Training"
echo "=========================================="
echo ""

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
    echo ""
fi

# Check if labels.json exists
if [ ! -f "/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/labels.json" ]; then
    echo "ERROR: labels.json not found!"
    exit 1
fi

echo "Starting Docker container..."
echo ""

sudo docker run --gpus all -it --rm --ipc=host \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios:/dataset_audios \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305:/environment \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305/results:/results \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/exp_lib:/exp_lib \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer:/code \
  env202305:latest bash -c "cd /code && python hm_conformer/main.py"
