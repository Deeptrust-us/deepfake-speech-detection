# Quick Start Guide - Running HM-Conformer with MultilingualDataset

## Option 1: Run with Docker (Recommended)

### Step 1: Build Docker Image
```bash
cd /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer
sudo docker build -t env202305 ./docker/Dockerfile
```

This will take a few minutes the first time.

### Step 2: Update arguments.py for Docker Paths

Edit `hm_conformer/arguments.py` and change the paths to Docker paths:

```python
# Comment out local paths:
# 'path_train'    : '/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios',
# 'labels_path'   : '/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/labels.json',
# 'dataset_root'  : '/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios',

# Uncomment Docker paths:
'path_train'    : '/dataset_audios',
'labels_path'   : '/dataset_audios/labels.json',
'dataset_root'  : '/dataset_audios',
```

### Step 3: Run Docker Container

```bash
cd /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer

sudo docker run --gpus all -it --rm --ipc=host \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios:/dataset_audios \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305:/environment \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305/results:/results \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/exp_lib:/exp_lib \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer:/code \
  env202305:latest
```

### Step 4: Run Training Inside Container

Once inside the Docker container, run:

```bash
cd /code
python hm_conformer/main.py
```

---

## Option 2: Run Locally (Without Docker)

### Step 1: Install Dependencies

Make sure you have all required packages:

```bash
cd /Users/usuario/Documents/github/deepfake-speech-detection

# Install Python dependencies
pip install torch torchvision torchaudio
pip install soundfile scipy numpy
pip install neptune-client wandb
pip install transformers datasets huggingface_hub
pip install torch-audiomentations torchsummary
```

### Step 2: Verify arguments.py Uses Local Paths

Make sure `hm_conformer/arguments.py` has local paths (not Docker paths):

```python
# For local usage:
'path_train'    : '/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios',
'labels_path'   : '/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/labels.json',
'dataset_root'  : '/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios',
```

### Step 3: Run Training

```bash
cd /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer

# Make sure exp_lib is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/exp_lib"

# Run training
python hm_conformer/main.py
```

---

## Important Configuration

### Check GPU Settings

In `hm_conformer/arguments.py`, make sure you have:

```python
'usable_gpu': '0,1',  # Change to your available GPUs (e.g., '0' for single GPU)
```

### Check Training/Test Mode

In `hm_conformer/arguments.py`:

```python
'TEST': False,  # Set to False for training, True for inference only
```

### Check Batch Size

The batch size is divided by number of GPUs. For example:
- `batch_size: 240` with 2 GPUs = 120 per GPU
- Adjust based on your GPU memory

---

## Troubleshooting

### Docker Issues

**If Docker can't find GPU:**
```bash
# Check nvidia-docker is installed
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

**If you get permission errors:**
```bash
sudo usermod -aG docker $USER
# Log out and log back in
```

### Local Run Issues

**If you get "ModuleNotFoundError":**
- Make sure all dependencies are installed
- Check PYTHONPATH includes exp_lib directory

**If you get "CUDA out of memory":**
- Reduce batch_size in arguments.py
- Use fewer GPUs

### Path Issues

**If dataset not found:**
- Verify labels.json exists: `ls -lh dataset_audios/labels.json`
- Check paths in arguments.py match your system

---

## Quick Commands Reference

### Docker - One-liner to build and run:
```bash
cd /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer && \
sudo docker build -t env202305 ./docker/Dockerfile && \
sudo docker run --gpus all -it --rm --ipc=host \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios:/dataset_audios \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305:/environment \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305/results:/results \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/exp_lib:/exp_lib \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer:/code \
  env202305:latest
```

### Check if labels.json exists:
```bash
ls -lh /Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/labels.json
```

---

## Next Steps

After training starts, you should see:
1. Dataset information printed (train/val/test split counts)
2. Training progress with loss values
3. Results saved to `results/` directory

For more details, see `DOCKER_SETUP.md` in the HM-Conformer directory.

