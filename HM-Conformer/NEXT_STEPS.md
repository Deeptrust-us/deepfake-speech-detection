# Next Steps - Start Training! 🚀

## ✅ Docker Image Built Successfully!

Your Docker image `env202305` is ready. Now you can start training!

## Step 1: Configure GPU (if needed)

Edit `hm_conformer/arguments.py` and set your GPU:

```python
'usable_gpu': '0',  # Use GPU 0 (single GPU)
# OR
'usable_gpu': '0,1',  # Use GPUs 0 and 1 (multi-GPU)
```

**Important**: If you're on macOS, you won't have NVIDIA GPUs. You'll need to:
- Remove `--gpus all` from the Docker command
- Training will be slower (CPU only)

## Step 2: Start Training

### Option 1: Use the Quick Script (Easiest)

```bash
cd /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer
./START_TRAINING.sh
```

### Option 2: Manual Docker Command

```bash
cd /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer

sudo docker run --gpus all -it --rm --ipc=host \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios:/dataset_audios \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305:/environment \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305/results:/results \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/exp_lib:/exp_lib \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer:/code \
  env202305:latest bash -c "cd /code && python hm_conformer/main.py"
```

### Option 3: Interactive Mode (For Testing)

```bash
cd /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer

sudo docker run --gpus all -it --rm --ipc=host \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios:/dataset_audios \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305:/environment \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305/results:/results \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/exp_lib:/exp_lib \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer:/code \
  env202305:latest

# Inside container:
cd /code
python hm_conformer/main.py
```

## Step 3: What You'll See

When training starts, you should see:

```
====================
  MultilingualDataset
====================
TRAIN: Real=XXX, Fake=XXX, Total=XXX
VAL:   Real=XXX, Fake=XXX, Total=XXX
TEST:  Real=XXX, Fake=XXX, Total=XXX
Class weights: Real=X.XXXX, Fake=X.XXXX
====================

[1|(loss): 0.1234]  # Training epoch 1
[2|(loss): 0.1123]  # Training epoch 2
...
```

## Important Notes

### For macOS Users:
- **No NVIDIA GPU support**: Docker Desktop on macOS doesn't support NVIDIA GPUs
- **Remove `--gpus all`**: Training will use CPU (very slow)
- **Alternative**: Use a Linux machine with NVIDIA GPUs, or use cloud GPU services

### For Linux Users with NVIDIA GPUs:
- Make sure `nvidia-docker2` is installed
- Verify GPU: `nvidia-smi`
- Training should work with GPU acceleration

### Training Configuration:
- **Epochs**: 200 (configurable in `arguments.py`)
- **Batch size**: 240 (divided by number of GPUs)
- **Early stopping**: After 6 evaluations without improvement
- **Checkpoints**: Saved every 5 epochs when validation improves

### Output Location:
- **Model checkpoints**: `HM-Conformer/env202305/results/check_point_DF_*.pt`
- **Training logs**: `HM-Conformer/env202305/results/`

## Troubleshooting

### If you get "CUDA out of memory":
- Reduce `batch_size` in `arguments.py`
- Use fewer GPUs

### If you get "Dataset not found":
- Verify `labels.json` exists
- Check Docker volume mounts are correct

### If training is too slow:
- Make sure you're using GPUs (not CPU)
- Check `nvidia-smi` to see GPU usage

## Ready to Start!

Run the command above and training will begin! 🎉

