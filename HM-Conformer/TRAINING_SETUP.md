# Training Setup Confirmation ✅

## Your Configuration is Ready for Training!

### ✅ What's Configured:

1. **Dataset**: MultilingualDataset loading from `labels.json`
   - Your dataset at: `/dataset_audios` (inside Docker)
   - Labels file: `/dataset_audios/labels.json`
   - Audio files: `/dataset_audios/audio/real/` and `/dataset_audios/audio/fake/`

2. **Training Mode**: `TEST = False` ✅
   - This will **TRAIN** the model (not just test)
   - Will train for **200 epochs**
   - Will save checkpoints during training

3. **Data Split**: 
   - **80%** for training
   - **10%** for validation
   - **10%** for testing

4. **Training Configuration**:
   - Batch size: 240 (divided by number of GPUs)
   - Learning rate: 1e-6
   - Epochs: 200
   - Early stopping: After 6 evaluations without improvement

### 📊 What Will Happen During Training:

1. **Dataset Loading**: 
   - Loads all audio files from `labels.json`
   - Splits into train/val/test sets
   - Prints dataset statistics

2. **Training Loop**:
   - For each epoch (1-200):
     - Train on training set
     - Calculate loss and metrics
     - Every 5 epochs: Evaluate on test set
     - Save best model checkpoints
     - Early stop if no improvement for 6 evaluations

3. **Model Checkpoints**:
   - Saved to: `/results/` (inside Docker)
   - Which maps to: `HM-Conformer/env202305/results/` (on your host)
   - Best models saved with epoch number

### 🚀 To Start Training:

```bash
cd /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer

# Build Docker image (first time only)
sudo docker build -t env202305 ./docker/Dockerfile

# Run Docker container and start training
sudo docker run --gpus all -it --rm --ipc=host \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios:/dataset_audios \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305:/environment \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305/results:/results \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/exp_lib:/exp_lib \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer:/code \
  env202305:latest bash -c "cd /code && python hm_conformer/main.py"
```

### 📝 Important Settings:

**GPU Configuration** (in `arguments.py`):
```python
'usable_gpu': '0',  # Change to '0,1' for 2 GPUs, '0,1,2,3' for 4 GPUs, etc.
```

**Training Parameters** (in `arguments.py`):
```python
'epoch': 200,           # Number of training epochs
'batch_size': 240,      # Total batch size (divided by # GPUs)
'lr': 1e-6,             # Learning rate
'rand_seed': 1,         # Random seed for reproducibility
```

### 📂 Output Files:

After training, you'll find:
- **Model checkpoints**: `HM-Conformer/env202305/results/check_point_DF_*.pt`
- **Training logs**: `HM-Conformer/env202305/results/` (various log files)
- **Best model**: Saved when validation improves

### ⚠️ Before Training:

1. **Check GPU**: Make sure you have GPUs available
   ```bash
   nvidia-smi
   ```

2. **Check Dataset**: Verify labels.json exists
   ```bash
   ls -lh /Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/labels.json
   ```

3. **Check Disk Space**: Training will save checkpoints, make sure you have enough space

### 🎯 Training Progress:

You'll see output like:
```
====================
  MultilingualDataset
====================
TRAIN: Real=XXX, Fake=XXX, Total=XXX
VAL:   Real=XXX, Fake=XXX, Total=XXX
TEST:  Real=XXX, Fake=XXX, Total=XXX
====================

[1|(loss): 0.1234]
[2|(loss): 0.1123]
...
```

This confirms training is running with your MultilingualDataset! 🎉

