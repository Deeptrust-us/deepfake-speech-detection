# Docker Setup for MultilingualDataset

This guide explains how to run HM-Conformer with the new MultilingualDataset using Docker.

## Prerequisites

1. **Docker installed** with NVIDIA GPU support (nvidia-docker2)
2. **NVIDIA drivers** installed
3. **Dataset organized**: Make sure you have run the dataset organization script and have `labels.json` ready

## Step 1: Build the Docker Image

Navigate to the HM-Conformer directory and build the Docker image:

```bash
cd HM-Conformer
./docker/build.sh
```

Or manually:

```bash
sudo docker build -t env202305 ./docker/Dockerfile
```

This will create a Docker image named `env202305` with all necessary dependencies including:
- PyTorch 1.13.0
- Audio processing libraries (soundfile, scipy)
- All HM-Conformer dependencies

## Step 2: Prepare Volume Mounts

Before running the container, you need to identify the paths to mount:

1. **Dataset path**: Path to your `dataset_audios` directory
   - Example: `/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios`
   
2. **HM-Conformer path**: Path to the HM-Conformer directory
   - Example: `/Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer`

3. **Results directory**: Where training results will be saved
   - Default: `{PATH_HM-Conformer}/env202305/results`

## Step 3: Update arguments.py for Docker

Before running, update the paths in `hm_conformer/arguments.py` to use Docker paths:

```python
# In arguments.py, update these paths for Docker:
'path_train'    : '/dataset_audios',
'labels_path'   : '/dataset_audios/labels.json',
'dataset_root'  : '/dataset_audios',
```

The paths inside the container will be:
- `/dataset_audios` - Your dataset_audios directory
- `/code` - HM-Conformer code
- `/exp_lib` - Experiment library
- `/results` - Training results

## Step 4: Run the Docker Container

### Option 1: Using the launch script

Edit `launch/launch.sh` and replace the placeholders:

```bash
# Edit launch/launch.sh and replace:
# {PATH_HM-Conformer} -> /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer
# {PATH_DATASET_AUDIOS} -> /Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios

# Then run:
./launch/launch.sh
```

### Option 2: Manual Docker run command

```bash
sudo docker run --gpus all -it --rm --ipc=host \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios:/dataset_audios \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305:/environment \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305/results:/results \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/exp_lib:/exp_lib \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer:/code \
  env202305:latest
```

**Important**: Replace the paths with your actual paths!

## Step 5: Run Training Inside Container

Once inside the Docker container, run:

```bash
cd /code
python hm_conformer/main.py
```

## Volume Mounts Explained

- `-v {PATH_DATASET_AUDIOS}:/dataset_audios` - Mounts your dataset directory
- `-v {PATH_HM-Conformer}/env202305:/environment` - Environment directory (optional)
- `-v {PATH_HM-Conformer}/env202305/results:/results` - Where results/logs are saved
- `-v {PATH_HM-Conformer}/exp_lib:/exp_lib` - Experiment library code
- `-v {PATH_HM-Conformer}:/code` - HM-Conformer source code

## Troubleshooting

### GPU not detected
Make sure you have nvidia-docker installed:
```bash
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

### Permission issues
You may need to use `sudo` for Docker commands, or add your user to the docker group:
```bash
sudo usermod -aG docker $USER
# Then log out and log back in
```

### Dataset not found
- Verify the dataset_audios directory is correctly mounted
- Check that `labels.json` exists in the mounted directory
- Verify paths in `arguments.py` match the container paths (use `/dataset_audios` not host paths)

### Missing dependencies
If you encounter missing Python packages, rebuild the Docker image or install them inside the container:
```bash
docker exec -it <container_name> pip install <package_name>
```

## Notes

- The `--gpus all` flag enables GPU access
- The `--ipc=host` flag improves performance for multi-GPU training
- The `-it` flags make the container interactive
- The `--rm` flag removes the container when you exit

