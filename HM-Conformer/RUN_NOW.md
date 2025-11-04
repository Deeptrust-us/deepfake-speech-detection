# How to Run HM-Conformer NOW

## Quick Start (3 Steps)

### Step 1: Build Docker Image (First time only)
```bash
cd /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer
sudo docker build -t env202305 ./docker/Dockerfile
```
⏱️ This takes 5-10 minutes the first time.

### Step 2: Run Docker Container
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

### Step 3: Run Training Inside Container
Once you're inside the Docker container, run:
```bash
cd /code
python hm_conformer/main.py
```

---

## One-Liner (All Steps Combined)

If you want to do everything in one command:

```bash
cd /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer && \
sudo docker build -t env202305 ./docker/Dockerfile && \
sudo docker run --gpus all -it --rm --ipc=host \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios:/dataset_audios \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305:/environment \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/env202305/results:/results \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer/exp_lib:/exp_lib \
  -v /Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer:/code \
  env202305:latest bash -c "cd /code && python hm_conformer/main.py"
```

---

## Important Notes

1. **GPU Required**: Make sure you have NVIDIA GPUs available
2. **Paths**: The script automatically mounts your dataset_audios directory
3. **Arguments**: Make sure `arguments.py` has Docker paths (already done)
4. **GPU Selection**: Edit `arguments.py` to set `'usable_gpu': '0'` or `'0,1'` for your GPUs

---

## Troubleshooting

**No GPU?** Remove `--gpus all` (but training will be very slow)

**Permission error?** Use `sudo` or add user to docker group:
```bash
sudo usermod -aG docker $USER
```

**Dataset not found?** Check that labels.json exists:
```bash
ls -lh /Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/labels.json
```
