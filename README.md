# Deepfake Speech Detection

This repository contains experiments for deepfake voice detection. Due to the recent increase in generative models capable of producing very realistic synthetic audio, we seek to develop a model capable of detecting this new type of synthetic audio.

## Datasets

- **MLAAD Dataset** ([Hugging Face](https://huggingface.co/datasets/mueller91/MLAAD)): Contains synthetic voice samples for training deepfake detection models
- **M-AILABS Dataset** ([GitHub](https://github.com/imdatceleste/m-ailabs-dataset)): Contains real voice samples to complement the synthetic data

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. For MLAAD dataset access, you may need to authenticate with Hugging Face:
```bash
huggingface-cli login
```

## Current Status

Currently, this repository contains only a `dataset-exploration.py` file for downloading and exploring the datasets.

## Usage

Run the dataset exploration script to download both datasets:
```bash
python dataset-exploration.py
```

This will download:
- MLAAD dataset to `MLAAD_local/` directory
- M-AILABS dataset to `stt_tts_dataset/` directory
