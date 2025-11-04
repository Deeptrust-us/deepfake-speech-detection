#!/bin/bash
# Launch script for HM-Conformer with Docker
# 
# Usage:
#   ./launch/launch.sh
#
# You need to set the following environment variables or replace them in the command:
#   - PATH_DB: Path to your datasets (for ASVspoof datasets)
#   - PATH_HM-Conformer: Path to HM-Conformer directory
#   - PATH_DATASET_AUDIOS: Path to dataset_audios directory (for MultilingualDataset)

# OLD: For ASVspoof datasets
# sudo docker run --gpus all -it --rm --ipc=host -v {PATH_DB}:/data -v \
# {PATH_HM-Conformer}/env202305:/environment -v \
# {PATH_HM-Conformer}/env202305/results:/results -v \
# {PATH_HM-Conformer}/exp_lib:/exp_lib -v \
# {PATH_HM-Conformer}:/code env202305:latest

# NEW: For MultilingualDataset
# Replace {PATH_HM-Conformer} and {PATH_DATASET_AUDIOS} with your actual paths
# Example:
#   PATH_HM-Conformer=/Users/usuario/Documents/github/deepfake-speech-detection/HM-Conformer
#   PATH_DATASET_AUDIOS=/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios

sudo docker run --gpus all -it --rm --ipc=host \
  -v {PATH_DATASET_AUDIOS}:/dataset_audios \
  -v {PATH_HM-Conformer}/env202305:/environment \
  -v {PATH_HM-Conformer}/env202305/results:/results \
  -v {PATH_HM-Conformer}/exp_lib:/exp_lib \
  -v {PATH_HM-Conformer}:/code \
  env202305:latest

# Inside the container, the dataset_audios will be available at /dataset_audios
# Make sure to update arguments.py to use:
#   'labels_path': '/dataset_audios/labels.json'
#   'dataset_root': '/dataset_audios'
