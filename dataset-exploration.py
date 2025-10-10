# %% [markdown]
# Deepfake Speech Detection - Dataset Download Script
# This notebook downloads and prepares datasets for deepfake speech detection

# %% [markdown]
# ## Dataset 1: MLAAD Dataset from Hugging Face
# This dataset will be downloaded using the datasets library

# %%
from datasets import load_dataset
import os

# Download from HF and cache
print("Downloading MLAAD dataset from Hugging Face...")
ds = load_dataset("mueller91/MLAAD")

# Optionally: Save the dataset to your own directory
print("Saving MLAAD dataset to local directory...")
ds.save_to_disk("MLAAD_local")

print("MLAAD dataset downloaded and saved successfully!")
print(f"Dataset structure: {ds}")
print(f"Dataset keys: {list(ds.keys())}")

# %% [markdown]
# ## Dataset 2: STT TTS Dataset from TAU-CETI
# This dataset will be downloaded from the provided URL

# %%
import urllib.request
import tarfile
import os

# Dataset URL
dataset_url = "https://ics.tau-ceti.space/data/Training/stt_tts/en_US.tgz"
dataset_filename = "en_US.tgz"
extract_dir = "stt_tts_dataset"

# Create directory for extraction if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Download the dataset
print(f"Downloading STT TTS dataset from {dataset_url}...")
urllib.request.urlretrieve(dataset_url, dataset_filename)
print("Download completed!")

# Extract the tar.gz file
print("Extracting the dataset...")
with tarfile.open(dataset_filename, 'r:gz') as tar:
    tar.extractall(path=extract_dir)

print("STT TTS dataset downloaded and extracted successfully!")
print(f"Dataset extracted to: {extract_dir}")

# Clean up the downloaded tar file
os.remove(dataset_filename)
print("Cleaned up downloaded tar file.")

# %% [markdown]
# ## Summary
# Both datasets have been downloaded and prepared:
# 1. MLAAD dataset saved to 'MLAAD_local' directory
# 2. STT TTS dataset extracted to 'stt_tts_dataset' directory

# %%
print("Dataset download process completed!")
print("Available datasets:")
print("- MLAAD_local/ (Hugging Face MLAAD dataset)")
print("- stt_tts_dataset/ (STT TTS dataset from TAU-CETI)")