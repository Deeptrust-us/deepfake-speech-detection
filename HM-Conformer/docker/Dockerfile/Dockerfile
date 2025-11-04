FROM nvcr.io/nvidia/pytorch:22.08-py3

# Set timezone to avoid interactive prompt during build
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

# Install system dependencies for audio processing
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata libsndfile1 git-lfs ffmpeg

# PyTorch is already installed in the base image (nvcr.io/nvidia/pytorch:22.08-py3)
# The base image includes PyTorch, torchaudio, and torchvision
# If you need specific versions, uncomment and adjust:
# RUN pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install pip --upgrade

# Verify PyTorch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN pip install neptune-client
RUN pip install wandb
# Install transformers from PyPI (stable version) instead of git to avoid dependency conflicts
RUN pip install transformers[torch]
RUN pip install datasets
RUN pip install huggingface_hub
RUN pip install torchsummary
RUN pip install torch-audiomentations

# Install additional dependencies for MultilingualDataset
RUN pip install soundfile
RUN pip install scipy