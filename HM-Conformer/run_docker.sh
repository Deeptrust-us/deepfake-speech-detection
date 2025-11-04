#!/bin/bash
# Quick script to run HM-Conformer with Docker
# Usage: ./run_docker.sh

set -e

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
DOCKER_IMAGE="env202305"
DATASET_PATH="$PROJECT_ROOT/dataset_audios"
RESULTS_PATH="$SCRIPT_DIR/env202305/results"
ENV_PATH="$SCRIPT_DIR/env202305"

# Check if dataset exists
if [ ! -f "$DATASET_PATH/labels.json" ]; then
    echo "ERROR: labels.json not found at $DATASET_PATH/labels.json"
    echo "Please make sure the dataset is organized and labels.json exists."
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_PATH"

echo "=========================================="
echo "HM-Conformer Docker Runner"
echo "=========================================="
echo "Dataset: $DATASET_PATH"
echo "Results: $RESULTS_PATH"
echo "=========================================="
echo ""

# Check if Docker image exists
if ! sudo docker images | grep -q "$DOCKER_IMAGE"; then
    echo "Docker image not found. Building..."
    cd "$SCRIPT_DIR"
    sudo docker build -t "$DOCKER_IMAGE" ./docker/Dockerfile
    echo ""
fi

# Run Docker container
echo "Starting Docker container..."
echo ""

sudo docker run --gpus all -it --rm --ipc=host \
  -v "$DATASET_PATH:/dataset_audios" \
  -v "$ENV_PATH:/environment" \
  -v "$RESULTS_PATH:/results" \
  -v "$SCRIPT_DIR/exp_lib:/exp_lib" \
  -v "$SCRIPT_DIR:/code" \
  "$DOCKER_IMAGE:latest" \
  bash -c "cd /code && python hm_conformer/main.py"

