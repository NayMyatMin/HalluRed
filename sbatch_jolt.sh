#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|v100-32gb|v100
#SBATCH --time=01-00:00:00
#SBATCH --partition=researchshort
#SBATCH --account=sunjunresearch
#SBATCH --qos=research-1-qos
#SBATCH --job-name=jolt-train
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

set -euo pipefail

WORKDIR="/common/home/users/m/myatmin.nay.2022/HalluRed"
cd "$WORKDIR"
mkdir -p logs data .cache/huggingface configs

# Load HF token if present
BASHRC_PATH=~/.bashrc
if [ -f "$BASHRC_PATH" ]; then
    HF_TOKEN_LINE=$(grep 'HUGGING_FACE_HUB_TOKEN' "$BASHRC_PATH" | tail -n 1 || true)
    if [ -n "${HF_TOKEN_LINE:-}" ]; then
        if [[ "$HF_TOKEN_LINE" == *"=\""* ]]; then
            HF_TOKEN=$(echo "$HF_TOKEN_LINE" | sed -E 's/.*="([^"]+)".*/\1/')
        elif [[ "$HF_TOKEN_LINE" == *"='"* ]]; then
            HF_TOKEN=$(echo "$HF_TOKEN_LINE" | sed -E "s/.*='([^']+)'.*/\1/")
        else
            HF_TOKEN=$(echo "$HF_TOKEN_LINE" | sed -E 's/.*=([^ ]+).*/\1/')
        fi
        if [ -n "${HF_TOKEN:-}" ]; then
            export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
            echo "Hugging Face token loaded from ~/.bashrc"
        fi
    fi
fi

export HF_HOME="$WORKDIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

module purge
module load Python/3.10.16-GCCcore-13.3.0
module load CUDA/12.6.0

if [ -d "$HOME/myenv" ]; then
    source "$HOME/myenv/bin/activate"
else
    echo "Python venv \"$HOME/myenv\" not found."
    exit 1
fi

CONFIG_PATH=${CONFIG_PATH:-configs/jolt_llama.yaml}

echo "[INFO] Starting JOLT training with $CONFIG_PATH"

srun -u python3 -u jolt_train.py --config "$CONFIG_PATH"


