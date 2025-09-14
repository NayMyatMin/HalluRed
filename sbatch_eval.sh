#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#|a100|v100-32gb|v100
#SBATCH --time=02-00:00:00
#SBATCH --partition=researchshort
#SBATCH --account=sunjunresearch
#SBATCH --qos=research-1-qos
#SBATCH --job-name=llm-check
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

set -euo pipefail

WORKDIR="/common/home/users/m/myatmin.nay.2022/HalluRed"
cd "$WORKDIR"
mkdir -p logs data ../llm_store/datasets

# Extract Hugging Face token from ~/.bashrc if set there
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

# Optional: respect pre-exported token env vars if user set them differently
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

# Module environment (EasyBuild)
module purge
module load Python/3.10.16-GCCcore-13.3.0
module load CUDA/12.6.0

# Activate user Python environment (expected pre-created)
if [ -d "$HOME/myenv" ]; then
    source "$HOME/myenv/bin/activate"
else
    echo "Python venv \"$HOME/myenv\" not found. Create it and install requirements before submitting."
    exit 1
fi

export PYTHONUNBUFFERED=1

# Hugging Face/Transformers caches
export HF_HOME="$WORKDIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Defaults (same as sbatch_run; add adapter support)
MODEL="${MODEL:-llama}"
DATASET="${DATASET:-fava_annot}"
N_SAMPLES="${N_SAMPLES:-200}"
USE_TOKLENS="${USE_TOKLENS:-1}"
MT_LIST="${MT_LIST:-logit hidden attns}"
ADAPTER_DIR="${ADAPTER_DIR:-}"

# Build method args
MT_ARGS=""
for m in $MT_LIST; do
    MT_ARGS="$MT_ARGS --mt $m"
done

TOKLENS_ARG=""
if [ "$USE_TOKLENS" = "1" ]; then
    TOKLENS_ARG="--use_toklens"
fi

ADAPTER_ARG=""
if [ -n "$ADAPTER_DIR" ]; then
    ADAPTER_ARG="--adapter_dir $ADAPTER_DIR"
fi

echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-} HOSTNAME=$(hostname)"
echo "[INFO] Running in: $WORKDIR"
echo "[INFO] Model: $MODEL | Dataset: $DATASET | N_SAMPLES: $N_SAMPLES | Methods: $MT_LIST | Toklens: $USE_TOKLENS | Adapter: ${ADAPTER_DIR:-none}"

# Quick dependency check to fail fast with a clear message
python - <<'PY'
import importlib, sys
required = [
    ("torch", "pip install torch --extra-index-url https://download.pytorch.org/whl/cu121"),
    ("transformers", "pip install transformers==4.40.1"),
    ("fastchat", "pip install fschat==0.2.36"),
    ("datasets", "pip install datasets==2.14.5"),
    ("sentencepiece", "pip install sentencepiece"),
    ("tqdm", "pip install tqdm"),
    ("sklearn", "pip install scikit-learn"),
    ("bs4", "pip install beautifulsoup4"),
    ("jsonlines", "pip install jsonlines"),
    ("peft", "pip install peft"),
]
missing = []
for mod, hint in required:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append((mod, hint))
if missing:
    print("Missing Python packages:")
    for mod, hint in missing:
        print(f" - {mod}: try `{hint}`")
    sys.exit(2)
PY

# Launch the detection run on the allocated GPU (adds adapter if provided)
srun -u python3 -u run_detection_combined.py \
    --model "$MODEL" $MT_ARGS --n_samples "$N_SAMPLES" --dataset "$DATASET" $TOKLENS_ARG $ADAPTER_ARG


