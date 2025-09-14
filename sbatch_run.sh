#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # Use 1 node
#SBATCH --cpus-per-task=10          # 10 CPUs for faster tokenization/IO
#SBATCH --mem=48GB                  # 48GB system memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --constraint=h100|a100|v100-32gb|v100           # Target GPUs specifically
#SBATCH --time=02-00:00:00          # Maximum run time of 2 days
##SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications for job start, end, and failure
#SBATCH --partition=researchshort   # Partition assigned
#SBATCH --account=sunjunresearch    # Account assigned (use myinfo to check)
#SBATCH --qos=research-1-qos        # QOS assigned (use myinfo to check)
#SBATCH --job-name=llm-check        # Job name
#SBATCH --output=logs/%x-%j.out     # Stdout log file
#SBATCH --error=logs/%x-%j.err      # Stderr log file

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

set -euo pipefail

# Resolve repository working directory
WORKDIR="/common/home/users/m/myatmin.nay.2022/HalluRed"
cd "$WORKDIR"

# Ensure log and data directories exist for SBATCH outputs and results
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

# Hugging Face/Transformers caches to fast local paths in the job workspace
export HF_HOME="$WORKDIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Default run configuration can be overridden via sbatch --export=ALL,VAR=val
# Canonical LLM-Check usage (includes hidden by default):
#   sbatch --export=ALL,MODEL=llama,DATASET=selfcheck,N_SAMPLES=0,MT_LIST="logit hidden attns",USE_TOKLENS=1 \
#          /common/home/users/m/myatmin.nay.2022/HalluRed/sbatch_run.sh
MODEL="${MODEL:-llama}"
DATASET="${DATASET:-selfcheck}"
N_SAMPLES="${N_SAMPLES:-200}"
USE_TOKLENS="${USE_TOKLENS:-0}"
# Space-separated list like: "logit hidden attns"; default includes 'hidden' for standard LLM-Check
MT_LIST="${MT_LIST:-logit hidden attns}"

# Build method args
MT_ARGS=""
for m in $MT_LIST; do
    MT_ARGS="$MT_ARGS --mt $m"
done

TOKLENS_ARG=""
if [ "$USE_TOKLENS" = "1" ]; then
    TOKLENS_ARG="--use_toklens"
fi

echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-} HOSTNAME=$(hostname)"
echo "[INFO] Running in: $WORKDIR"
echo "[INFO] Model: $MODEL | Dataset: $DATASET | N_SAMPLES: $N_SAMPLES | Methods: $MT_LIST | Toklens: $USE_TOKLENS"

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

# Launch the detection run on the allocated GPU
srun -u python3 -u run_detection_combined.py \
    --model "$MODEL" $MT_ARGS --n_samples "$N_SAMPLES" --dataset "$DATASET" $TOKLENS_ARG

