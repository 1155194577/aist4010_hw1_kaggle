#!/bin/bash
#SBATCH --job-name=aist4010-sweep
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    echo "Create .venv or submit with PYTHON_BIN=/path/to/python" >&2
    exit 1
fi

if [[ ! -d data/train || ! -d data/val || ! -d data/test ]]; then
    if [[ "${DOWNLOAD_DATA:-0}" == "1" ]]; then
        "$PYTHON_BIN" download_data.py
    else
        echo "Training data not found under $PWD/data" >&2
        echo "Run '$PYTHON_BIN download_data.py' before sbatch, or submit with DOWNLOAD_DATA=1" >&2
        exit 1
    fi
fi

mkdir -p checkpoints

export WANDB_PROJECT="${WANDB_PROJECT:-aist4010-hw1}"
export PYTHONUNBUFFERED=1

printf 'Job ID: %s\n' "${SLURM_JOB_ID:-local}"
printf 'Host: %s\n' "$(hostname)"
printf 'Working directory: %s\n' "$PWD"
printf 'Python: %s\n' "$PYTHON_BIN"
printf 'CUDA devices: %s\n' "${CUDA_VISIBLE_DEVICES:-all}"

srun "$PYTHON_BIN" sweep.py
