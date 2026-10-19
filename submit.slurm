#!/bin/bash
#SBATCH --job-name=aist4010
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=07:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_8h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --constraint=highcpucount
#
# CUHK CSE GPGPU cluster (submit node linux5-16).
#   export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
#   sbatch submit.slurm                          # train.py, Config defaults
#   sbatch submit.slurm sweep.py                 # sweep.py
#   sbatch submit.slurm train.py --seed 43       # flags are passed to the script
#   sbatch submit.slurm train.py --backbone resnet34 --checkpoint checkpoints/r34.pt
#
# 10 CPUs + 1 GPU is a 10:1 ratio, which requires the "highcpucount" feature
# (gpu24-53) per the cluster docs; config.num_workers=8 then fits.
# For a short job on the higher-priority default queue:
#   sbatch -p gpu_2h --time=01:50:00 --cpus-per-task=6 submit.slurm
# (6:1 needs no feature; set Config.num_workers to 6 in that case.)
#
# Titan RTX nodes (gpu54-59) use a separate queue:
#   sbatch -p ex_batch --qos=ex_gpu --gres=gpu:1 submit.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export SLURM_CONF="${SLURM_CONF:-/opt1/slurm/gpu-slurm.conf}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-aist4010-hw1}"

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

TARGET="${1:-train.py}"
shift || true

printf 'Job ID: %s\n' "${SLURM_JOB_ID:-local}"
printf 'Host: %s\n' "$(hostname)"
printf 'Partition: %s\n' "${SLURM_JOB_PARTITION:-unknown}"
printf 'Working directory: %s\n' "$PWD"
printf 'Python: %s\n' "$PYTHON_BIN"
printf 'Target: %s %s\n' "$TARGET" "$*"
printf 'CUDA devices: %s\n' "${CUDA_VISIBLE_DEVICES:-all}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

srun "$PYTHON_BIN" "$TARGET" "$@"
