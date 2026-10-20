#!/usr/bin/env bash
#
# sweep.sh — run the controlled experiments (and any future ones) from one place.
#
#   ./sweep.sh                 # run every experiment sequentially (AutoDL, gpu_24h)
#   ./sweep.sh r1 r3           # only the runs whose name starts with r1 / r3
#   ./sweep.sh --list          # show the experiment list
#   ./sweep.sh --dry-run       # print the commands, run nothing
#   ./sweep.sh --slurm         # submit one Slurm job per experiment (cluster)
#   ./sweep.sh --shutdown      # power off when the last local run finishes (AutoDL)
#
# Every run writes:
#   checkpoints/sweep/<name>.pt        best checkpoint (selected on val accuracy)
#   checkpoints/sweep/<name>_last.pt   resumable checkpoint (written every epoch)
#   logs/<name>.log                    full stdout, appended across attempts
#   logs/summary.tsv                   one line per run of the last invocation
#
# Re-running the same command resumes an interrupted run (`--resume last` is a
# no-op when there is no checkpoint yet). Delete the *_last.pt to start over.
#
# W&B: set WANDB_MODE=offline on machines without `wandb login`, then
# `wandb sync wandb/offline-run-*` later.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# experiments: name | train.py arguments
# Keep one variable per run so the comparison stays controlled.
# ---------------------------------------------------------------------------
RUNS=(
  "r1_ce_repro      | --head linear  --no-bnneck --dropout 0   --random-erasing 0    --ema-decay 0     --epochs 64"
  "r2_ce_100        | --head linear  --no-bnneck --dropout 0   --random-erasing 0    --ema-decay 0     --epochs 100"
  "r3_ce_reg        | --head linear  --no-bnneck --dropout 0.2 --random-erasing 0.25 --ema-decay 0.999 --epochs 100"
  "r4_arcface_iso   | --head arcface --no-bnneck --dropout 0   --random-erasing 0    --ema-decay 0     --epochs 64"
  "r5_ce_r34        | --head linear  --no-bnneck --dropout 0   --random-erasing 0    --ema-decay 0     --epochs 100 --backbone resnet34"
)

SEED="${SEED:-42}"
CKPT_DIR="${CKPT_DIR:-checkpoints/sweep}"
LOG_DIR="${LOG_DIR:-logs}"
SUMMARY="$LOG_DIR/summary.tsv"

MODE=local
DRY_RUN=0
SHUTDOWN=0
LIST_ONLY=0
FILTERS=()

usage() {
    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm)     MODE=slurm; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --shutdown)  SHUTDOWN=1; shift ;;
        --list)      LIST_ONLY=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        -*)          echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)           FILTERS+=("$1"); shift ;;
    esac
done

if [[ "$LIST_ONLY" == "1" ]]; then
    for entry in "${RUNS[@]}"; do
        name="${entry%%|*}"; name="${name// /}"
        printf '%-18s %s\n' "$name" "${entry#*|}"
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
if [[ "$MODE" == "local" ]]; then
    if [[ -z "${PYTHON_BIN:-}" ]]; then
        if [[ -x .venv/bin/python ]]; then
            PYTHON_BIN=".venv/bin/python"
        elif [[ -x venv/bin/python ]]; then
            PYTHON_BIN="venv/bin/python"
        else
            PYTHON_BIN="$(command -v python3 || true)"
        fi
    fi
    [[ -z "$PYTHON_BIN" ]] && PYTHON_BIN="python3"
    if [[ "$DRY_RUN" != "1" ]]; then
        if [[ ! -x "$PYTHON_BIN" ]]; then
            echo "ERROR: no Python interpreter. Set PYTHON_BIN=/path/to/python." >&2
            exit 1
        fi
        if ! "$PYTHON_BIN" -c 'import torch' 2>/dev/null; then
            echo "ERROR: $PYTHON_BIN cannot import torch." >&2
            exit 1
        fi
        if [[ "${SKIP_DATA_CHECK:-0}" != "1" ]]; then
            for split in train val test; do
                if [[ ! -d "data/$split" ]]; then
                    echo "ERROR: data/$split is missing (run from the repo, or set SKIP_DATA_CHECK=1)." >&2
                    exit 1
                fi
            done
        fi
    fi
fi

mkdir -p "$CKPT_DIR" "$LOG_DIR"
: > "$SUMMARY"

selected() {
    local name="$1" filter
    [[ ${#FILTERS[@]} -eq 0 ]] && return 0
    for filter in "${FILTERS[@]}"; do
        [[ "$name" == "$filter"* ]] && return 0
    done
    return 1
}

best_epoch() {
    # best epoch = last "Checkpoint saved" line, since it is only written on improvement
    grep -oE 'Checkpoint saved at epoch [0-9]+' "$1" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || true
}

best_accuracy() {
    local log="$1"
    if grep -q 'Validation Accuracy: ' "$log" 2>/dev/null; then
        grep -oE 'Validation Accuracy: [0-9]+\.[0-9]+' "$log" \
            | awk '{print $3}' | sort -g | tail -1
    else
        echo "NA"
    fi
}

run_local() {
    local name="$1"; shift
    local args=("$@")
    local ckpt="$CKPT_DIR/$name.pt"
    local log="$LOG_DIR/$name.log"
    local cmd=("$PYTHON_BIN" train.py "${args[@]}" --seed "$SEED" \
               --checkpoint "$ckpt" --resume last)

    {
        echo
        echo "### $(date '+%F %T') $name"
        printf '### cmd:'
        printf ' %q' "${cmd[@]}"
        echo
    } | tee -a "$log"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY-RUN %s: ' "$name"
        printf '%q ' "${cmd[@]}"
        echo
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$SEED" "-" "-" "-" "dry-run" >> "$SUMMARY"
        return 0
    fi

    local start=$SECONDS status
    "${cmd[@]}" 2>&1 | tee -a "$log"
    status=$?
    local elapsed=$((SECONDS - start))

    local acc epoch
    acc="$(best_accuracy "$log")"
    epoch="$(best_epoch "$log")"
    [[ -z "$epoch" ]] && epoch="-"
    printf '%s\t%s\t%s\t%s\t%ss\t%s\n' \
        "$name" "$SEED" "$acc" "${epoch:-"-"}" "$elapsed" "$status" >> "$SUMMARY"
    echo "### $name done in ${elapsed}s, best val acc ${acc}, exit ${status}"
}

run_slurm() {
    local name="$1"; shift
    local args=("$@")
    local ckpt="$CKPT_DIR/$name.pt"
    local cmd=(sbatch --job-name="$name" submit.slurm train.py "${args[@]}" \
               --seed "$SEED" --checkpoint "$ckpt" --resume last)

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY-RUN %s: ' "$name"
        printf '%q ' "${cmd[@]}"
        echo
        return 0
    fi
    "${cmd[@]}"
}

for entry in "${RUNS[@]}"; do
    name="${entry%%|*}"
    name="${name// /}"
    args="${entry#*|}"
    selected "$name" || continue
    read -r -a ARGS <<< "$args"

    if [[ "$MODE" == "slurm" ]]; then
        run_slurm "$name" "${ARGS[@]}"
    else
        run_local "$name" "${ARGS[@]}"
    fi
done

if [[ "$MODE" == "local" && "$DRY_RUN" != "1" ]]; then
    echo
    printf '%-18s %-6s %-10s %-8s %-8s\n' "RUN" "SEED" "BEST_ACC" "EPOCH" "TIME"
    awk -F'\t' '{printf "%-18s %-6s %-10s %-8s %-8s\n", $1, $2, $3, $4, $5}' "$SUMMARY"
    echo
    echo "summary: $SUMMARY"
fi

if [[ "$SHUTDOWN" == "1" && "$MODE" == "local" && "$DRY_RUN" != "1" ]]; then
    if [[ -x /usr/bin/shutdown ]]; then
        echo "shutting down"
        /usr/bin/shutdown
    else
        echo "--shutdown requested but /usr/bin/shutdown is missing" >&2
    fi
fi
