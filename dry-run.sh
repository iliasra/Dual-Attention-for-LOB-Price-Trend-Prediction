#!/bin/bash
#PBS -N dry-run
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1:gpu_type=A100

set -euo pipefail

PROJECT_DIR="$HOME/Dual-Attention-for-LOB-Price-Trend-Prediction"
WORK_DIR="$TMPDIR/Dual-Attention-for-LOB-Price-Trend-Prediction"
RESULT_FILE="dry_run_results.txt"

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate thesis

cp -r "$PROJECT_DIR" "$TMPDIR/"
cd "$WORK_DIR"

export PYTHONPATH="$WORK_DIR/src:${PYTHONPATH:-}"
python scripts/vram_dry_run.py \
  --batch-size 32 \
  --sequence-length 100 \
  --num-features 240 \
  --steps 5 \
> "$TMPDIR/$RESULT_FILE" 2>&1

cp "$TMPDIR/$RESULT_FILE" "$HOME/$RESULT_FILE"

