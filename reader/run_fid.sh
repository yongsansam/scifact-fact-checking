#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_TYPE="${1:-}"
RETRIEVER="${2:-}"
[[ "$MODEL_TYPE" =~ ^(t5|bart)$ ]] || { echo "Usage: $0 t5|bart RETRIEVER" >&2; exit 2; }
[[ "$RETRIEVER" =~ ^(tfidf|bm25|contriever|dpr)$ ]] || { echo "Invalid retriever" >&2; exit 2; }

TRAIN="$ROOT/data/reader/$RETRIEVER/train_top20_raw_abstract.json"
DEV="$ROOT/data/reader/$RETRIEVER/dev_top20_raw_abstract.json"
OUT="$ROOT/checkpoints/fid_${MODEL_TYPE}_${RETRIEVER}"
mkdir -p "$OUT" "$ROOT/outputs/readers/$RETRIEVER/$MODEL_TYPE"

if [[ "$MODEL_TYPE" == t5 ]]; then
  cd "$ROOT/reader/fid_t5"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train_reader.py \
    --name "fid_t5_${RETRIEVER}" --checkpoint_dir "$ROOT/checkpoints" \
    --model_name "$ROOT/models/t5-base" --train_data "$TRAIN" --eval_data "$DEV" \
    --n_context 3 --text_maxlength 341 --answer_maxlength 8 \
    --per_gpu_batch_size 1 --accumulation_steps 8 --total_steps 1000 \
    --eval_freq 100 --save_freq 2000 --lr 1e-4 --optim adamw \
    --weight_decay 0.01 --scheduler fixed --balance_labels \
    --selection_metric macro_f1 --use_checkpoint --seed 0
else
  cd "$ROOT/reader"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m src.controlled_seq2seq train \
    --architecture fid --model "$ROOT/models/bart-base" \
    --train-data "$TRAIN" --dev-data "$DEV" --output-dir "$OUT" \
    --n-context 3 --passage-length 341 --micro-batch-size 1 \
    --accumulation-steps 8 --max-updates 1000 --eval-freq 100 \
    --learning-rate 1e-4 --weight-decay 0.01 --seed 0 --gradient-checkpointing
fi

