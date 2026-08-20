#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TYPE="${1:-}"; RETRIEVER="${2:-}"; K="${3:-3}"
[[ "$TYPE" =~ ^(t5|bart)$ ]] || { echo "Usage: $0 t5|bart RETRIEVER [K]" >&2; exit 2; }
DATA="$ROOT/data/reader/$RETRIEVER/dev_top20_raw_abstract.json"
OUT="$ROOT/outputs/readers/$RETRIEVER/$TYPE"; mkdir -p "$OUT"

if [[ "$TYPE" == t5 ]]; then
  CHECKPOINT="$ROOT/checkpoints/fid_t5_${RETRIEVER}/checkpoint/best_dev"
  cd "$ROOT/reader/fid_t5"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python test_scifact_fid_claim_label.py \
    --eval-data "$DATA" --model-path "$CHECKPOINT" --tokenizer-name "$ROOT/models/t5-base" \
    --n-context "$K" --text-maxlength 341 --batch-size 1 --device cuda \
    --local-files-only --score-reduction sum --output "$OUT/top${K}.tsv" \
    --scores-output "$OUT/top${K}.scores.jsonl"
  python evaluate_scifact_fid_claim_label.py \
    --claims "$ROOT/data/scifact/claims_dev.jsonl" --predictions "$OUT/top${K}.tsv" \
    --metrics-output "$OUT/top${K}.metrics.json" --n-context "$K"
else
  CHECKPOINT="$ROOT/checkpoints/fid_bart_${RETRIEVER}/best"
  cd "$ROOT/reader"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m src.controlled_seq2seq evaluate \
    --architecture fid --model "$CHECKPOINT" --data "$DATA" --n-context "$K" \
    --passage-length 341 --batch-size 1 --output "$OUT/top${K}.tsv" \
    --scores-output "$OUT/top${K}.scores.jsonl" --metrics-output "$OUT/top${K}.metrics.json"
fi

