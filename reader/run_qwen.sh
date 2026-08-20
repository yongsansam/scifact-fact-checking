#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETRIEVER="${1:-}"
K="${2:-3}"
[[ "$RETRIEVER" =~ ^(tfidf|bm25|contriever|dpr)$ ]] || {
  echo "Usage: $0 tfidf|bm25|contriever|dpr [K]" >&2; exit 2;
}
DATA="$ROOT/data/reader/$RETRIEVER/dev_top20_raw_abstract.json"
OUT="$ROOT/outputs/readers/$RETRIEVER/qwen"
mkdir -p "$OUT"
cd "$ROOT/reader/qwen"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python qwen_reader.py \
  --model "$ROOT/models/qwen2.5-7b-instruct" --data "$DATA" \
  --n-context "$K" --passage-length 341 \
  --output "$OUT/top${K}.tsv" --scores-output "$OUT/top${K}.scores.jsonl" \
  --metrics-output "$OUT/top${K}.metrics.json"

