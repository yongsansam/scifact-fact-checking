#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-all}"

download_data() {
  mkdir -p "$ROOT/data/scifact"
  local base="https://raw.githubusercontent.com/allenai/scifact/master/data"
  for file in corpus.jsonl claims_train.jsonl claims_dev.jsonl claims_test.jsonl; do
    curl -fL --retry 3 "$base/$file" -o "$ROOT/data/scifact/$file"
  done
}

download_models() {
  command -v huggingface-cli >/dev/null || {
    echo "huggingface-cli is required: pip install huggingface_hub" >&2; exit 1;
  }
  mkdir -p "$ROOT/models"
  huggingface-cli download t5-base --local-dir "$ROOT/models/t5-base"
  huggingface-cli download facebook/bart-base --local-dir "$ROOT/models/bart-base"
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir "$ROOT/models/qwen2.5-7b-instruct"
  huggingface-cli download facebook/contriever --local-dir "$ROOT/models/contriever"
  huggingface-cli download facebook/dpr-question_encoder-single-nq-base \
    --local-dir "$ROOT/models/dpr-question-encoder"
  huggingface-cli download facebook/dpr-ctx_encoder-single-nq-base \
    --local-dir "$ROOT/models/dpr-context-encoder"
}

case "$TARGET" in
  data) download_data ;;
  models) download_models ;;
  all) download_data; download_models ;;
  *) echo "Usage: $0 [data|models|all]" >&2; exit 2 ;;
esac

