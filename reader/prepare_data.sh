#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETRIEVER="${1:-}"
[[ "$RETRIEVER" =~ ^(tfidf|bm25|contriever|dpr)$ ]] || {
  echo "Usage: $0 tfidf|bm25|contriever|dpr [top-k]" >&2; exit 2;
}
K="${2:-20}"
for SPLIT in train dev; do
  python "$ROOT/reader/data/prepare_reader_data.py" \
    --claims "$ROOT/data/scifact/claims_${SPLIT}.jsonl" \
    --corpus "$ROOT/data/scifact/corpus.jsonl" \
    --retrieval "$ROOT/outputs/retrieval/$RETRIEVER/${SPLIT}_top${K}.jsonl" \
    --output "$ROOT/data/reader/$RETRIEVER/${SPLIT}_top${K}_raw_abstract.json" \
    --n-context "$K"
done

