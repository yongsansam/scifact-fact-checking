#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TYPE="${1:-}"
SPLIT="${2:-dev}"
K="${3:-20}"
[[ "$TYPE" =~ ^(tfidf|bm25|contriever|dpr)$ ]] || { echo "Usage: $0 RETRIEVER [train|dev] [K]" >&2; exit 2; }
[[ "$SPLIT" =~ ^(train|dev)$ ]] || { echo "Split must be train or dev" >&2; exit 2; }
CORPUS="$ROOT/data/scifact/corpus.jsonl"
CLAIMS="$ROOT/data/scifact/claims_${SPLIT}.jsonl"
OUTDIR="$ROOT/outputs/retrieval/$TYPE"
OUT="$OUTDIR/${SPLIT}_top${K}.jsonl"
mkdir -p "$OUTDIR" "$ROOT/indexes"

case "$TYPE" in
  tfidf)
    python "$ROOT/retriever/tfidf/retrieve.py" --corpus "$CORPUS" \
      --dataset "$CLAIMS" --k "$K" --min-gram 1 --max-gram 2 --output "$OUT" ;;
  bm25)
    COLLECTION="$ROOT/indexes/bm25_collection"
    INDEX="$ROOT/indexes/bm25"
    if [[ ! -d "$INDEX" ]]; then
      python "$ROOT/retriever/bm25/prepare_collection.py" --corpus "$CORPUS" --output-dir "$COLLECTION"
      python -m pyserini.index.lucene --collection JsonCollection \
        --input "$COLLECTION" --index "$INDEX" --generator DefaultLuceneDocumentGenerator \
        --threads 8 --storePositions --storeDocvectors --storeRaw
    fi
    python "$ROOT/retriever/bm25/run_bm25.py" --index "$INDEX" --corpus "$CORPUS" \
      --claims "$CLAIMS" --output-prefix "$OUTDIR/${SPLIT}_top${K}" \
      --reader-topk "$K" --eval-ks "1,3,5,10,${K}"
    mv "$OUTDIR/${SPLIT}_top${K}.retrieval.jsonl" "$OUT" ;;
  contriever)
    python "$ROOT/retriever/dense_retrieve.py" --type contriever --corpus "$CORPUS" \
      --claims "$CLAIMS" --output "$OUT" --topk "$K" --model "$ROOT/models/contriever" ;;
  dpr)
    CKPT="$ROOT/checkpoints/dpr/best"
    test -d "$CKPT/question_encoder" || { echo "Train DPR first; see retriever/README.md" >&2; exit 1; }
    python "$ROOT/retriever/dense_retrieve.py" --type dpr --corpus "$CORPUS" \
      --claims "$CLAIMS" --output "$OUT" --topk "$K" \
      --question-model "$CKPT/question_encoder" --context-model "$CKPT/ctx_encoder" ;;
esac
echo "Wrote $OUT"

