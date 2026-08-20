# Retriever reproduction

Retrievers `{"claim_id": ..., "doc_ids": [...]}` JSONL files 작성
```bash
bash retriever/run_retriever.sh tfidf train 20
bash retriever/run_retriever.sh tfidf dev 20
bash retriever/run_retriever.sh bm25 train 20
bash retriever/run_retriever.sh bm25 dev 20
bash retriever/run_retriever.sh contriever train 20
bash retriever/run_retriever.sh contriever dev 20
```

Train DPR :

```bash
python retriever/dpr/train_dpr_scifact.py \
  --corpus data/scifact/corpus.jsonl \
  --train-claims data/scifact/claims_train.jsonl \
  --dev-claims data/scifact/claims_dev.jsonl \
  --bm25-index indexes/bm25 \
  --question-model models/dpr-question-encoder \
  --context-model models/dpr-context-encoder \
  --output-dir checkpoints/dpr --epochs 2 --fp16

bash retriever/run_retriever.sh dpr train 20
bash retriever/run_retriever.sh dpr dev 20
```

