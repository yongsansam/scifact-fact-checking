# Retriever reproduction

모든 Retriever는 최종적으로 다음 형식의 JSONL 파일을 생성한다.

```json
{"claim_id": 3, "doc_ids": [14717500, 23389795, 4632921]}
```

통합 실행 명령은 다음과 같다.

```bash
bash retriever/run_retriever.sh tfidf train 20
bash retriever/run_retriever.sh tfidf dev 20
bash retriever/run_retriever.sh bm25 train 20
bash retriever/run_retriever.sh bm25 dev 20
bash retriever/run_retriever.sh contriever train 20
bash retriever/run_retriever.sh contriever dev 20
```

## BM25 with Pyserini

`run_retriever.sh bm25`는 아래의 Lucene 인덱싱과 검색을 순서대로 실행한다.

```bash
# SciFact corpus를 Pyserini JsonCollection 형식으로 변환
python retriever/bm25/prepare_collection.py \
  --corpus data/scifact/corpus.jsonl \
  --output-dir indexes/bm25_collection

# Lucene BM25 index 생성
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input indexes/bm25_collection \
  --index indexes/bm25 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw

# Top-20 검색 및 Recall@K 계산
python retriever/bm25/run_bm25.py \
  --index indexes/bm25 \
  --corpus data/scifact/corpus.jsonl \
  --claims data/scifact/claims_dev.jsonl \
  --output-prefix outputs/retrieval/bm25/dev_top20 \
  --reader-topk 20 \
  --eval-ks 1,3,5,10,20
```

검색 결과는 `dev_top20.retrieval.jsonl`에 저장된다. 통합 스크립트는 이를
Reader가 사용하는 `dev_top20.jsonl` 이름으로 정리한다.

## Contriever with Pyserini

Contriever는 corpus와 claim을 변환한 뒤 FAISS index를 만들고 검색한다.

```bash
# corpus.jsonl과 queries.tsv 생성
python retriever/contriever/prepare_data.py \
  --corpus data/scifact/corpus.jsonl \
  --claims data/scifact/claims_dev.jsonl \
  --output-dir indexes/contriever_input/dev

# Contriever corpus embedding 및 FAISS index 생성
python -m pyserini.encode \
  --encoder-class contriever \
  --encoder models/contriever \
  --input indexes/contriever_input/dev/corpus.jsonl \
  --fields title text \
  --output indexes/contriever \
  --batch 32 \
  --fp16

# claim 검색 결과를 TREC 형식으로 생성
python -m pyserini.search.faiss \
  --encoder-class contriever \
  --encoder models/contriever \
  --index indexes/contriever \
  --topics indexes/contriever_input/dev/queries.tsv \
  --output outputs/retrieval/contriever/dev_top20.trec \
  --batch 128 \
  --threads 8 \
  --hits 20

# TREC 결과를 Reader 공통 JSONL 형식으로 변환
python retriever/contriever/trec_to_json.py \
  --run outputs/retrieval/contriever/dev_top20.trec \
  --output outputs/retrieval/contriever/dev_top20.jsonl \
  --topk 20
```

명령은 반드시 프로젝트 루트에서 실행한다. 다른 디렉터리에서 실행한다면
`--topics`를 포함한 입력 경로를 절대경로로 지정해야 한다.

## SciFact-finetuned DPR

DPR은 BM25가 생성한 hard negative를 이용해 SciFact에서 학습한다.

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
