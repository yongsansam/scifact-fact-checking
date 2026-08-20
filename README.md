# SciFact Multi-Document Fact Checking

SciFact claim에 대해 네 Retriever가 검색한 abstract를 Reader가 통합해
`SUPPORTS`, `REFUTES`, `NOINFO` 중 하나를 예측하는 재현 코드다.

Reader 구현은 Facebook Research의 Fusion-in-Decoder 구조와 AllenAI의
SciFact 데이터 형식을 기반으로 프로젝트의 claim-level 3-way 설정에 맞게 구성했다.

## 범위

- Retriever: TF-IDF, BM25, Contriever, SciFact-finetuned DPR
- Reader: FiD-T5, FiD-BART, Qwen2.5-7B-Instruct
- Reader 학습: Retriever별 Top-3 abstracts
- 평가: claim-level Accuracy와 Macro-F1

Natural Questions 재현 코드, binary evidence-only 실험, VeriSci ablation,
체크포인트와 생성 결과는 포함하지 않는다.

## 설치 및 다운로드

FiD와 Qwen은 의존성 충돌을 피하기 위해 별도 환경 사용을 권장한다.

```bash
conda create -n scifact-fid python=3.8 -y
conda activate scifact-fid
pip install -r requirements-fid.txt

conda create -n scifact-qwen python=3.10 -y
conda activate scifact-qwen
pip install -r requirements-qwen.txt
```

데이터와 모델은 Git으로 추적하지 않는다.

```bash
bash download.sh all
# 또는 bash download.sh data / bash download.sh models
```

## 디렉터리

```text
retriever/          # TF-IDF, BM25, Contriever, DPR
reader/data/        # retrieval 결과를 FiD/Qwen JSON으로 변환
reader/fid_t5/      # FiD-T5
reader/src/         # FiD-BART 공통 구현
reader/qwen/        # Qwen zero-shot reader
data/               # download.sh 생성, Git 제외
models/             # download.sh 생성, Git 제외
checkpoints/        # 학습 시 생성, Git 제외
outputs/            # 검색 및 평가 결과, Git 제외
```

## Reader 데이터 생성

Retriever 결과는 다음 JSONL 형식을 사용한다.

```json
{"claim_id": 3, "doc_ids": [14717500, 23389795, 4632921]}
```

Top-20 검색 결과를 Reader 형식으로 변환하면 K=1, 3, 5 등으로 잘라 평가할 수 있다.

먼저 검색 결과를 생성한다. DPR 학습 명령은 `retriever/README.md`에 있다.

```bash
for retriever in tfidf bm25 contriever; do
  bash retriever/run_retriever.sh "$retriever" train 20
  bash retriever/run_retriever.sh "$retriever" dev 20
done
```

```bash
bash reader/prepare_data.sh bm25 20
bash reader/prepare_data.sh tfidf 20
bash reader/prepare_data.sh contriever 20
bash reader/prepare_data.sh dpr 20
```

## Reader 실행

```bash
# Retriever별 Top-3로 fine-tuning
bash reader/run_fid.sh t5 bm25
bash reader/run_fid.sh bart bm25

# 같은 체크포인트로 K=1,3,5 평가
for k in 1 3 5; do
  bash reader/evaluate_fid.sh t5 bm25 "$k"
  bash reader/evaluate_fid.sh bart bm25 "$k"
done

# Qwen zero-shot, Top-3
conda activate scifact-qwen
bash reader/run_qwen.sh bm25 3
```

다른 Retriever는 `bm25`를 `tfidf`, `contriever`, `dpr`로 교체한다.

## 주의

DPR의 기반 encoder 이름에는 원 사전학습 데이터인 `nq`가 포함되지만,
이 저장소는 NQ benchmark를 재현하지 않는다. DPR은 SciFact evidence pair와
BM25 hard negative로 추가 학습한 뒤 SciFact 검색에만 사용한다.
