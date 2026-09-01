# Advanced Recommender Systems

A hands-on project focused on modern recommendation systems for implicit-feedback datasets.

The project is built around the RetailRocket dataset and gradually evolves from classical collaborative filtering methods to industrial-scale retrieval and ranking architectures used in modern recommendation systems.

---

## Project Structure

```
advanced-recommender-systems/
│
├── data/
│   ├── external/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── datasets/
│   ├── features/
│   ├── models/
│   ├── retrieval/
│   ├── ranking/
│   ├── evaluation/
│   └── utils/
│
├── tests/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

This project uses the **RetailRocket Recommender System Dataset**, which contains real user interactions collected from an e-commerce website.

The dataset includes:

- page views
- add-to-cart events
- transactions
- item metadata
- category hierarchy

Unlike MovieLens, RetailRocket provides **implicit user feedback**, making it suitable for building production-style recommender systems.

---

## Project Roadmap

### Phase 1 — Implicit Feedback

- Dataset exploration
- Interaction matrix
- Confidence matrix
- Implicit ALS
- Offline evaluation

### Phase 2 — Pairwise Learning

- Bayesian Personalized Ranking (BPR)
- Negative sampling
- Ranking loss

### Phase 3 — Candidate Generation

- User embeddings
- Item embeddings
- FAISS
- HNSW

### Phase 4 — Neural Retrieval

- Two-Tower Networks
- Contrastive Learning
- Hard Negative Mining

### Phase 5 — Ranking

- LightGBM Ranker
- Feature Engineering
- LambdaMART

### Phase 6 — Production Pipeline

- Retrieval + Ranking
- Inference pipeline
- ANN indexing
- End-to-end recommendation workflow

---

## Phase 1 — Implicit Feedback

The first phase focuses on building a classical collaborative filtering baseline for implicit-feedback recommendation.

### Interaction Processing

Raw user events are aggregated into user-item interactions.

Multiple interactions between the same user and item are combined into a single interaction with an aggregated confidence weight.

Global user and item indices are created to efficiently represent the interaction data as a sparse matrix.

### Implicit ALS

An **Alternating Least Squares (ALS)** model is used as the first collaborative filtering baseline.

The model learns latent representations for users and items based on implicit interactions rather than explicit ratings.

Previously observed items are excluded from the recommendation list during evaluation.

### Temporal Evaluation

The model is evaluated using a temporal train/test split.

The evaluation consists of **10 consecutive one-day test windows**:

- training data contains interactions observed before the test period
- test data contains interactions from the following day
- users and items not observed during training are excluded from evaluation

This setup better reflects the real recommendation scenario where the model predicts future user interactions.

### Evaluation Metrics

The recommendation quality is evaluated using:

- **Precision@10** — fraction of recommended items that are relevant
- **Recall@10** — fraction of relevant items that were successfully recommended
- **NDCG@10** — ranking quality with higher importance assigned to relevant items appearing near the top
- **Hit Rate@10** — fraction of users for whom at least one relevant item appears in the Top-10 recommendations

### Baseline Results

Average results across 10 temporal evaluation windows:

| Metric | Score |
|---|---:|
| Precision@10 | 0.49% |
| Recall@10 | 3.09% |
| NDCG@10 | 2.16% |
| Hit Rate@10 | 4.16% |

The results are treated as a **baseline** for subsequent experiments.

The relatively low ranking metrics indicate that the basic ALS model has limited ability to predict the items users will interact with during the following day. This provides a reference point for evaluating more advanced retrieval and ranking approaches in later phases.

---

## Setup

Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies

```bash
python -m pip install -r requirements.txt
```

Launch Jupyter

```bash
jupyter notebook
```

---

## Goal

The objective of this project is to reproduce the architecture of modern industrial recommender systems and gain practical experience with algorithms commonly used in real-world recommendation pipelines.

The project progressively moves from classical collaborative filtering to:

Collaborative Filtering → Pairwise Learning → Candidate Generation → Neural Retrieval → Ranking → Production Pipeline