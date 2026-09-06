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

The project follows a gradual progression from classical recommendation models to a modern industrial recommendation pipeline.

### Phase 1 — Exploratory Analysis

- Dataset exploration
- Interaction analysis
- User and item behavior
- Temporal patterns
- Sparsity analysis

### Phase 2 — Recommendation Baseline

- Additive baseline
- User and item biases
- Popularity-based ranking
- Offline evaluation

### Phase 3 — Latent Factor Models

- Implicit feedback
- Confidence matrix
- Alternating Least Squares (ALS)
- Latent user and item representations
- Offline evaluation

### Phase 4 — Item-Based Collaborative Filtering

- Item-item similarity
- Similarity-based recommendation
- User history aggregation
- Offline evaluation

### Phase 5 — Pairwise Ranking

- Bayesian Personalized Ranking (BPR)
- Negative sampling
- Pairwise ranking loss

### Phase 6 — Retrieval

- Candidate generation
- User embeddings
- Item embeddings
- Approximate nearest-neighbor search
- FAISS / HNSW

### Phase 7 — Neural Retrieval

- Two-Tower Networks
- Contrastive Learning
- In-batch negatives
- Hard Negative Mining

### Phase 8 — Ranking

- Feature Engineering
- LightGBM Ranker
- LambdaMART
- Ranking-specific objectives

### Phase 9 — Industrial Recommendation Pipeline

- Retrieval + Ranking
- ANN indexing
- Inference pipeline
- Scalability and latency
- End-to-end recommendation workflow

### Notebooks

| Notebook | Topic |
|---|---|
| `01_eda.ipynb` | Exploratory Data Analysis |
| `02_baseline.ipynb` | Recommendation Baseline |
| `03_als.ipynb` | Implicit ALS |
| `04_item_based.ipynb` | Item-Based Collaborative Filtering |
| `05_pairwise_ranking.ipynb` | Bayesian Personalized Ranking |
| `06_retrieval.ipynb` | Candidate Retrieval |
| `07_neural_retrieval.ipynb` | Neural Retrieval |
| `08_ranking.ipynb` | Learning to Rank |
| `09_industrial_pipeline.ipynb` | Industrial Recommendation Pipeline |

---

## Phase 1 — Exploratory Analysis

The first phase focuses on understanding the RetailRocket dataset and its main characteristics before building recommendation models.

### Dataset Exploration

The analysis covers:

- user and item activity
- interaction types
- interaction distribution
- item popularity
- long-tail behavior
- temporal patterns
- sparsity of the user-item interaction matrix

The goal is to understand the data and identify the main challenges of the recommendation task.

---

## Phase 2 — Recommendation Baseline

The second phase introduces a simple baseline to establish a reference point for subsequent recommendation models.

### Baseline Model

An additive model is used:

$$
\hat{r}_{ui} = \mu + b_u + b_i
$$

where:

- $\mu$ is the global interaction strength
- $b_u$ is the user bias
- $b_i$ is the item bias

For Top-K recommendation, the global mean and user bias are constant for a given user and therefore do not affect the ranking of candidate items.

As a result, the baseline effectively behaves like a popularity-based recommendation model.

### Baseline Results

Average results across 10 temporal evaluation windows:

| Metric | Score |
|---|---:|
| Precision@10 | ~0.10% |
| Recall@10 | ~0.55% |
| NDCG@10 | ~0.36% |

The baseline provides a simple reference point for evaluating more advanced collaborative filtering models.

The relatively low performance demonstrates the limitation of popularity-based recommendation and motivates the use of personalized collaborative filtering.

---

## Phase 3 — Implicit ALS

The third phase introduces **Alternating Least Squares (ALS)** for implicit-feedback recommendation.

### Interaction Processing

Raw user events are aggregated into user-item interactions.

Multiple interactions between the same user and item are combined into a single interaction with an aggregated weight.

Global user and item indices are created to efficiently represent the interaction data as a sparse matrix.

### Implicit ALS

The **Alternating Least Squares (ALS)** model learns latent representations for users and items from implicit interactions rather than explicit ratings.

The model uses:

- binary preferences derived from observed interactions
- confidence values based on interaction strength
- sparse matrix operations
- latent user and item representations

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

### ALS Results

Average results across 10 temporal evaluation windows:

| Metric | Score |
|---|---:|
| Precision@10 | 0.49% |
| Recall@10 | 3.09% |
| NDCG@10 | 2.16% |
| Hit Rate@10 | 4.16% |

ALS substantially outperforms the simple baseline on all reported ranking metrics.

This demonstrates the benefit of modeling user-item interactions with latent representations rather than relying primarily on item popularity.

---

## Phase 4 — Item-Based Collaborative Filtering

The fourth phase explores **Item-Based Collaborative Filtering**.

Instead of learning latent representations for users and items, the model uses item-to-item similarity derived from user interaction patterns.

The recommendation process consists of:

1. identifying items the user has interacted with
2. finding similar items
3. aggregating item similarities to produce personalized recommendations

The same temporal evaluation procedure and Top-K metrics are used to make the results directly comparable with the previous models.

---

## Phase 5 — Pairwise Ranking

The fifth phase introduces **Bayesian Personalized Ranking (BPR)** and pairwise learning.

The goal is to optimize the relative ordering of items rather than reconstructing the interaction matrix.

Topics covered:

- positive and negative interactions
- negative sampling
- pairwise ranking loss
- personalized ranking

---

## Phase 6 — Retrieval

The sixth phase focuses on **candidate generation**.

The goal is to efficiently retrieve a relatively small set of potentially relevant items from a large item catalog.

Topics covered:

- user embeddings
- item embeddings
- approximate nearest-neighbor search
- FAISS
- HNSW

---

## Phase 7 — Neural Retrieval

The seventh phase extends candidate generation with neural recommendation models.

Topics covered:

- Two-Tower Networks
- contrastive learning
- in-batch negatives
- hard negative mining

The goal is to learn user and item representations optimized for efficient retrieval.

---

## Phase 8 — Ranking

The eighth phase focuses on re-ranking retrieved candidates using richer features.

Topics covered:

- feature engineering
- LightGBM Ranker
- LambdaMART
- ranking-specific objectives

The goal is to improve the ordering of a relatively small candidate set.

---

## Phase 9 — Industrial Recommendation Pipeline

The final phase combines the previous components into an end-to-end recommendation architecture.

The high-level pipeline is:

```text
User / Context
      ↓
Candidate Retrieval
      ↓
Ranking
      ↓
Top-K Recommendations

---

## Evaluation

Whenever applicable, all recommendation models use the same temporal evaluation framework to make experiments directly comparable.

The main offline metrics are:

- **Precision@10**
- **Recall@10**
- **NDCG@10**
- **Hit Rate@10**

The objective is not to predict an exact rating, but to rank items that the user is likely to interact with higher than other candidate items.

The evaluation uses future interactions from the test period as relevant items, while items already observed during training are excluded from the recommendation candidates.

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

The project progressively moves from:

Exploratory Analysis → Baseline → ALS → Item-Based Collaborative Filtering → Pairwise Ranking → Retrieval → Neural Retrieval → Ranking → Industrial Recommendation Pipeline