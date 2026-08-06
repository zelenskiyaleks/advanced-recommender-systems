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

The objective of this project is to reproduce the architecture of modern industrial recommender systems and gain practical experience with algorithms commonly used in companies.# advanced-recommender-systems