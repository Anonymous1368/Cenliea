# Cenliea

**Cenliea: A Cross-Encoder NLI Framework Enhanced by LLM-Based Reasoning for Transferable, Domain-Agnostic Entity Alignment**

This repository provides the complete codebase for **Cenliea** and **Cenliea+**, two complementary pipelines for entity alignment (EA) across heterogeneous knowledge graphs (KGs). The framework combines structured input encoding, multilingual NLI-based embeddings, and LLM-augmented reasoning to offer a generalizable, explainable alignment approach.

📄 For technical details, see our **anonymous paper** on OpenReview:  
🔗 https://openreview.net/forum?id=v4Fnw1oySH
---

## 📌 Overview

Cenliea introduces a neurosymbolic EA pipeline with the following core phases:
- 🧱 **Structured prompt construction** from RDF graphs
- 🧠 **NLI vectorization** for entity pair similarity estimation
- 📈 **Binary classifier training** for alignment detection
- 🔍 **Cenliea+ (optional)**: LLM-generated explanations augment difficult or low-confidence cases

---

## 🧱 Components

### [`dataset_preparation/`](./dataset_preparation)

- Converts RDF/XML files into structured JSON
- Extracts direct and one-hop features
- Computes GPT-token similarity for negative sampling
- Produces `.parquet` files for downstream NLI encoding

📖 See [`dataset_preparation/README.md`](./dataset_preparation/README.md)

---

### [`nli_vectorization/`](./nli_vectorization)

- Phase 1 of Cenliea: Bidirectional NLI embeddings using [`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
- Loads `.parquet` input datasets and outputs embeddings per pair
- Outputs are cached for reuse or fallback scenarios in Cenliea+

📖 See [`nli_vectorization/README.md`](./nli_vectorization/README.md)

---

### [`Cenliea_plus/`](./Cenliea_plus)

- Optional Phase 2: LLM-based alignment justification using [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- Generates premise-hypothesis pairs from structured features
- Uses NLI to vectorize LLM-generated hypotheses
- Automatically falls back to Phase 1 embeddings if LLM output is invalid

📖 See [`Cenliea_plus/README.md`](./Cenliea_plus/README.md)

---

### [`ea_classifier/`](./ea_classifier)

- Trains a lightweight binary classifier using NLI (or NLI+LLM) embeddings
- Supports hyperparameter tuning and evaluation
- Produces final `.pkl` models and `.json` evaluation statistics

📖 See [`ea_classifier/README.md`](./ea_classifier/README.md)

---

## ⚙️ Setup

Each module contains its own `requirements.txt`.

For example, to set up the classifier:

```bash
cd ea_classifier
pip install -r requirements.txt
```

Make sure you are logged in to [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub) to access HuggingFace models:

```python
from huggingface_hub import login
login()
```

---

## 📬 Contact

This repository is part of an anonymous submission. Please use the OpenReview discussion page for questions or feedback.
