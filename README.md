# Cenliea: Cross-Encoder NLI-based embedding for Entity Alignment

Code Appendix for the Paper **Cenliea: A Cross-Encoder NLI Framework Enhanced by LLM-Based Reasoning for Transferable, Domain-Agnostic Entity Alignment**.

This repository provides the complete codebase for **Cenliea** and **Cenliea+**, two complementary pipelines for entity alignment (EA) across heterogeneous knowledge graphs (KGs). The framework combines structured input encoding, multilingual NLI-based embeddings, and LLM-augmented reasoning to offer a generalizable, explainable alignment approach.

The **Cenliea** pipeline uses cross-encoder NLI embeddings to assess alignment between entity descriptions from heterogeneous KGs. It leverages the [mDeBERTa-v3-base-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) model to generate joint semantic entailment vectors from structured entity features.

**Cenliea+** extends this by prompting a large language model ([Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)) to generate salient alignment hypotheses, which are then transformed into NLI inputs. The resulting embeddings are pooled and passed to a binary classifier to determine entity correspondence. 

---

## 🧭 Cenliea and Cenliea+ Pipelines

<img src="assests/AAAI_Diagram.png" alt="CENLIEA Pipeline" width="400"/>

## 🧭 Embedding Procedure

<img src="assests/Embedding_procedure.png" alt="CENLIEA+ Embedding Procedure" width="400"/>

## 📌 Overview

Cenliea introduces a neurosymbolic EA pipeline with the following core phases:
- 🧱 **Structured prompt construction** from RDF graphs
- 🧠 **NLI vectorization** for entity pair similarity estimation
- 📈 **Binary classifier training** for alignment detection
- 🔍 **Cenliea+ (optional)**: LLM-generated explanations augment difficult or low-confidence cases

---

## 🧱 Components

### [1. `dataset_preparation/`](./dataset_preparation)

- Converts RDF/XML files into structured JSON
- Extracts direct and one-hop features
- Computes GPT-token similarity for negative sampling
- Produces `.parquet` files for downstream NLI encoding

📖 See [`dataset_preparation/README.md`](./dataset_preparation/README.md)

---
### [2. `vectorization/`](./vectorization)

#### [2.1. `Cenliea/`](./vectorization/Cenliea)

- Phase 1 of Cenliea: Bidirectional NLI embeddings using [`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
- Loads `.parquet` input datasets and outputs embeddings per pair
- Outputs are cached for reuse or fallback scenarios in Cenliea+

📖 See [`Cenliea/README.md`](./vectorization/Cenliea/README.md)

---

#### [2.2. `Cenliea_plus/`](./vectorization/Cenliea_plus)

- Optional Phase 2: LLM-based alignment justification using [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- Generates premise-hypothesis pairs from structured features
- Uses NLI to vectorize LLM-generated hypotheses
- Automatically falls back to Phase 1 embeddings if LLM output is invalid

📖 See [`Cenliea_plus/README.md`](./vectorization/Cenliea_plus/README.md)

---

### [3. `ea_classifier/`](./ea_classifier)

- Train and test a lightweight binary classifier using NLI or NLI+LLM embeddings.
- Includes:
  - `hp_tuning_train_binary_classifier.py`: Performs hyperparameter tuning and saves the trained model as a `.pkl` file.
  - `test_binary_classifier.py`: Loads the saved model and evaluates it on a held-out test set.
- Outputs include trained classifiers and `.json` files containing macro/micro F1 scores and confusion matrices.

📖 See [`ea_classifier/README.md`](./ea_classifier/README.md)

---

### [4. `pretrained_models/`](./pretrained_models)

- Contains pretrained lightweight binary classifiers trained on 768-D embeddings from **Cenliea** and **Cenliea+** pipelines.
- Classifiers are stored as `.pkl` files:
  - `best_sklearn_mlp_Cenliea.pkl`
  - `best_sklearn_mlp_Cenliea_plus.pkl`
- These models can be directly applied to new datasets **without retraining**.

**To use:**
1. Prepare and vectorize a new dataset using:
   - [`dataset_preparation`](./dataset_preparation/)
   - [`vectorization/Cenliea`](./vectorization/Cenliea) or [`vectorization/Cenliea_plus`](./vectorization/Cenliea_plus)

2. Use the evaluation script in [`ea_classifier`](./ea_classifier) to test performance:
```bash
python test_binary_classifier.py \
  --dp path/to/your/vectorized.csv \
  --split test \
  --model_path pretrained_classifiers/best_sklearn_mlp_Cenliea.pkl \
  --output results_on_new_data.json
```

📖 See [`ea_classifier/README.md`](./ea_classifier/README.md) for argument details.

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

## 🧪 Reproducibility & Experimental Setup

- Binary classifiers are trained on 768-dimensional NLI embeddings produced by the [Cenliea](./cenliea/) and [Cenliea+](./cenliea_plus/) pipelines.
- Lightweight MLP architectures were tuned using `GridSearchCV` with 4-fold cross-validation, optimizing positive-class F1-score.
- Final models:
  - **Cenliea**: 128-64 MLP, ReLU activation, learning rate = 0.001
  - **Cenliea+**: 64-16 MLP, Tanh activation, learning rate = 0.0005
- LLM outputs were evaluated for stability. Repeated runs of **Cenliea+** showed minimal performance variance (≤±5%), confirming robustness.

## 💻 Runtime Environment

All experiments were run on **Google Colab Pro**:

- **Cenliea** used **NVIDIA L4 GPUs**
- **Cenliea+** used **NVIDIA A100 GPUs**

Average per-sample runtime:

| Pipeline   | Phase                        | Time / Sample |
|------------|------------------------------|----------------|
| Cenliea    | NLI Embedding                | 0.055 sec      |
| Cenliea+   | LLM Inference + NLI Embedding| 0.421 sec      |

---

## 📬 Contact

This repository is part of an anonymous submission. Please use the OpenReview discussion page for questions or feedback.
