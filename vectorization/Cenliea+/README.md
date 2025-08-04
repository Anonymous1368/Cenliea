# 🔁 Cenliea+: LLM-Augmented NLI Embedding

Cenliea+ builds on the Cenliea pipeline by incorporating **alignment reasoning from a Large Language Model (LLM)** (Mistral-7B-Instruct) into the embedding process.

---

## 🧪 Pipeline Overview

1. Load `.csv` dataset containing entity pairs with `query`, `entity1_text`, and `entity2_text`.
2. Use `mistralai/Mistral-7B-Instruct-v0.2` to generate alignment evidence (e.g., `(2, a)`).
3. Parse the LLM output and retrieve referenced attribute-value pairs.
4. Use `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` to embed the tuples.
5. Fallback to existing Cenliea vectors if no valid reasoning is found.

---

## 🚀 Usage

```bash
python cenliea_plus_phase2_embed.py \
  --dp path/to/nli_vectorized.csv \           # Input CSV file from Cenliea Phase 1 (with NLI vectors, queries, etc.)
  --split test \                              # Dataset split label (used in logs or naming output)
  --output path/to/CenlieaPlus_output.csv \   # Path to save the final CSV with LLM-augmented vectors
```

### 🔧 Optional Arguments

- `--max_samples`: Limit the number of samples processed (for debugging or testing).
- `--nli_only`: Path to another Cenliea vector CSV file used for fallback.
- `--vec_col`: Column name in fallback file containing 768D NLI-only (No LLM) vectors.
- `--gpu_ids`: Comma-separated GPU IDs to use (e.g., `0,1`).
- `--batch_size`: Batch size for NLI vectorization.

---

## 💾 Output

Each output CSV includes:
- `response_Cenliea_plus`: Raw response from Mistral.
- `tuples`: Extracted (attr1, attr2) justification pairs.
- `tuple_count`: Number of valid pairs.
- `Cenliea_plus_vectors`: 768D vector (LLM + NLI or fallback-to-NLI vectors).
- `inference_mode`: `"mistral_augmented"` or `"fallback_to_nli"`.

---

## 🧩 Dependencies

```bash
pip install -r requirements.txt
```

Login to Hugging Face before use:
```python
from huggingface_hub import login
login(token="your_token_here")
```

---

## 📌 Notes

- LLM prompting uses a consistent one-shot template with numbered features.
- Tuples like `(3, b)` are mapped back to entity feature values.
- GPU (≥16 GB) is recommended for running Mistral inference.
