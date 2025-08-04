# 🧠 Cenliea: NLI-Based Embedding for Entity Alignment

This module computes bidirectional **Natural Language Inference (NLI)** embeddings using the pretrained [MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) model.

Each pair of entities is represented as two textual inputs (`entity1_text`, `entity2_text`). The model encodes their semantic relationship bidirectionally and outputs a 768-dimensional pooled embedding.

---

## 🚀 Usage

```bash
python cenliea_phase1_embed.py \
  -dp /path/to/dataset \
  --split test \
  --pattern "*.parquet" \
  -o /path/to/output/
```

### 🔧 Arguments

- `-dp`: Path to the dataset directory (should contain `.parquet` files).
- `--split`: Dataset split to process (`train`, `test`, etc.).
- `--pattern`: Glob pattern for reading shard files (default: `"*.parquet"`).
- `-o`: Output directory where the vectorized `.csv` files will be saved.

---

## 💾 Output

Each output CSV includes:
- `Cenliea_vectors`: 768D vector per sample from NLI inference.
- `inference_mode`: ASet to `"NLI_only"` for Cenliea.

---

## 🧩 Dependencies

```bash
pip install -r requirements.txt
```

Login to Hugging Face before running the script:
```python
from huggingface_hub import login
login(token="your_token_here")
```

---

## 📌 Notes

- This is baseline **Cenliea**Cenliea pipeline.
- No LLM prompting or fallback logic is applied.
- Output files can be used directly for classification or downstream reasoning.
