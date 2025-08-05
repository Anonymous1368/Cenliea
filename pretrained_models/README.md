### 🔁 Using Pretrained Classifiers (Zero-Shot Inference)

We provide two pretrained MLP classifiers:

- `best_mlp_CENLIEA.pkl`
- `best_mlp_CENLIEA_PLUS.pkl`

These can be loaded for **zero-shot inference** without retraining or fine-tuning.

To apply these classifiers to a new dataset:

1. **Prepare the input data** using the scripts in [`dataset_preparation/`](./dataset_preparation).
2. **Vectorize your data** using the appropriate pipeline:
   - [`vectorization/cenliea/`](./vectorization/cenliea) for standard NLI-based embeddings.
   - [`vectorization/cenliea_plus/`](./vectorization/cenliea_plus) for LLM-enhanced embeddings.
3. **Evaluate alignment predictions** by loading the `.pkl` model in [`ea_classifier/`](./ea_classifier) and running the `test_binary_classifier.py` script on the vectorized `.csv` files.

This pipeline supports fast and modular evaluation on any structured entity pair dataset.
