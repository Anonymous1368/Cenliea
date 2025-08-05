### 🔁 Using Pretrained Classifiers (Zero-Shot Inference)

We provide two pretrained MLP classifiers:

- `best_mlp_CENLIEA.pkl`
- `best_mlp_CENLIEA_PLUS.pkl`

These can be loaded for **zero-shot inference** without retraining or fine-tuning.

To apply these classifiers to a new dataset:

1. **Prepare the input data** using the scripts in [`dataset_preparation/`](../dataset_preparation).
2. **Vectorize your data** using the appropriate pipeline:
   - [`vectorization/cenliea/`](../vectorization/Cenliea) for standard NLI-based embeddings.
   - [`vectorization/cenliea_plus/`](../vectorization/Cenliea_plus) for LLM-enhanced embeddings.
3. **Evaluate alignment predictions** by loading the `.pkl` model in [`ea_classifier/`](../ea_classifier) and running the `test_binary_classifier.py` script on the vectorized `.csv` files.

This pipeline supports fast and modular evaluation on any structured entity pair dataset.

---

**Training Setup**: The classifiers were trained on a **Mixed dataset** of ~32,000 samples, derived from knowledge graphs in AgroLD, SPIMBENCH, and 5 KGs from the OAEI Knowledge Graph Track.  
>
> 📦 **Datasets Used for Training**:
> - OAEI Knowledge Graph Track v4: [Download](https://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v4/knowledgegraph_v4.zip)
> - SPIMBENCH: [Download](https://oaei.ontologymatching.org/2018/spimbench.html)
> - AgroLD: [Paper](https://academic.oup.com/database/article/doi/10.1093/database/baab036/6272502)
