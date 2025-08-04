# EA Classifier: Training and Evaluation

This directory provides tools to **train** and **evaluate** a binary classifier on entity alignment data that has been vectorized via either:

- 🟦 `Cenliea`: using direct bidirectional NLI embeddings, or
- 🟪 `Cenliea+`: using LLM-derived hypotheses augmented with NLI embeddings.

---

## 🔧 Contents

- `train_classifier.py`: GridSearch-based hyperparameter tuning and final training using `MLPClassifier` (scikit-learn).
- `test_classifier.py`: Evaluation of trained classifiers on new datasets with prediction, confidence, and breakdown analysis.

---

## ⚙️ Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

Required packages:
- `scikit-learn`
- `numpy`
- `pandas`
- `joblib`

---

## 🚀 Usage

### 🔹 Train Classifier

```bash
python train_classifier.py \
  --dp /path/to/train_data.csv \
  --split train \
  --output /path/to/save/model.pkl \
  --vec_col Cenliea_vectors \
  --eq Cenliea
```

**Arguments**:
- `--dp`: Path to CSV file with input vectors and labels.
- `--split`: Data split name (used in filenames).
- `--output`: Path to save the trained `.pkl` model.
- `--vec_col`: Column name containing vectorized embeddings (default: `Cenliea_vectors`).
- `--eq`: Tag for model naming (e.g., "Cenliea" or "CenlieaPlus").

---

### 🔹 Evaluate Classifier

```bash
python test_classifier.py \
  --dp /path/to/test_data.csv \
  --model /path/to/model.pkl \
  --output /path/to/output_with_predictions.csv \
  --json_out eval_stats.json \
  --vec_col Cenliea_vectors
```

**Arguments**:
- `--dp`: Path to test CSV file or directory.
- `--model`: Path to trained `.pkl` model.
- `--output`: Path to save test predictions with confidence.
- `--json_out`: Output path for evaluation summary JSON.
- `--vec_col`: Vector column name (`Cenliea_vectors`, `CenlieaPlus_vectors`, etc.).

---

## 📊 Output

- Predictions saved to `.csv` with predicted labels, confidences, and probabilities.
- Evaluation summary `.json` with classification report and confusion matrix (overall and per-subset).

The **JSON** contains:
- **Overall metrics**: accuracy, precision, recall, F1-score, confusion matrix
- **Per-subset breakdowns**: results for various dataset slices (`mistral_augmented`, `fallback_and_full`, etc.)
- **Label distribution**: True vs predicted distribution of alignment labels
- **Inference mode distribution**: Number of samples per alignment strategy (e.g., LLM-augmented vs fallback)

### 📈 Example Metrics (from sample JSON)

- Total Samples: 16,223
- Overall Accuracy: `93.1%`
- Positive-class F1: `93.3%`
- Subset (`mistral_augmented`) Accuracy: `95.0%`
- Subset (`fallback_and_full`) Accuracy: `91.2%`

### 📊 Subset Definitions

- **mistral_augmented**: Samples for which LLM-generated alignment justifications were successfully produced and used for vectorization
- **fallback_and_full**: Samples where the LLM failed to generate a valid response and the fallback NLI-only vectors were used
- **all**: Complete test set
- **dataset::X**: Per-dataset metrics (e.g. AgroLD, StarWars, Marvel, etc.)

---

## 🧹 Note on Duplicates

Both scripts automatically handle entity-pair duplicates. When conflicts exist (e.g., same pair with both labels 0 and 1), the positive sample (`label=1`) is retained.

---

## 📁 Suggested Structure

```
ea_classifier/
├── train_classifier.py
├── test_classifier.py
├── requirements.txt
└── README.md
```
