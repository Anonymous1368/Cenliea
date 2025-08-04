#!/usr/bin/env python3
"""
test_binary_classifier.py
----------------------------------------------------

Evaluate an already trained binary classifier (saved as a scikit-learn .pkl file)
on vectorized entity alignment data. Supports batch prediction, confidence scores,
per-subset breakdowns (e.g., by `dataset`, `inference_mode`), and JSON stats export.

Features:
- Loads input vectors (from `.csv` files or directory) and a classifier `.pkl`
- Handles duplicates by keeping only one label per unique entity pair
- Outputs a new `.csv` with prediction labels, confidence, and probabilities
- Exports detailed evaluation statistics in `.json` format

Example usage:
--------------
```bash
python test_binary_classifier.py \
  --dp "/path/to/input.csv" \
  --model "/path/to/Cenliea.pkl" \
  --output "/path/to/output_with_preds.csv" \
  --json_out "eval_stats.json"
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os
import re
# helper for per‑subset evaluation (same as before)
from sklearn.metrics import precision_recall_fscore_support

# === Parse arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--dp",     required=True, help="Input CSV or directory of CSV files containing entity vectors")
parser.add_argument("--model",  required=True, help="Path to scikit-learn model (.pkl) trained with MLPClassifier")
parser.add_argument("--output", required=True, help="Path to save updated CSV with predictions")
parser.add_argument("--json_out", default="inference_eval_stats.json",
                    help="Path to save evaluation statistics as JSON")
parser.add_argument("--vec_col", default="Cenliea_vectors", help="Column name containing JSON-encoded vectors")
args = parser.parse_args()
vec_col = args.vec_col
print("column to classify: ", vec_col)

# ----------------------------------------------------------------------
# 1)  Load data  – vector column is NLI_vectors_before (JSON list → np.float32)
# ----------------------------------------------------------------------

def fast_vec(s):
    """ Convert vector strings or lists to np.float32 arrays,
        Supports JSON or NumPy-style string representations"""
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.array(s, dtype=np.float32)
    elif isinstance(s, str):
        s = s.strip()
        try:
            # Try JSON parsing first
            return np.array(json.loads(s), dtype=np.float32)
        except json.JSONDecodeError:
            # Fallback: parse NumPy-style array string
            try:
                float_list = list(map(float, re.findall(r'-?\d+\.\d+|-?\d+', s)))
                return np.array(float_list, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Unrecognized vector format: {s[:80]}...") from e
    else:
        raise TypeError(f"Unexpected type for vector input: {type(s)}")


def load_all_csvs(input_path_pattern):
    """
    # Load multiple CSV files and merge them
    # Drop duplicate rows based on the 'entities' column (if any exists)
    # Keep the first occurrence to ensure a single label per pair
    (if by fault the negative sample is chosen as the correct positive candidate)
    """
    csv_files = glob.glob(input_path_pattern)
    print("CSV files to read:\n", csv_files)
    all_dfs = []
    for fpath in csv_files:
        df = pd.read_csv(fpath, dtype={"label": int},
                         converters={vec_col: fast_vec})
        df["source_file"] = os.path.basename(fpath)
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)

    # Count duplicated rows grouped by label for NLI CSV
    dupes = df[df.duplicated(subset="entities", keep=False)]
    dupe_counts = dupes.groupby("label").size()

    # Group by 'entities' and filter for those with exactly 2 occurrences (i.e. true duplicates)
    duplicated_groups = dupes.groupby("entities").filter(lambda x: len(x) > 1)
    print("\n🔁 Duplicated rows in CSV by label:")
    print(dupe_counts)
    dupes = df["entities"].duplicated().sum()

    # Sort so that label=1 rows come first
    df = df.sort_values(by="label", ascending=False)
    print(f"🔍 Duplicated rows in NLI CSV:  {dupes}")
    # Drop duplicates based on 'entities' column in both DataFrames
    df = df.drop_duplicates(subset="entities", keep="first").reset_index(drop=True)
    print("Number of non-duplicated samples: ", len(df))

    return pd.concat([df], ignore_index=True)

# Check if the given path is a directory
if os.path.isdir(args.dp):
    if "NLI_only" in args.model or "Cenliea.pkl" in args.model or "Cenliea_ZhEn" in args.model:
        pattern = os.path.join(args.dp, "*nli_predictions.csv")
    if "CLEA" in args.model or "Cenliea_plus.pkl" in args.model or "Cenliea_plus_ZhEn" in args.model:
        pattern = os.path.join(args.dp, "*NeuroSym_with_response.csv")

    df = load_all_csvs(pattern)
    print(f"Loaded {len(df)} rows from {len(df['source_file'].unique())} files in {args.dp}")
else:
    df = pd.read_csv(args.dp, dtype={"label": int},
                     converters={vec_col: fast_vec})
    # Count duplicated rows grouped by label for NLI CSV
    dupes = df[df.duplicated(subset="entities", keep=False)]
    dupe_counts = dupes.groupby("label").size()

    # Group by 'entities' and filter for those with exactly 2 occurrences (i.e. true duplicates)
    duplicated_groups = dupes.groupby("entities").filter(lambda x: len(x) > 1)
    print("\n🔁 Duplicated rows in CSV by label:")
    print(dupe_counts)
    dupes = df["entities"].duplicated().sum()

    print(f"🔍 Duplicated rows in NLI CSV:  {dupes}")
    # Sort so that label=1 rows come first
    df = df.sort_values(by="label", ascending=False)
    # Drop duplicates based on 'entities' column in both DataFrames
    df = df.drop_duplicates(subset="entities", keep="first").reset_index(drop=True)
    print("Number of non-duplicated samples: ", len(df))
    print(f"Loaded {len(df)} rows from {args.dp}")

X_np = np.stack(df[vec_col].to_numpy())  # shape (N, 768)

# ----------------------------------------------------------------------
# 2)  Load the **sklearn** model (ImbPipeline or plain Pipeline) & predict
# ----------------------------------------------------------------------
print(f"Loading scikit‑learn model from: {args.model}")
model = joblib.load(args.model)
print("Model loaded – performing inference…")

# Run prediction + probability estimation
preds = model.predict(X_np)
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_np)
    confidences = probs.max(axis=1)
else:
    # fall‑back if the classifier lacks predict_proba()
    probs = np.zeros((len(preds), 2), dtype=np.float32)
    probs[np.arange(len(preds)), preds] = 1.0
    confidences = np.ones(len(preds), dtype=np.float32)

# ----------------------------------------------------------------------
# 3)  Update dataframe & save
# ----------------------------------------------------------------------
# Save predictions and metadata back to CSV
# Probabilities are stored as JSON-compatible lists
df["predicted_label"]       = preds
df["prediction_confidence"] = confidences
# store probs as Python lists (JSON serialisable)
df["prediction_probs"]      = probs.tolist()

df.to_csv(args.output, index=False)
print(f"✅ Predictions written to {args.output}")

# ----------------------------------------------------------------------
# 4)  Build evaluation JSON (unchanged logic)
# ----------------------------------------------------------------------

# Compute evaluation metrics: label distributions, PRF1, confusion matrix
# Group results by 'inference_mode' and 'dataset' if present
stats_json = {
    "total_samples": len(df),
    "label_distribution_true": df["label"].value_counts().to_dict(),
    "label_distribution_pred": df["predicted_label"].value_counts().to_dict(),
    "subsets": {},
}

if "inference_mode" in df.columns:
    stats_json["inference_mode_distribution"] = (
        df["inference_mode"].value_counts().to_dict()
    )


def subset_eval(name, subdf):
    report = classification_report(subdf["label"], subdf["predicted_label"],
                                   output_dict=True, zero_division=0)
    cm     = confusion_matrix(subdf["label"], subdf["predicted_label"]).tolist()
    stats_json["subsets"][name] = {
        "classification_report": report,
        "confusion_matrix": cm,
        "num_samples": len(subdf),
    }

# inference‑mode‑specific subsets
if "inference_mode" in df.columns:
    subset_eval("mistral_augmented", df[df["inference_mode"] == "mistral_augmented"])
    subset_eval("fallback_and_full",
                df[df["inference_mode"].isin(["fallback_to_nli", "full_only"])])

subset_eval("all", df)  # always include overall subset

# per‑dataset breakdown (if present)
if "dataset" in df.columns:
    for ds in df["dataset"].unique():
        subset_eval(f"dataset::{ds}", df[df["dataset"] == ds])

with open(args.json_out, "w") as f:
    json.dump(stats_json, f, indent=4)
print(f"📊 Stats written to {args.json_out}")
