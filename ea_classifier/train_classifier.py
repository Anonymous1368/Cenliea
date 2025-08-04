###############################################################################
# Grid Search Training Script for EA Classifier (Sklearn MLP)
#
# This script:
# 1. Loads vectorized entity alignment data from CSV
# 2. Cleans duplicates (keeps label=1 if conflict exists)
# 3. Parses 768-D vectors and splits into train/val sets
# 4. Performs hyperparameter tuning with GridSearchCV
# 5. Retrains final model on full data using best params
# 6. Saves trained model as a .pkl file
###############################################################################



import os
import sys
import ast
import argparse
import numpy as np
import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, make_scorer

###############################################################################
# 1) ARGUMENT PARSING
###############################################################################
# Parse CLI arguments for data path, output, and vector column
parser = argparse.ArgumentParser(description="Sklearn MLP GridSearch for entitiy alignment binary classification using vec_col")
parser.add_argument("--dp", required=True, help="Path to the CSV file with columns [Cenliea_vectors, label]")
parser.add_argument("--split", required=True, help="Data split name (not strictly used here)")
parser.add_argument("--output", default=None, help="Path to save the final model")
parser.add_argument("--eq", default="default", help="Equality mode for naming outputs")
parser.add_argument("--vec_col", default="Cenliea_vectors", help="Name of the column containing input vectors")

args = parser.parse_args()

dp          = args.dp
split       = args.split
output_file = args.output
vec_col     = args.vec_col

###############################################################################
# 2) MAKE OUTPUT DIRECTORY
###############################################################################

# Create output directory if needed
if output_file is None:
    base_dir = os.path.dirname(dp)
    out_dir = base_dir
else:
    out_dir = os.path.dirname(output_file)
    if out_dir == "":
        out_dir = "."

os.makedirs(out_dir, exist_ok=True)

# Output model path
wrapper_path = os.path.join(out_dir, f"best_sklearn_mlp_wrapper_{args.eq}.pkl")

###############################################################################
# 3) LOAD DATA
###############################################################################
df = pd.read_csv(dp)
print(f"Loaded {len(df)} rows from {dp}")

# Show stats on duplicated entity pairs
dupes = df[df.duplicated(subset="entities", keep=False)]
dupe_counts = dupes.groupby("label").size()
print("\n🔁 Duplicated rows in CSV by label:")
print(dupe_counts)

# Group by 'entities' and filter for those with exactly 2 occurrences (i.e. true duplicates)
duplicated_groups = dupes.groupby("entities").filter(lambda x: len(x) > 1)

# Display the first 3 duplicated 'entities' groups
first_3_entities = duplicated_groups["entities"].drop_duplicates().head(3)

print("\n🔍 First 3 duplicated entity groups in NLI CSV:")
for ent in first_3_entities:
    print(f"\n▶ Entity Pair: {ent}")
    print(dupes[dupes["entities"] == ent].to_string(index=False))

dupes = df["entities"].duplicated().sum()
print(f"🔍 Duplicated rows in NLI CSV:  {dupes}")

# Drop duplicated rows, prioritizing label=1
df = df.sort_values(by="label", ascending=False)
df = df.drop_duplicates(subset="entities", keep="first").reset_index(drop=True)
print("Number of non-duplicated samples: ", len(df))

# Parse vector column (from stringified JSON to list of floats)
def parse_list(s):
    if isinstance(s, str):
        try:
            # Try parsing as a valid Python list
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            # Fallback: parse space-separated floats inside brackets
            s = re.sub(r"[\[\]]", "", s)  # remove [ and ]
            return [float(x) for x in s.strip().split()]
    return s


df[vec_col] = df[vec_col].apply(parse_list)
X_list = df[vec_col].tolist()

X = np.array(X_list, dtype=np.float32)
y = df['label'].values

print("Feature shape:", X.shape, "Label shape:", y.shape)

###############################################################################
# 4) TRAIN/VAL SPLIT
###############################################################################

# Stratified train/val split (90/10)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

X_full = np.concatenate([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

###############################################################################
# 5) DEFINE GRID SEARCH
###############################################################################
#  (128,16,8), (256,128), (128,64),  (128,),

# Hyperparameter search grid for MLPClassifier
param_grid = {
    "hidden_layer_sizes": [
        (64,16), (16,64), (128,64),
    ],
    "learning_rate_init": [ 0.01, 0.001, 0.0005, 0.0001],
    "activation": ["tanh", "relu"],
    "max_iter": [50, 100, 150],
    "solver": ["adam", "sgd"]
}

scorer = make_scorer(f1_score, pos_label=1)

# Run 4-fold CV grid search in parallel
mlp = MLPClassifier(random_state=42)
grid_search = GridSearchCV(mlp, param_grid, scoring=scorer, cv=4, verbose=3, n_jobs=-1)
grid_search.fit(X_full, y_full)

# Log tuning results
print("\n=== Hyperparameter tuning results ===")
results = grid_search.cv_results_
for mean_score, std_score, params in zip(results["mean_test_score"], results["std_test_score"], results["params"]):
    print(f"Params: {params} => Mean F1: {mean_score:.4f} (+/- {std_score:.4f})")

print("\nBest Validation F1:", grid_search.best_score_)
print("Best Hyperparameters:", grid_search.best_params_)

###############################################################################
# 6) RETRAIN ON FULL DATA
###############################################################################
print("\nRetraining on the entire dataset with best hyperparams...")


X_full = np.concatenate([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

# Fit best model on all available data
best_mlp = MLPClassifier(**grid_search.best_params_, random_state=42)
best_mlp.fit(X_train, y_train)

# Evaluate performance
y_pred = best_mlp.predict(X_val)
final_f1 = f1_score(y_val, y_pred, pos_label=1)

print(f"\nFinal model F1 score on full data: {final_f1:.4f}")


###############################################################################
# 7) SAVE FINAL MODEL
###############################################################################
# Save model to disk
joblib.dump(best_mlp, wrapper_path)
print(f"Saved final sklearn MLP model to {wrapper_path}")

print("Done.")
