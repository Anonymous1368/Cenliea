#!/usr/bin/env python
import os
import torch
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from tqdm import tqdm
import glob

from sklearn.metrics import precision_recall_fscore_support

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--Dataset_path', required=True, help='Path to dataset folder')
parser.add_argument('--split', default="test", help='Split folder name (e.g. test)')
parser.add_argument('--pattern', default="*.parquet", help='Pattern for shard files')
parser.add_argument('-o', '--output', default=None, help='Final merged CSV path')
args = parser.parse_args()

# === Extract dataset name from the dataset path ===
dataset_name = os.path.basename(args.Dataset_path)

# === Define output base dir ===
output_dir = os.path.join(args.output, dataset_name)

# === Make sure the output folder exists ===
os.makedirs(output_dir, exist_ok=True)

split_dir = os.path.join(args.Dataset_path, args.split)
output_path = os.path.join(output_dir, f"{args.split}_nli_predictions.csv")


model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Helper Functions ===
def get_embedding(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.hidden_states[-1].squeeze(0)
        token_type_ids = inputs["token_type_ids"].squeeze(0)
        num_zeros = (token_type_ids == 0).sum().item()
        s1 = hidden[:num_zeros].mean(dim=0)
        s2 = hidden[num_zeros:].mean(dim=0)
        return torch.stack([s1.cpu(), s2.cpu()])  # shape: (2, 768)

def bidirectional_embedding(p, h):
    e1 = get_embedding(p, h)  # forward
    e2 = get_embedding(h, p)  # backward
    return torch.stack([e1, e2])  # shape: (2, 2, 768)

def structured_mean_concat(embedding):
    fp = embedding[0, 0, :]  # forward premise
    fh = embedding[0, 1, :]  # forward hypothesis
    bp = embedding[1, 0, :]  # backward premise
    bh = embedding[1, 1, :]  # backward hypothesis
    pool = lambda x: torch.nn.functional.avg_pool1d(x.unsqueeze(0).unsqueeze(0), kernel_size=4).squeeze()
    return torch.cat([pool(fp), pool(fh), pool(bp), pool(bh)], dim=0)  # (768,)

# === Process All Shards ===
all_records = []
shard_files = sorted(glob.glob(os.path.join(split_dir, args.pattern)))
print(f"Found {len(shard_files)} shards...")

for idx, file in enumerate(shard_files):
    print(f"Processing: {file}")
    dataset = load_dataset("parquet", data_files=file, split="train")

    embeddings = []
    original_items = []

    for item in tqdm(dataset, desc=f"Embedding {os.path.basename(file)}"):
        p = item["entity1_text"]
        h = item["entity2_text"]
        emb = bidirectional_embedding(p, h)
        flat = structured_mean_concat(emb).numpy()
        embeddings.append(flat)
        item["Cenliea_vectors"] = flat.tolist()
        original_items.append(item)

    print("Length of pooled vector:", len(flat))

    for item in original_items:
        item["inference_mode"] = "NLI_only"

    # Save per-shard results
    shard_df = pd.DataFrame(original_items)
    shard_filename = args.split + '_' + os.path.basename(file).replace(".parquet", f"_predictions.csv")
    shard_output_path = os.path.join(output_dir, shard_filename)
    shard_df.to_csv(shard_output_path, index=False)
    print(f"Saved shard vectors to: {shard_output_path}")

    # Accumulate all for merged output
    all_records.extend(original_items)

# === Save Merged CSV ===
df = pd.DataFrame(all_records)
df.to_csv(output_path, index=False)
print(f"Merged predictions saved to: {output_path}")

# === Compute and Save Statistics (Commenting out classifier-based stats) ===
stats_lines = []
stats_lines.append(f"Total samples: {len(df)}")

stats_path = os.path.splitext(output_path)[0] + "_stats.txt"
with open(stats_path, "w") as f:
    f.write("\n".join(stats_lines))

print(f"\nStatistics saved to: {stats_path}")
print("\n".join(stats_lines))
