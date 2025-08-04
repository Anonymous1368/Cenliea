#!/usr/bin/env python
import os
import argparse
import json
from datasets import load_dataset, disable_progress_bar
# Disable unnecessary progress bars from the datasets library
disable_progress_bar()
import pandas as pd


# Parse arguments: dataset path (root directory of train/test/test_imb shards)
parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--Dataset_path', required=True, help='Path to dataset root folder')
parser.add_argument('--num_proc', type=int, default=8, help='Number of CPU processes')
args = parser.parse_args()

# Format direct entity attributes into enumerated text (e.g., "1. hasTitle: example")
def format_direct_features(direct_dict):
    """
    Replicates the style used in 1_generate_train_test_neg_sampling.py,
    e.g.
      1. pagestotales: 364,
      2. année: 2010,
      a. description: ...
      b. dateofdeath: ...
    """
    lines = []
    for k, v in direct_dict.items():
        # k might look like "1. pagestotales" or "a. description"
        lines.append(f"{k}: {v},")
    return "\n".join(lines)

# Format 1-hop neighbor features in a structured format used by prompting (e.g., "R-1.2. hasGenre -> label: value")
def format_neighbor_features(neighbors_dict):
    """
    Replicates neighbor features style used in make_prompt():
    R-1.1.  <relation_label> ->  <attr_label>: <value>,
    R-a.2.  <relation_label> ->  <attr_label>: <value>,
    ...
    """
    lines = []
    for rel_key, attr_dict in neighbors_dict.items():
        # rel_key might look like "1. maximilien de robespierre"
        # or "a. some_other_entity"
        # We'll parse out the prefix before the dot (e.g. "1" or "a")
        # then the label after the dot
        rel_prefix = rel_key.split('.')[0]  # e.g. "1" or "a"
        # remove "<prefix>. " => 2 characters for "<prefix>." plus 1 space => +2
        rel_label = rel_key[len(rel_prefix)+2:] if len(rel_key) > len(rel_prefix)+2 else rel_key

        if attr_dict:
            # For each attribute inside this neighbor
            for attr_key, val in attr_dict.items():
                # attr_key might be "1. pagestotales" => prefix="1", label="pagestotales"
                attr_prefix = attr_key.split('.')[0]
                attr_label = attr_key[len(attr_prefix)+2:] if len(attr_key) > len(attr_prefix)+2 else attr_key
                lines.append(f"R-{rel_prefix}.{attr_prefix}.  {rel_label} ->  {attr_label}: {val},")
        else:
            # If this neighbor has no attributes
            lines.append(f"R-{rel_prefix}. {rel_label},")
    return "\n".join(lines)

# Extracts and formats entity1/entity2 descriptions from JSON fields into final text format
def extract_from_json_fields(item):
    """
    Reads each row's self-attr-val and 1hop-attr-val,
    then builds entity1_text and entity2_text with the same enumerations used
    in 1_generate_train_test_neg_sampling.py.
    """
    try:
        self_attr = json.loads(item["self-attr-val"])
    except Exception as e:
        print(f"[Warning] Failed to parse self-attr-val: {item['self-attr-val']}, error: {e}")
        self_attr = {"entity1": {}, "entity2": {}}

    try:
        hop_attr = json.loads(item["1hop-attr-val"])
    except Exception as e:
        print(f"[Warning] Failed to parse 1hop-attr-val: {item['1hop-attr-val']}, error: {e}")
        hop_attr = {"entity1": {}, "entity2": {}}

    # Get direct-attributes dictionaries
    ent1_self = self_attr.get("entity1", {})
    ent2_self = self_attr.get("entity2", {})

    # Get neighbor-attributes dictionaries
    ent1_hop = hop_attr.get("entity1", {})
    ent2_hop = hop_attr.get("entity2", {})

    # Format them the same as in make_prompt()
    entity1_text = (
        "Entity 1 direct features:\n"
        + format_direct_features(ent1_self)
        + "\nEntity 1 neighbor's features:\n"
        + format_neighbor_features(ent1_hop)
    )

    entity2_text = (
        "Entity 2 direct features:\n"
        + format_direct_features(ent2_self)
        + "\nEntity 2 neighbor's features:\n"
        + format_neighbor_features(ent2_hop)
    )

    return {
        "entity1_text": entity1_text,
        "entity2_text": entity2_text
    }

if __name__ == '__main__':
    # splits = ["train", "test", "test_imb"]
    # Automatically find all subdirectories (e.g., train, test, test_imb, etc.)
    all_splits = [
        d for d in os.listdir(args.Dataset_path)
        if os.path.isdir(os.path.join(args.Dataset_path, d))
    ]

    for split in all_splits:
        split_path = os.path.join(args.Dataset_path, split)
        shard_files = sorted(f for f in os.listdir(split_path) if f.endswith(".json") and "shard" in f)
        # Loop over each split and process all shard JSON files
        for fname in shard_files:
            fpath = os.path.join(split_path, fname)
            print(f"Processing: {fpath}")
            dataset = load_dataset("json", data_files=fpath, split="train")
            # Apply text extraction function to all samples (parallelized)
            dataset = dataset.map(extract_from_json_fields, num_proc=args.num_proc)
            out_path = os.path.splitext(fpath)[0] + ".parquet"
            # Save the processed dataset with added entity1_text/entity2_text fields as a .parquet file
            dataset.to_parquet(out_path)
            print(f"Saved: {out_path}")
