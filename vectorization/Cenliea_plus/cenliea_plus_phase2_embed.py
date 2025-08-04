import os
import re
import argparse
import joblib
import pandas as pd
import numpy as np
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from torch.nn.utils.rnn import pad_sequence
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Argument parser to allow dynamic adapter path input
parser = argparse.ArgumentParser(description="Phase 2 Inference Pipeline for Entity Pair Classification")
parser.add_argument("--dp", required=True, help="Path to the dataset CSV file or directory containing it")
parser.add_argument("--split", required=True, help="Name of the data split (used for output naming)")
parser.add_argument("--output", help="Output CSV path for results")
parser.add_argument("--gpu_ids", default="0", help="Comma-separated GPU IDs to use (default: 0)")
parser.add_argument("--batch_size", default="32", help="Batch size for Mistral response generation (default: 16)")
parser.add_argument("--max_samples", type=int, default=None,
                    help="Maximum number of samples to process from the dataset (default: all)")
parser.add_argument("--nli_only", required=False, help="Optional CSV file with NLI_only_vector for fallback_to_nli correction")
parser.add_argument("--vec_col", default="Cenliea_vectors", help="Name of the column containing input vectors")


args = parser.parse_args()
vec_col = args.vec_col


# Load the base model (Mistral)
mistral_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
mistral_model = AutoModelForCausalLM.from_pretrained(
    mistral_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# Load the tokenizer
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)

mistral_model.eval()

# Set adapter if needed (typically named "default")
# mistral_model.set_adapter("default")

# Ensure pad_token is defined
if mistral_tokenizer.pad_token is None:
    mistral_tokenizer.pad_token = mistral_tokenizer.eos_token


def build_chat_prompt(item):
    """Build a one-shot chat prompt with example and the query features."""
    inst = (
        "Use numerical-alphabetical enumerators given in the prompt and give me "
        "tuples of features of Entity 1 and Entity 2 e.g. (2, a) which represent their salient similarities."
    )
    shot_sample = """Entity 1 direct features:
1. genre: historical fiction,
2. title: les misérables,
3. datecreated: 1862-04-03T11:55:27.366000+02:0,
Entity 2 direct features:
a. hasname: The wretched poor,
b. description: it is one of the greatest french novels,
c. dateprinted: 1862-03-31T08:00:00,

Entity 1 neighbor's features:
R-1. about -> rdf schema comment: it has been popularized through numerous adaptations,
R-2. publishedby -> displayname: A. Lacroix Verboeckhoven & Cie,
R-3. createdby -> hasname: Victor-Marie Hugo Vicomte Hugo,

Entity 2 neighbor's features:
R-a. author -> Victor Hugo,
R-b. publisher -> hasname: Jean Baptiste Constant Marie Albert Lacroix,
"""
    shot_response = """(2, a): Both entities have a similar title, although with English to French translation (2. title: les misérables, AND a. hasname: The wretched poor).
(3, c): Both entities have similar close dates in 1862 (3. datecreated: 1862-04-03, AND c. dateprinted: 1862-03-31).
(R-3, R-a): Both entities have the same author (R-3. createdby -> hasname: Victor-Marie Hugo Vicomte Hugo, AND R-a. author -> Victor Hugo)."""
    query = item["query"]
    return [
        {"role": "user", "content": query + inst}
    ]

def extract_tuples(row):
    """
    Extract (entity1_feature, entity2_feature) tuples from the model's response
    using enumerated IDs, with the same precautions used in evaluate_ref_model
    to avoid 'unbalanced parenthesis' errors and handle fallback if no matches.
    """

    pattern = "\(([^\)\(]+)\)"  #for mistral: \(([^\)\(]+)\)  ;for seq2seq: """\(([^\)|\(]+)\)'"`"""
    # for batch in query_response_pairs:
    query = str(row["query"])
    response = str(row["response_Cenliea_plus"])
      #print("query:\n", query)
      #print("response:\n", response)
    response_tuples = re.findall(pattern, response)#use regex to extract all texts between parentheses
    response_tuples = [t.replace(" ", "") for t in response_tuples]
    response_tuples = [t.replace("\t", "") for t in response_tuples]
      #print("response_tuples:\n", response_tuples)
    flag = 0
    sampl_result = []
    if response_tuples: #if there are any tuples
        #print("there are tuples in response")
        query = query.replace("[\{\}\[\]\(\)]", "")
        query_entity1 = str(row["entity1_text"])
        query_entity2 = str(row["entity2_text"])


        for tup in response_tuples:
          #print("tuple:", tup)
          tup = tup.split(',')
          if len(tup)==2: #if the tuple split by comma (Note: Avoid samples like: len(('ii'))==2)
            if tup[0] and tup[1]: #if entries of tuple is not None
              ind_prem =  tup[0]
              ind_hypo =  tup[1]
              try:
                prem_attr = re.findall(f"\n{ind_prem}\. ([\s\S]*?),", query_entity1)  # "^{ind_prem}\. ([\s\S]*?)\n"  :for mistral, " {ind_prem}\. (.*?), .+?\. "  :for seq2seq
              except:
                prem_attr = re.findall(re.escape(f"\n{ind_prem}\. ([\s\S]*?),"), query_entity1)
              if prem_attr: #if the first indicator is related to Entity 1
                #print("the first indicator is related/existed to Entity 1. Its related attribute value:\n", prem_attr, type(prem_attr))
                prem_attr = prem_attr[0]
              try:
                hyp_attr = re.findall(f"\n{ind_hypo}\. ([\s\S]*?),", query_entity2)
              except:
                hyp_attr = re.findall(re.escape(f"\n{ind_hypo}\. ([\s\S]*?),"), query_entity2)
              if hyp_attr: #if second indicator is related to Entity 2
                #print("the second indicator is related/existed to Entity 2. Its related attribute value:\n", hyp_attr, type(hyp_attr))
                hyp_attr = hyp_attr[0]
              if prem_attr and hyp_attr: #if indicators were not fake
                  premise, hypothesis = prem_attr, hyp_attr #use regex to extract the premise related to tup[0] and hypothesis regarding tup[1]
                  sampl_result.append([premise, hypothesis])
                  #print("correct premise, hypothesis:", premise, hypothesis)
                  flag = 1 #there is at least one correct tuple

    return sampl_result

def main():

    batch_size = int(args.batch_size)

    # Set CUDA devices for PyTorch if specified
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device_ids = [int(i) for i in args.gpu_ids.split(",") if i.strip().isdigit()]
    primary_device = torch.device(f"cuda:{device_ids[0]}") if len(device_ids) > 0 and torch.cuda.is_available() else torch.device("cpu")

    # Load the dataset
    data_files = args.dp
    if os.path.isdir(args.dp):
        file_candidate = os.path.join(args.dp, f"{args.split}.csv")
        if os.path.isfile(file_candidate):
            data_files = file_candidate

    df = pd.read_csv(data_files)
    ds = Dataset.from_pandas(df)
    # ds = load_dataset("csv", data_files=data_files, split="train")
    df = ds.to_pandas()

    # Limit to max_samples if specified
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    # If 'inference_mode' missing, create it
    if "inference_mode" not in df.columns:
        df["inference_mode"] = None

    to_process_df = df.copy()
    # Ensure the Cenliea_plus_vectors column exists for all rows
    if "Cenliea_plus_vectors" not in to_process_df.columns:
        to_process_df["Cenliea_plus_vectors"] = None

    # Load multilingual NLI model for embedding
    entail_model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    entail_tokenizer = AutoTokenizer.from_pretrained(entail_model_name)
    entail_model = AutoModelForSequenceClassification.from_pretrained(entail_model_name, output_hidden_states=True)
    entail_model.to(primary_device)
    entail_model.eval()

    # Load Mistral-7B-Instruct for generation
    mistral_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
    mistral_model = AutoModelForCausalLM.from_pretrained(
        mistral_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    mistral_model.eval()
    print("Mistral model loaded in dtype:", next(mistral_model.parameters()).dtype)

    if mistral_tokenizer.pad_token is None:
        mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
        mistral_tokenizer.padding_side = "left"
        mistral_tokenizer.truncation_side = "left"

    # Helper functions for NLI embedding
    def get_embedding(premise, hypothesis):
        inputs = entail_tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(primary_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = entail_model(**inputs)
        hidden_states = outputs.hidden_states[-1].squeeze(0)
        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"].squeeze(0)
            num_prem_tokens = (token_type_ids == 0).sum().item()
        else:
            sep_id = entail_tokenizer.sep_token_id
            sep_index = (inputs["input_ids"] == sep_id).nonzero(as_tuple=True)[1]
            num_prem_tokens = sep_index[0].item() if len(sep_index) > 0 else hidden_states.size(0) // 2
        prem_vec = hidden_states[:num_prem_tokens].mean(dim=0)
        hypo_vec = hidden_states[num_prem_tokens:].mean(dim=0)
        return torch.stack([prem_vec.cpu(), hypo_vec.cpu()])

    def bidirectional_embedding(p, h):
        return torch.stack([get_embedding(p, h), get_embedding(h, p)])

    def structured_mean_concat(embeddings):
        # DO NOT CHANGE THIS FUNCTION
        all_embs = torch.stack(embeddings)
        fp = all_embs[:, 0, 0, :].mean(dim=0)
        fh = all_embs[:, 0, 1, :].mean(dim=0)
        bp = all_embs[:, 1, 0, :].mean(dim=0)
        bh = all_embs[:, 1, 1, :].mean(dim=0)
        pooled = lambda t: torch.nn.functional.avg_pool1d(t.unsqueeze(0).unsqueeze(0), kernel_size=4).squeeze()
        fp_pool, fh_pool = pooled(fp), pooled(fh)
        bp_pool, bh_pool = pooled(bp), pooled(bh)
        return torch.cat([fp_pool, fh_pool, bp_pool, bh_pool], dim=0)

    # ------------------------------------------------------------------
    # BATCH GENERATION WITH MISTRAL
    # ------------------------------------------------------------------
    responses = []
    queries_list = to_process_df["query"].tolist()
    total_batches = (len(queries_list) + batch_size - 1) // batch_size
    batch_counter = 0

    for start_idx in range(0, len(queries_list), batch_size):
        batch_queries = queries_list[start_idx : start_idx + batch_size]
        print(
            f"Processing batch {batch_counter+1}/{total_batches} "
            f"(processed: {len(responses)}, remaining: {len(queries_list) - len(responses)})",
            flush=True
        )
        batch_counter += 1

        chat_prompts = [build_chat_prompt({"query": q}) for q in batch_queries]
        input_ids_list = []
        attention_masks_list = []
        for prompt in chat_prompts:
            encoded = mistral_tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=False,
                truncation=True
            )
            enc_ids = encoded.squeeze(0)
            input_ids_list.append(enc_ids)
            attn_mask = (enc_ids != mistral_tokenizer.pad_token_id).long()
            attention_masks_list.append(attn_mask)

        reversed_ids = [ids.flip(0) for ids in input_ids_list]
        reversed_attn = [mask.flip(0) for mask in attention_masks_list]
        padded_ids = pad_sequence(reversed_ids, batch_first=True, padding_value=mistral_tokenizer.pad_token_id).flip(1)
        padded_attn = pad_sequence(reversed_attn, batch_first=True, padding_value=0).flip(1)

        with torch.no_grad():
            output_ids = mistral_model.generate(
            input_ids=padded_ids.to(primary_device),
            attention_mask=padded_attn.to(primary_device),
            max_new_tokens=110,
            do_sample=False,
            temperature=0.7,
            # top_p=0.9,
            pad_token_id=mistral_tokenizer.eos_token_id,
        )

        decoded_batch = mistral_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for resp in decoded_batch:
            answer = re.split(r"\[/INST\]", resp)[-1].strip()
            responses.append(answer)

    # Store raw model responses
    to_process_df["response_Cenliea_plus"] = responses
    to_process_df["tuples"] = to_process_df.apply(extract_tuples, axis=1)
    to_process_df["tuple_count"] = to_process_df["tuples"].apply(len)

    # ------------------------------------------------------------------
    # FOR ROWS WITH TUPLES, CREATE EMBEDDINGS
    # AND SET inference_mode = "mistral_augmented"
    # ------------------------------------------------------------------
    valid_df = to_process_df[to_process_df["tuple_count"] > 0].copy()
    if len(valid_df) > 0:
        all_embeddings = []
        for tset in valid_df["tuples"]:
            tuple_embs = [bidirectional_embedding(prem, hypo) for prem, hypo in tset]
            all_embeddings.append(tuple_embs)

        final_vectors = torch.stack([structured_mean_concat(embs) for embs in all_embeddings])
        # Store these vectors so you can inspect or reuse them later
        valid_df["Cenliea_plus_vectors"] = final_vectors.numpy().tolist()

        # We set inference_mode because we know these samples used Mistral for vectorization
        valid_df["inference_mode"] = "mistral_augmented"

        # Update
        to_process_df.update(valid_df)

    # For rows with no tuples, keep the fallback
    to_process_df.loc[to_process_df["tuple_count"] == 0, "inference_mode"] = "fallback_to_nli"

    # === Optional: Replace fallback_to_nli vectors using external CSV file containing NLI-only vectors ===
    if args.nli_only:
        print("Loading fallback NLI-only vectors from:", args.nli_only)
        df_nli_only = pd.read_csv(args.nli_only, converters={vec_col: lambda s: np.array(json.loads(s), dtype=np.float32)})
        fallback_idx = to_process_df[to_process_df["inference_mode"] == "fallback_to_nli"].index

        print(f"Attempting to correct {len(fallback_idx)} fallback vectors...")

        # Check entity consistency for first 3 fallback samples
        for i, idx in enumerate(fallback_idx[:3]):
            e1, e2 = to_process_df.loc[idx, "entities"], df_nli_only.loc[idx, "entities"]
            print(f"  Sample {i+1} (index {idx})")
            print(f"    input : {e1}")
            print(f"    nli_only : {e2}")
            if e1 != e2:
                print("    ⚠️ WARNING: Entity mismatch detected!")

        # Replace the fallback vectors
        to_process_df.loc[fallback_idx, "Cenliea_plus_vectors"] = df_nli_only.loc[fallback_idx, vec_col]
        print(f"✅ Replaced Cenliea_plus_vectors for {len(fallback_idx)} fallback_to_nli samples.")
    else:
        print(f"No external NLI-only file provided. Using internal {vec_col} values.")
        fallback_idx = to_process_df[to_process_df["inference_mode"] == "fallback_to_nli"].index
        to_process_df.loc[fallback_idx, "Cenliea_plus_vectors"] = df.loc[fallback_idx, vec_col]
        print(f"Replaced Cenliea_plus_vectors for {len(fallback_idx)} fallback_to_nli samples from internal column.")

    # RECOMBINE
    for col in [
        "response_Cenliea_plus",
        "tuples",
        "tuple_count",
        "Cenliea_plus_vectors",
        "inference_mode"
    ]:
        if col in to_process_df.columns:
            df.loc[to_process_df.index, col] = to_process_df[col]

    # Adjust the output filename based on max_samples
    if args.output:
        if args.max_samples:
            base, ext = os.path.splitext(args.output)
            output_path = f"{base}_{args.max_samples}{ext}"
        else:
            output_path = args.output
    else:
        output_path = f"{args.split}_Cenliea_plus_response"
        output_path += f"_{args.max_samples}.csv" if args.max_samples else ".csv"

    df.to_csv(output_path, index=False)
    print(f"Saved updated dataset to {output_path}")

    # SUMMARY
    m_aug = (df["inference_mode"] == "mistral_augmented").sum()
    m_fallback = (df["inference_mode"] == "fallback_to_nli").sum()
    avg_tuples = df[df["inference_mode"] == "mistral_augmented"]["tuple_count"].mean()
    avg_tuples = 0.0 if pd.isna(avg_tuples) else avg_tuples

    print("\n--- Inference Mode Summary ---")
    print(f"Mistral-augmented: {m_aug}")
    print(f"Fallback-to-NLI: {m_fallback}")
    print(f"Average number of tuples (mistral_augmented): {avg_tuples:.2f}")


if __name__ == "__main__":
    main()
