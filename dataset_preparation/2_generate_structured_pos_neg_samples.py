# 2_generate_structured_pos_neg_samples.py
# ----------------------------------------
# This script constructs a structured dataset for entity alignment.
# It includes both positive (from ground truth) and hard negative pairs (via GPT-token similarity),
# and formats entity features into a prompt-friendly JSON structure for downstream NLI models.

#In this file, we remove the json escape characters
import sklearn
from sklearn.model_selection import train_test_split
import os
from random import sample
import random
random.seed(8)

import pickle
import re
from Param import *
import string
import pandas as pd
import time
from datetime import timedelta

import sys
import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken  # Make sure to install this package via `pip install tiktoken`
from sklearn.utils import shuffle  # for shuffling

DICT_NAME = str(time.strftime("%Y%m%d%H%M%S"))
# Check if the directory already exists
if not os.path.exists(INPUT_PATH):
    os.makedirs(INPUT_PATH)
start_time = time.monotonic()

def attr_val_dict():
    """Creates a dictionary file of all entities in each 2 source and target that "contains
    all the text attribute values.
    """
    list_of_dict = []
    list_rel_dict = []

    fnames = [INPUT_PATH+'kg1_triples', INPUT_PATH+'kg2_triples']
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
    for fname in fnames:
        desc_dict = {}
        rel_dict = {}
        with open(fname, "r") as f:
            for line in f.readlines():
                line = line.lower()
                h, r, t = line.rstrip('\n').split(' ',2)
                h = h.strip('<>').lower()
                r = r.strip('<>').split('/')[-1]
                r = r.translate(translator)

                if t.startswith('<') or t.startswith('_'):
                    t = t.strip('<>')
                    t = t.strip('> .').lower()
                    if h not in rel_dict:
                        rel_dict[h] = {t:r}
                    else:
                        rel_dict[h][t] = r
                else:
                    t = t.strip(' .')
                    t = re.sub(r'\\([\t\"\\/bfnrt])', r'\1', t)

                    t = re.findall('\"(.+)\"', t)
                    if t:
                        t = t[0]

                        if h in desc_dict:
                            if len(desc_dict[h].keys())<MAX_ATTR_NO:
                                if r not in desc_dict[h]:
                                    if len(t) < MAX_ATTR_VAL_LEN:
                                        desc_dict[h][r] = t
                                    else:
                                        desc_dict[h][r] = t[:MAX_ATTR_VAL_LEN-1] #+'"'
                        else:
                            if len(t) < MAX_ATTR_VAL_LEN:
                                desc_dict[h] = {r: t}
                            else:
                                desc_dict[h] = {r: t[:MAX_ATTR_VAL_LEN-1]} #+'"'}
        decoded_json = json.dumps(desc_dict, ensure_ascii=False)
        desc_dict = json.loads(decoded_json)

        decoded_json = json.dumps(rel_dict, ensure_ascii=False)
        rel_dict = json.loads(decoded_json)

        list_of_dict.append(desc_dict)
        list_rel_dict.append(rel_dict)

    list_not_blank_dict = [] #we keep the attribute-values of not blank nodes here
    for dic in list_of_dict:
        not_blank = {}
        for k in dic.keys():
            if not k.startswith("_"):
                not_blank[k] = dic[k]
        list_not_blank_dict.append(not_blank)

    print("Length of direct attribute-value dictionary in KGs (including blanks):", len(list_of_dict[0]), len(list_of_dict[1]))
    print("Length of direct attribute-value dictionary in KGs (not including blanks):", len(list_not_blank_dict[0]), len(list_not_blank_dict[1]))

    #add 1-hop neighbor's attribute-values
    ind = 0
    list_of_dict_1hop = []
    for kg in list_not_blank_dict:
        dict_1hop = {}
        kg1_rels = list_rel_dict[ind]
        kg1_attrs = list_of_dict[ind]
        for k in kg.keys():
            dict_1hop[k] = {}
            if k in kg1_rels.keys(): #if entity has some relation properties
                for tail in kg1_rels[k].keys():#for each tail of entity
                    rel = kg1_rels[k][tail]
                    if tail in kg1_attrs.keys() and len(dict_1hop[k].keys())<MAX_REL_NO:#if tail has any attr-values
                        #Add attr-values' dict of the tail, with relation type as key
                        dict_1hop[k][rel] = kg1_attrs[tail]
        #Check if there are not-blank entities that don't have direct attr-values but
        #their 1-hop neighbors have some attribute-values
        for ent in kg1_rels:
            if not ent.startswith('_'):
                flag = 0 #becomes one if there is at least one 1-hop neigh having attr-values
                for tail in kg1_rels[ent]:
                    rel = kg1_rels[ent][tail]
                    if tail in kg1_attrs:
                        if ent not in dict_1hop:
                            flag = 1
                            dict_1hop[ent]= {rel: kg1_attrs[tail]}
                        else:
                            if len(dict_1hop[ent].keys())<MAX_REL_NO:
                                flag = 1
                                dict_1hop[ent][rel] = kg1_attrs[tail]
                if flag == 0:
                    for tail in kg1_rels[ent]:
                        rel = kg1_rels[ent][tail]
                        if ent not in dict_1hop:
                            dict_1hop[ent]= {rel: None}
                        else:
                            if len(dict_1hop[ent].keys())<MAX_REL_NO:
                                dict_1hop[ent][rel] = None

        decoded_json = json.dumps(dict_1hop, ensure_ascii=False)
        dict_1hop = json.loads(decoded_json)
        ind +=1
        list_of_dict_1hop.append(dict_1hop)

    #Add missing entities to the first list
    inde = 0
    for dic in list_of_dict_1hop:
        for k in dic:
            if k not in list_not_blank_dict[inde]:
                list_not_blank_dict[inde][k] = {}
        inde +=1


    print("Length of final self attribute-value dictionary in KGs (ignoring blanks):", len(list_not_blank_dict[0]), len(list_not_blank_dict[1]))

    print("\n*********Self attribute values*********\n")
    print("\n***************** KG1 *****************\n")
    #Add numerical indicator to the attributes of entities in KG1
    for k in list_not_blank_dict[0].keys():
        ind = 1
        if list_not_blank_dict[0][k]:
            att_keys = list(list_not_blank_dict[0][k].keys())
            for att in att_keys:
                list_not_blank_dict[0][k][str(ind)+'. '+att] = list_not_blank_dict[0][k].pop(att)
                ind +=1
    #print
    count = 0
    keys = list(list_not_blank_dict[0].keys())
    keys.sort()
    for k in keys:
        if count <10:
            print(k, list_not_blank_dict[0][k])
            count +=1

    print("\n**********************************\n")
    print("\n***************** KG2 *****************\n")
    alphabet = list(string.ascii_lowercase)

    #Add alphabetical indicator to the attributes of entities in KG2
    for k in list_not_blank_dict[1].keys():
        ind = 0
        if list_not_blank_dict[1][k]:
            att_keys = list(list_not_blank_dict[1][k].keys())
            for att in att_keys:
                list_not_blank_dict[1][k][alphabet[ind]+'. '+att] = list_not_blank_dict[1][k].pop(att)
                ind +=1
    #print
    count = 0
    keys = list(list_not_blank_dict[1].keys())
    keys.sort() #To print in the same order in each run
    for k in keys:
        if count <10:
            print(k, list_not_blank_dict[1][k])
            count +=1

    #Save the self attribute values dictionaries of KG1 and KG2
    with open(INPUT_PATH+DICT_NAME+'_0', 'wb') as handle:
        pickle.dump(list_not_blank_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("list_not_blank_dict (self attribute values) {}_0 was created!".format(DICT_NAME))


    print("\n**********************************\n")
    print("Length of 1-hop attribute-value dictionary in KGs:", len(list_of_dict_1hop[0]), len(list_of_dict_1hop[1]))
    print("Length of Self attribute-value dictionary in KGs:", len(list_not_blank_dict[0]), len(list_not_blank_dict[1]))

    print("\n********* 1-hop attribute values *********\n")
    print("\n***************** KG1 *****************\n")
    #Add numerical indicator to the direct relations and attributes of tail entities in KG1
    for k in list_of_dict_1hop[0].keys():
        ind = 1
        if list_of_dict_1hop[0][k]:#if entity has any relations
            rel_keys = list(list_of_dict_1hop[0][k].keys())
            for rel in rel_keys:#indexing the relations
                list_of_dict_1hop[0][k][str(ind)+'. '+rel] = list_of_dict_1hop[0][k].pop(rel)
                ind +=1
            rel_keys = list(list_of_dict_1hop[0][k].keys())
            for rel in rel_keys:
                ind = 1
                if list_of_dict_1hop[0][k][rel]:#if the tail entity has any attributes
                    att_keys = list(list_of_dict_1hop[0][k][rel].keys())
                    for att in att_keys:
                        list_of_dict_1hop[0][k][rel][str(ind)+'. '+att] = list_of_dict_1hop[0][k][rel].pop(att)
                        ind +=1
    #print
    count = 0
    keys = list(list_of_dict_1hop[0].keys())
    keys.sort()
    for k in keys:
        if count <10:
            print(k, list_of_dict_1hop[0][k])
            count +=1

    print("\n**********************************\n")
    print("\n***************** KG2 *****************\n")
    #Add alphabetical indicator to the direct relations and attributes of tail entities in KG2
    for k in list_of_dict_1hop[1].keys():
        ind = 0
        if list_of_dict_1hop[1][k]:#if entity has any relations
            rel_keys = list(list_of_dict_1hop[1][k].keys())
            for rel in rel_keys:#indexing the relations
                list_of_dict_1hop[1][k][alphabet[ind]+'. '+rel] = list_of_dict_1hop[1][k].pop(rel)
                ind +=1
            rel_keys = list(list_of_dict_1hop[1][k].keys())
            for rel in rel_keys:
                ind = 1
                if list_of_dict_1hop[1][k][rel]:#if the tail entity has any attributes
                    att_keys = list(list_of_dict_1hop[1][k][rel].keys())
                    for att in att_keys:
                        list_of_dict_1hop[1][k][rel][str(ind)+'. '+att] = list_of_dict_1hop[1][k][rel].pop(att)
                        ind +=1
    #print
    count = 0
    keys = list(list_of_dict_1hop[1].keys())
    keys.sort() #To print in the same order in each run
    for k in keys:
        if count <10:
            print(k, list_of_dict_1hop[1][k])
            count +=1

    with open(INPUT_PATH+DICT_NAME+'_1', 'wb') as handle:
        pickle.dump(list_of_dict_1hop, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("list_of_dict_1hop {}_1 was created!".format(DICT_NAME))

def get_text_from_dict(diction):
    ent_str = ""
    #print(diction)
    #if diction.keys():
    for k, v in diction.items():
        if v:
            ent_str += k + ": "+str(v) +",\n"
        else:
            ent_str += k + ": None,\n"

    return ent_str

# Initialize the encoder (using the "cl100k_base" model, which is common for GPT-3 and GPT-4)
encoder = tiktoken.get_encoding("cl100k_base")

# Function to encode a dictionary of texts into token embeddings
def encode_texts(text_dict):
    # Tokenize each text and return a dictionary with token ids for each text
    return {key: encoder.encode(text) for key, text in text_dict.items()}

# Function to calculate the cosine similarity between tokenized texts
def calculate_similarity(dict1_encodings, dict2_encodings):
    # Convert dict_values to lists and then concatenate them
    all_encodings = list(dict1_encodings.values()) + list(dict2_encodings.values())

    # Get unique tokens to create binary bag-of-tokens vectors
    unique_tokens = set(token for encoding in all_encodings for token in encoding)
    print("Unique tokens in the encoded head enitities (KG1 ground truth head entities), and selected tail entities (in KG2 for negative sampling):", len(unique_tokens))
    token_to_index = {token: i for i, token in enumerate(unique_tokens)}

    def vectorize(tokens):
        vector = np.zeros(len(unique_tokens))
        for token in tokens:
            vector[token_to_index[token]] += 1
        return vector

    # Create vectors for each text in dict1 and dict2
    dict1_vectors = np.array([vectorize(encoding) for encoding in dict1_encodings.values()])
    dict2_vectors = np.array([vectorize(encoding) for encoding in dict2_encodings.values()])
    # Calculate cosine similarities between each pair of vectors from dict1 and dict2
    similarities = cosine_similarity(dict1_vectors, dict2_vectors)
    return similarities


def positive_sample():
    print('*'*100)
    print("Start formatting the positive samples...\n\n")
    #Create Positive samples file
    list_not_blank_dict = pickle.load(open(INPUT_PATH+DICT_NAME+'_0', "rb"), encoding='utf-8')
    list_of_dict_1hop = pickle.load(open(INPUT_PATH+DICT_NAME+'_1', "rb"))
    keys=["entities", "self-attr-val", "1hop-attr-val", "label", "GPT-similarity-score", "dataset"]
    final_list = []
    with open(INPUT_PATH+"same_as", "r") as f:
        i = 0
        dict1 = {}
        dict2 = {}
        for line in f.readlines():
            line = line.lower()
            ent1, ent2 = line.rstrip('\n').strip().split()

            if ent1 not in list_not_blank_dict[0]: # and ent1 not in list_not_blank_dict[1]:
                print("Blank node:", ent1)
                continue
            if ent2 not in list_not_blank_dict[1]: # and ent2 not in list_not_blank_dict[1]:
                print("Blank node:", ent2)
                continue
            if ent1 in dict1 or ent2 in dict2:
                print("repeated head or tail entities in the same_as file:", ent1, ent2)
                continue

            ent1_attr = list_not_blank_dict[0][str(ent1)]
            ent1_rel = list_of_dict_1hop[0][ent1]
            ent2_attr = list_not_blank_dict[1][ent2]
            ent2_rel = list_of_dict_1hop[1][ent2]
            ents = {"entity1": ent1,
            "entity2": ent2}
            attrs = {"entity1": ent1_attr, "entity2": ent2_attr}
            hop1s = {"entity1": ent1_rel, "entity2": ent2_rel}

            ent1_attr_str = get_text_from_dict(ent1_attr)
            ent1_rel_str = get_text_from_dict(ent1_rel)
            ent1_text = "entity's ttribute values:"+ ent1_attr_str + "attribute values of entity's 1-hop neighbors:" + ent1_rel_str
            dict1[ent1] = ent1_text

            ent2_attr_str = get_text_from_dict(ent2_attr)
            ent2_rel_str = get_text_from_dict(ent2_rel)
            ent2_text = "entity's ttribute values:"+ ent2_attr_str + "attribute values of entity's 1-hop neighbors:" + ent2_rel_str
            dict2[ent2] = ent2_text



            row = [{keys[0]: json.dumps(ents, ensure_ascii=False), keys[1]: json.dumps(attrs, ensure_ascii=False), keys[2]: json.dumps(hop1s, ensure_ascii=False), keys[3]: 1, keys[4]: 1.0, keys[5]: DATASET}]
            final_list.extend(row)
            i+=1
    # Encode the texts in both dictionaries
    encoded_dict1 = encode_texts(dict1)
    encoded_dict2 = encode_texts(dict2)

    # Calculate similarities
    print("Starting cosine similarity on", len(dict1), "x", len(dict2))
    sim_score = calculate_similarity(encoded_dict1, encoded_dict2)
    pos_sim_score = []
    print("len(final_list):", len(final_list))
    for i in range(len(final_list)):
        pos_sim_score.append(sim_score[i][i])
    print("\n\n Average similarity score for all positive samples: ", sum(pos_sim_score)/len(pos_sim_score))
    for i in range(len(final_list)):
        final_list[i][keys[4]] = pos_sim_score[i]


    df = pd.DataFrame.from_records(final_list)

    queries = df.apply(make_prompt, axis=1)
    df["query"] = queries

    df.to_json(INPUT_PATH+"pos_samples.jsonl", orient="records", lines=True)
    print(df.info())
    print('*'*100)
    return df


def shuffle_triple_file(fnames):
    for fname in fnames:
        with open(fname, "r") as f:
            triples_list = f.readlines()
            random.shuffle(triples_list) #shuffle the triples

        with open(fname, "w") as f:
            f.writelines(triples_list)


def negative_sample(ent_pairs_df, top_k):
    #Create Negative samples file using negative sampling strategy of finding most difficult negative pairs i.e.
    #pairs that are mostly similar but those are not the same
    print("Start negative sampling...\n\n")
    list_not_blank_dict = pickle.load(open(INPUT_PATH+DICT_NAME+'_0', "rb"))
    list_of_dict_1hop = pickle.load(open(INPUT_PATH+DICT_NAME+'_1', "rb"))
    ent_pairs = ent_pairs_df

    keys = ent_pairs.keys()
    print("keys: ", keys)

    #Prepare two dicts of entities' texts, first list contains all head entities in the ground truth,
    #Second list contains all entities in KG2. We consider a text like this for each entity:
    # entity's ttribute values: ...., attribute values of entity's 1-hop neighbors: ... .
    dict1 = {}
    dict2 = {}

    matched_ents_tup = set()
    ent2_aligned_uri = []
    for ents in list(ent_pairs["entities"]):
        ents = json.loads(ents)
        ent1 = ents["entity1"]
        ent2 = ents["entity2"]
        ent2_aligned_uri.append(ent2)
        matched_ents_tup.add((ent1.strip().lower(), ent2.strip().lower()))
        ent1_attr = list_not_blank_dict[0][ent1]
        ent1_attr_str = get_text_from_dict(ent1_attr)

        ent1_rel = list_of_dict_1hop[0][ent1]
        ent1_rel_str = get_text_from_dict(ent1_rel)

        ent_text = "entity's ttribute values:"+ ent1_attr_str + "attribute values of entity's 1-hop neighbors:" + ent1_rel_str
        dict1[ent1] = ent_text

    for ent2 in ent2_aligned_uri:
        ent2_attr = list_not_blank_dict[1][ent2]
        ent2_attr_str = get_text_from_dict(ent2_attr)

        ent2_rel = list_of_dict_1hop[1][ent2]
        ent2_rel_str = get_text_from_dict(ent2_rel)

        ent_text = "entity's ttribute values:"+ ent2_attr_str + "attribute values of entity's 1-hop neighbors:" + ent2_rel_str
        dict2[ent2] = ent_text
    count = len(ent2_aligned_uri)
    max_candids = MAX_CANDID
    print("\n\n Max number of candidates, kg2 number of entities:",max_candids, len(list_not_blank_dict[1]))
    if len(list_not_blank_dict[1]) > max_candids: #if max number of candidates are less than the total number of entities in KG2
        print("sampling over the kg2 due to the large size of KGs...")
        while count <= max_candids: #to avoid OOM killed error, we consider a max number of candidates
            ent2 = random.choice(list(list_not_blank_dict[1].keys()))
            if ent2 not in dict2: #to avoid repetative negative pairs
                ent2_attr = list_not_blank_dict[1][ent2]
                ent2_attr_str = get_text_from_dict(ent2_attr)

                ent2_rel = list_of_dict_1hop[1][ent2]
                ent2_rel_str = get_text_from_dict(ent2_rel)

                ent_text = "entity's ttribute values:"+ ent2_attr_str + "attribute values of entity's 1-hop neighbors:" + ent2_rel_str
                dict2[ent2] = ent_text
                count += 1

    else: #if the target graph is small enough
        max_candids =  len(list_not_blank_dict[1])
        for ent2 in list_not_blank_dict[1].keys():
            if ent2 not in dict2: #to avoid repetative negative pairs
                ent2_attr = list_not_blank_dict[1][ent2]
                ent2_attr_str = get_text_from_dict(ent2_attr)

                ent2_rel = list_of_dict_1hop[1][ent2]
                ent2_rel_str = get_text_from_dict(ent2_rel)

                ent_text = "entity's ttribute values:"+ ent2_attr_str + "attribute values of entity's 1-hop neighbors:" + ent2_rel_str
                dict2[ent2] = ent_text
                count += 1

    print("\n Number of candidates for nagtive sampling:", len(dict2))

    # Encode the texts in both dictionaries
    encoded_dict1 = encode_texts(dict1)
    encoded_dict2 = encode_texts(dict2)

    # Calculate similarities
    similarity_matrix = calculate_similarity(encoded_dict1, encoded_dict2)

    # Mapping index to keys for retrieving top matches with identifiers
    dict1_keys = list(encoded_dict1.keys())
    dict2_keys = list(encoded_dict2.keys())

    # Find the top k most similar texts in dict2 for each text in dict1

    top_matches = {}
    for i, similarities in enumerate(similarity_matrix):
        top_indices = np.argsort(similarities)[::-1]  # Get top k indices, sorted by similarity  [-(top_k+1):][::-1]
        dict1_key = dict1_keys[i]
        top_matches[dict1_key] = [(dict2_keys[j], similarities[j]) for j in top_indices]


    neg_pairs = []
    neg_pairs_imb = []

    for m in matched_ents_tup:
        print("matched_ents_tup: ",m, type(m))
        m1 = m
        break


    total_negatives_added = 0  # count total added negatives
    for k, v in top_matches.items(): #each value in top_matches is a list of length l, each entry of the list is a tuple like ('http://identifiers.org/oryzabase.gene/8714', 0.8498)
        ind_count = 0
        match_count = 0
        ind_count = 0
        flag = 0
        while match_count < top_k:
            if (k.strip().lower(), v[ind_count][0].strip().lower()) not in matched_ents_tup:
                if flag ==0:
                    neg_pairs.append((k, v[ind_count])) #Adding the top similar in a list containing k similar entities
                    neg_pairs_imb.append((k, v[ind_count]))
                    match_count +=1
                    flag = 1
                    ind_count +=1
                else:
                    neg_pairs_imb.append((k, v[ind_count])) #Adding the top similar in a list containing k similar entities
                    match_count +=1
                    ind_count +=1
            else:
                ind_count +=1
        total_negatives_added += 1
        if total_negatives_added % 1000 == 0:
            print(f"🟢 Added {total_negatives_added} negative samples so far...")


    keys=["entities", "self-attr-val", "1hop-attr-val", "label", "GPT-similarity-score", "dataset"]

    final_list = []
    for i in range(len(neg_pairs)):
        ent1, ent2, sim_score = neg_pairs[i][0], neg_pairs[i][1][0], round(neg_pairs[i][1][1],5) #each entry of neg_pairs in like (ent1, (ent2, sim_score))
        ents = {"entity1": ent1,
        "entity2": ent2}
        ent1_attr = list_not_blank_dict[0][ent1]
        ent1_rel = list_of_dict_1hop[0][ent1]
        ent2_attr = list_not_blank_dict[1][ent2]
        ent2_rel = list_of_dict_1hop[1][ent2]
        attrs = {"entity1": ent1_attr, "entity2": ent2_attr}
        hop1s = {"entity1": ent1_rel, "entity2": ent2_rel}
        row = [{keys[0]: json.dumps(ents, ensure_ascii=False), keys[1]: json.dumps(attrs, ensure_ascii=False), keys[2]: json.dumps(hop1s, ensure_ascii=False), keys[3]: 0, keys[4]: sim_score, keys[5]: DATASET}]
        final_list.extend(row)

    df = pd.DataFrame.from_records(final_list)
    queries = df.apply(make_prompt, axis=1)
    df["query"] = queries

    final_list_imb = []
    for i in range(len(neg_pairs_imb)):
        ent1, ent2, sim_score = neg_pairs_imb[i][0], neg_pairs_imb[i][1][0], round(neg_pairs_imb[i][1][1],5) #each entry of neg_pairs_imb in like (ent1, (ent2, sim_score))
        ents = {"entity1": ent1,
        "entity2": ent2}
        ent1_attr = list_not_blank_dict[0][ent1]
        ent1_rel = list_of_dict_1hop[0][ent1]
        ent2_attr = list_not_blank_dict[1][ent2]
        ent2_rel = list_of_dict_1hop[1][ent2]
        attrs = {"entity1": ent1_attr, "entity2": ent2_attr}
        hop1s = {"entity1": ent1_rel, "entity2": ent2_rel}
        row = [{keys[0]: json.dumps(ents, ensure_ascii=False), keys[1]: json.dumps(attrs, ensure_ascii=False), keys[2]: json.dumps(hop1s, ensure_ascii=False), keys[3]: 0, keys[4]: sim_score, keys[5]: DATASET}]
        final_list_imb.extend(row)

    df_imb = pd.DataFrame.from_records(final_list_imb)
    queries = df_imb.apply(make_prompt, axis=1)
    df_imb["query"] = queries


    print('*'*100)
    return df, df_imb

def make_prompt(sample): #ref_sim_model=ref_sim_model):

  p2 = "\nEntity 1 direct features:\n"
  p3 = "\nEntity 2 direct features:\n"
  p4 = "\nEntity 1 neighbor's features:\n"
  p5 = "\nEntity 2 neighbor's features:\n"
  #p6 = """\nUse numerical-alphabetical enumerators given in the prompt (don't generate new enumerators) and give me tuples of features of entity 1 & 2 e.g. (2, c) which represent their salient similarities:\n"""
  def self_attr(ent_dict):
    out_str = ""
    for k, v in ent_dict.items():
      out_str += k + ": "+v.replace("\\", "") +",\n"
    return out_str

  def neig_attr(ent_dict):
    out_str = ""
    for rel, attr_dict in ent_dict.items():
      rel_ind = rel.split()[0]
      rel_label = rel.replace(rel_ind,"")
      if attr_dict:
        for attr, v in attr_dict.items():
          attr_ind = attr.split()[0]
          attr_name = attr.replace(attr_ind,"")
          out_str += "R-"+rel_ind+attr_ind+" "+ rel_label+ " -> " \
          +attr_name+ ": "+v.replace("\\", "") +",\n"
      else:
        out_str += "R-"+ rel+ ",\n"
    return out_str

  keys = list(sample.keys())
  prompt = p2+ self_attr(json.loads(sample["self-attr-val"])["entity1"]) \
                  +p3+self_attr(json.loads(sample["self-attr-val"])["entity2"])+ p4 \
                  + neig_attr(json.loads(sample["1hop-attr-val"])["entity1"]) + p5 \
                  + neig_attr(json.loads(sample["1hop-attr-val"])["entity2"])

  return prompt

def concat_dfs(df1, df2):
    df_mix =  pd.concat([df1, df2])
    return df_mix


if __name__ == '__main__':
    print("----------------create Direct attr-values dictionary-------------\n")
    fnames = [INPUT_PATH+'kg1_triples', INPUT_PATH+'kg2_triples']
    shuffle_triple_file(fnames)
    attr_val_dict()

    pos_df = positive_sample()
    df_train_pos, df_test_pos = train_test_split(pos_df, test_size=TEST, shuffle=False) #We shuffled the data after
    df_train_pos = df_train_pos.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    df_test_pos = df_test_pos.applymap(lambda x: x.lower() if isinstance(x, str) else x)


    print("Generating negative samples for train set...\n")
    df_train_neg, _ = negative_sample(df_train_pos, top_k = NEG_SAMPLES)
    print(f"Generating negative samples (1 negatives per positive) and imbalanced negative samples ({NEG_SAMPLES} negatives per positive) for test set...\n")
    df_test_neg, df_test_neg_imb = negative_sample(df_test_pos, top_k = NEG_SAMPLES) #balanced and imbalanced test set

    fname = 'test_pos'
    df_test_pos.to_json(INPUT_PATH+f"{fname}.json", orient="records")
    df_test_pos.to_csv(INPUT_PATH+fname+'.csv', index=False)
    print("{}.csv and {}.json was created!".format(fname, fname))

    fname = 'train_pos'
    df_train_pos.to_json(INPUT_PATH+f"{fname}.json", orient="records")
    df_train_pos.to_csv(INPUT_PATH+fname+'.csv', index=False)
    print("{}.csv and {}.json was created!".format(fname, fname))

    fname = 'test_neg'
    df_test_neg.to_json(INPUT_PATH+f"{fname}.json", orient="records")
    df_test_neg.to_csv(INPUT_PATH+fname+'.csv', index=False)
    print("{}.csv and {}.json was created!".format(fname, fname))

    fname = 'train_neg'
    df_train_neg.to_json(INPUT_PATH+f"{fname}.json", orient="records")
    df_train_neg.to_csv(INPUT_PATH+fname+'.csv', index=False)
    print("{}.csv and {}.json was created!".format(fname, fname))

    fname ='test_neg_imb'
    df_test_neg_imb.to_json(INPUT_PATH+f"{fname}.json", orient="records")
    df_test_neg_imb.to_csv(INPUT_PATH+fname+'.csv', index=False)
    print("{}.csv and {}.json was created!".format(fname, fname))

    df_train_mix =  concat_dfs(df_train_pos, df_train_neg)
    df_train_mix = shuffle(df_train_mix, random_state=8)
    df_train_mix.to_json(INPUT_PATH+'/'+"train.json", orient="records")

    df_test_mix =  concat_dfs(df_test_pos, df_test_neg)
    df_test_mix = shuffle(df_test_mix, random_state=8)
    df_test_mix.to_json(INPUT_PATH+'/'+"test.json", orient="records")

    df_test_mix_imb =  concat_dfs(df_test_pos, df_test_neg_imb)
    df_test_mix_imb = shuffle(df_test_mix_imb, random_state=8)
    df_test_mix_imb.to_json(INPUT_PATH+'/'+"test_imb.json", orient="records")
