import os
import glob

from random import sample
import random
import json
import re
from Param import *
import pandas as pd
import time
import sys
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, load_from_disk, interleave_datasets, concatenate_datasets
import shutil

random_state = 8
random.seed(random_state)

start_time = time.time()


def integ_prompt_datasets(sample_no = SHARD_SIZE):
    dataset_names = os.listdir(ALL_DATA_PATH)
    #dataset_names = ['dor', 'spimbench']
    print(dataset_names)

    #Adding first dataset
    print(f"Adding dataset {dataset_names[0]}...")
    with open(ALL_DATA_PATH+dataset_names[0]+"/train.json") as train_file:
        jsonObj = json.load(train_file)
        df_train = pd.DataFrame(jsonObj)
    with open(ALL_DATA_PATH+dataset_names[0]+"/test.json") as test_file:
        jsonObj = json.load(test_file)
        df_test = pd.DataFrame(jsonObj)
    with open(ALL_DATA_PATH+dataset_names[0]+"/test_imb.json") as test_file:
        jsonObj = json.load(test_file)
        df_test_imb = pd.DataFrame(jsonObj)

    mixed_dataset_path = MIXED_DATASET_PATH
    if not os.path.exists(mixed_dataset_path):
        os.makedirs(mixed_dataset_path)

    print(MIXED_DATA_CONFIG)
    if MIXED_DATA_CONFIG:
        #Adding other datasets
        for d in dataset_names[1:]:
            print(f"Adding dataset {d}...")
            with open(ALL_DATA_PATH+d+"/train.json") as train_file:
                jsonObj = json.load(train_file)
                df_train = pd.concat([df_train, pd.DataFrame(jsonObj)], ignore_index=True)
            with open(ALL_DATA_PATH+d+"/test.json") as test_file:
                jsonObj = json.load(test_file)
                df_test = pd.concat([df_test, pd.DataFrame(jsonObj)], ignore_index=True)
            #For imbalanced test set we should find test_imb.json files
            with open(ALL_DATA_PATH+d+"/test_imb.json") as test_file:
                jsonObj = json.load(test_file)
                df_test_imb = pd.concat([df_test_imb, pd.DataFrame(jsonObj)], ignore_index=True)

    df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True) #shuffle the rows
    df_test = df_test.sample(frac=1, random_state=random_state).reset_index(drop=True) #shuffle the rows
    df_test_imb = df_test_imb.sample(frac=1, random_state=random_state).reset_index(drop=True) #shuffle the rows
    print("Number of train and test and imbalanced test samples:", len(df_train), len(df_test), len(df_test_imb))

    if MIXED_DATA_CONFIG:
        dfs = [df_train, df_test, df_test_imb]
        splits = ['train', 'test', 'test_imb']
        no_shard = len(df_train)//sample_no
        test_slice = len(df_test)//no_shard
        slices = [sample_no, test_slice, sample_no]
        splits_shard_no = [no_shard, no_shard, len(df_test_imb)//sample_no]
    else:
        df_test = pd.concat([df_train, df_test])
        df_test_imb = pd.concat([df_train, df_test_imb])
        dfs =  [df_test, df_test_imb]
        splits = ['test', 'test_imb']
        no_shard = len(dfs[0])//sample_no
        slices = [sample_no, sample_no]
        splits_shard_no = [no_shard, len(df_test_imb)//sample_no]

    for df, split, ind, no_shard in zip(dfs, splits, slices, splits_shard_no):
        count = 0
        fdir = MIXED_DATASET_PATH+'/'+split
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        for count in range(no_shard):
            df.iloc[count*ind:count*ind+ind, :].to_json(fdir+"/shard"+str(count)+".json", orient="records")
            count +=1

        print("number of shards",count)
        print(f"length of the ignored {split} shards: ",len(df.iloc[count*ind:, :]))


if __name__ == '__main__':

    print("----------------create Mixed Dataset-------------\n")
    integ_prompt_datasets(sample_no=SHARD_SIZE,)

    with open('./Param.py', 'r') as f:
        par = f.read()
    with open(MIXED_DATASET_PATH+'/params', 'w') as f:
        f.write(par)
    print("--- Runtime: %s seconds ---" % (time.time() - start_time))
