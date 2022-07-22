import datasets
from fastcore.utils import compose
from clean_funcs import *
import json
import time

import gzip
import hashlib
import multiprocessing
import os
import shutil
import time
from pathlib import Path
import numpy as np
from datasets import load_dataset
from minhash_deduplication import deduplicate_dataset

def get_hash(example):
    """Get hash of content field."""
    return {"hash": hashlib.md5(example["text"].strip().encode("utf-8")).hexdigest()}

def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False

def compress_file(file_path):
    """Compress a file with g-zip."""
    with open(file_path, "rb") as f_in:
        with gzip.open(str(file_path) + ".gz", "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.unlink(file_path)

def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False

def filter(example, uniques):
    """Filter dataset with unique values."""
    if not check_uniques(example, uniques):
        return False
    else:
        return True

name = 'part_4'
id_train_cleaned = datasets.load_dataset('/data/clean_mc4_id/clean_mc4_id.py', name)      

len_train = len(id_train_cleaned['train'])
print(f"Size of original dataset train: {len_train}")

"""
id_train_cleaned_dedup = id_train_cleaned['train'].map(get_hash, num_proc=96, writer_batch_size=100000)

# Deduplicate hashes
uniques = set(id_train_cleaned_dedup.unique("hash"))
frac = len(uniques) / len(id_train_cleaned_dedup)
print(f"Fraction of duplicates: {1-frac:.2%}")

# Deduplicate data

dataset_train_deduplicated = id_train_cleaned_dedup.filter(filter, fn_kwargs={"uniques": uniques})

"""


t_start = time.time()
ds_filter, duplicate_clusters = deduplicate_dataset(id_train_cleaned['train'],0.75)
print(f"Size of original dataset train: {len(id_train_cleaned['train'])}")
print(f"Size of filtered dataset train: {len(ds_filter)}")
print(f"Time to deduplicate dataset: {time.time()-t_start:.2f}")

with open('/data/mc4_id_clean/mc4_id/duplicate_clusters.json', "w") as f:
    json.dump(duplicate_clusters, f)


t_start = time.time()
for file_number, index in enumerate(range(0, len(ds_filter), 500000)):
    file_path = str(f"/data/mc4_id_clean/mc4_id/{name}-{file_number+1:012}.json")
    end_index = min(len(ds_filter), index + 50000)
    ds_filter.select(list(range(index, end_index))).to_json(file_path)
    compress_file(file_path)
print(f"Time to save dataset: {time.time()-t_start:.2f}")