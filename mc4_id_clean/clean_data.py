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
id_mc4_train = datasets.load_dataset("mc4", "id",split='train')
print(id_mc4_train)

data_preprocessing_funcs = compose(*[fix_html, remove_control_char, remove_remaining_control_chars, remove_unicode_symbols,
                                   standardise_punc, remove_news_tags, replace_urls, replace_usernames, remove_duplicate_words_punctuation, remove_multi_space])
data_stats_funcs = compose(*[count_alphabet, count_numbers, count_upper, count_str_len,
                           predict_lang, calculate_alphabet_ratio, calculate_number_ratio, calculate_upper_ratio])

min_alphabet_ratio = 0.75
max_upper_ratio = 0.10
max_number_ratio = 0.05
min_pred_lang_percentage = 0.65
from minhash_deduplication import deduplicate_dataset
import hashlib

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

# TRAIN SPLIT
num_rows = id_mc4_train.num_rows
print(f"Original dataset train rows {num_rows}")
id_mc4_train = id_mc4_train.map(
    data_preprocessing_funcs, num_proc=96, batched=True, writer_batch_size=100000)
id_train_only_longer = id_mc4_train.filter(
    lambda example: len(example['text'].split()) >= 20, num_proc=96)
#id_mc4_train.cleanup_cache_files()
num_rows = id_train_only_longer.num_rows
print(f"Only longer texts dataset train rows {num_rows}")
id_train_only_longer_new = id_train_only_longer.map(
    data_stats_funcs, num_proc=96, batched=False, writer_batch_size=100000)

id_train_cleaned = id_train_only_longer_new.filter(lambda example: example['alphabet_ratio'] > min_alphabet_ratio and example['upper_ratio'] < max_upper_ratio and example[
                                               'number_ratio'] < max_number_ratio and example['predicted_lang'] == '__label__id' and example['predicted_lang_percentage'] > min_pred_lang_percentage, num_proc=96)
#id_train_only_longer.cleanup_cache_files()
num_rows = id_train_cleaned.num_rows
print(f"Final cleaned dataset train rows {num_rows}")
#id_train_cleaned.cleanup_cache_files()
"""
# VAL SPLIT
id_mc4_validation = datasets.load_dataset("mc4", "id",split='validation')
num_rows = id_mc4_validation.num_rows
print(f"Original dataset val rows {num_rows}")
id_mc4_validation = id_mc4_validation.map(
    data_preprocessing_funcs, num_proc=96, batched=True)

id_val_only_longer = id_mc4_validation.filter(
    lambda example: len(example['text'].split()) >= 20, num_proc=96)
num_rows = id_val_only_longer.num_rows
print(f"Only longer texts dataset val rows {num_rows}")

id_val_only_longer = id_val_only_longer.map(
    data_stats_funcs, num_proc=96, batched=False)

id_val_cleaned = id_val_only_longer.filter(lambda example: example['alphabet_ratio'] > min_alphabet_ratio and example['upper_ratio'] < max_upper_ratio and example['number_ratio']
                                           < max_number_ratio and example['predicted_lang'] == '__label__id' and example['predicted_lang_percentage'] > min_pred_lang_percentage, num_proc=96)
num_rows = id_val_cleaned.num_rows
print(f"Final cleaned dataset val rows {num_rows}")
id_val_cleaned = id_val_cleaned.remove_columns(["alphabet_len", "number_len", "upper_len", "total_len", "predicted_lang", "predicted_lang_percentage", "alphabet_ratio", "number_ratio", "upper_ratio"])
"""

id_train_cleaned = id_train_cleaned.remove_columns(["alphabet_len", "number_len", "upper_len", "total_len", "predicted_lang", "predicted_lang_percentage", "alphabet_ratio", "number_ratio", "upper_ratio"])
id_train_cleaned.to_csv("train.csv", num_proc=96, index=False)

len_train = len(id_train_cleaned)
print(f"Size of original dataset train: {len_train}")

id_train_cleaned_dedup = id_train_cleaned.map(get_hash, num_proc=96, writer_batch_size=100000)

# Deduplicate hashes
uniques = set(id_train_cleaned_dedup.unique("hash"))
frac = len(uniques) / len(id_train_cleaned_dedup)
print(f"Fraction of duplicates: {1-frac:.2%}")

# Deduplicate data

dataset_train_deduplicated = id_train_cleaned_dedup.filter(filter, fn_kwargs={"uniques": uniques})

t_start = time.time()
ds_filter, duplicate_clusters = deduplicate_dataset(dataset_train_deduplicated,0.85)
print(f"Size of original dataset train: {len(id_train_cleaned_dedup)}")
print(f"Size of filtered dataset train: {len(ds_filter)}")
print(f"Time to deduplicate dataset: {time.time()-t_start:.2f}")

with open('/data/mc4_id_clean/mc4_id/duplicate_clusters.json', "w") as f:
    json.dump(duplicate_clusters, f)


t_start = time.time()
for file_number, index in enumerate(range(0, len(ds_filter), 10000000)):
    file_path = str(f"/data/mc4_id_clean/mc4_id/file-{file_number+1:012}.json")
    end_index = min(len(ds_filter), index + 10000000)
    ds_filter.select(list(range(index, end_index))).to_json(file_path)
    compress_file(file_path)
print(f"Time to save dataset: {time.time()-t_start:.2f}")

