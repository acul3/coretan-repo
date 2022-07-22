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


t_start = time.time()
id_mc4_validation = datasets.load_dataset("mc4", "id",split='validation')
ds_filter, duplicate_clusters = deduplicate_dataset(id_mc4_validation,0.85)

with open('/data/mc4_id_clean/mc4_id/duplicate_clusters.json', "w") as f:
    json.dump(duplicate_clusters, f)
print(f"Time to deduplicate dataset: {time.time()-t_start:.2f}")
print(f"Size of original dataset: {len(id_mc4_validation)}")
print(f"Size of deduplicate dataset: {len(ds_filter)}")

print(ds_filter)