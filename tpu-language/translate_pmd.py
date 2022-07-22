import argparse
import csv
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate
import gc
import gzip
import jax
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import FlaxMarianMTModel, MarianTokenizer
import json
import glob
all_files = sorted(glob.glob('/data/pmd/data/*'))

model = FlaxMarianMTModel.from_pretrained("Wikidepia/marian-nmt-enid", from_pt=True)

tokenizer = MarianTokenizer.from_pretrained("Wikidepia/marian-nmt-enid", source_lang="en")

def generate(params, batch):
      output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], params=params, num_beams=4, max_length=64, early_stopping=True).sequences
      return output_ids

p_generate = jax.pmap(generate, "batch")

p_params = replicate(model.params)
BATCH_SIZE = 1024

def run_generate(input_str):
  inputs = tokenizer(input_str, return_tensors="jax", padding="max_length", truncation=True, max_length=64)
  p_inputs = shard(inputs.data)
  output_ids = p_generate(p_params, p_inputs)
  output_strings = tokenizer.batch_decode(output_ids.reshape(-1, 64), skip_special_tokens=True, max_length=64)
  return output_strings

def arrange_data_pmd(dataset_id,caption,annotator_id,image_id,original_caption):
    try:
        lis_ = []
        caption_len = len(original_caption)
        if caption_len < BATCH_SIZE :
            dummy = ["tidak"] * (BATCH_SIZE - caption_len)
            caption = caption + dummy
            original_caption = original_caption + dummy
            print("panjang original",len(original_caption),len(dummy))
            
        caption = run_generate(caption)[0:caption_len]
        original_caption = run_generate(original_caption)[0:caption_len]

        for dataset_id,caption,annotator_id,image_id,original_caption in zip(dataset_id,caption,annotator_id,image_id,original_caption):  # add other captions
                lis_.append({"dataset_id":dataset_id,"annotator_id":annotator_id,"image_id":image_id,"caption":caption,"original_caption":original_caption})

        gc.collect()
        return lis_

    except Exception as e:
        print(e)
        return

for file in enumerate(tqdm(all_files,desc='iterate fille',position = 0,leave=True)):
    #print(file[1])
    file_name = file[1].split('/')[-1]
    ls = []
    with open(file[1], "rb") as f:
        for i,line in enumerate(f):
            if line:
                example = json.loads(line)
                ls.append(example)
    for i in tqdm(range(0,len(ls),BATCH_SIZE),desc='Translating',position = 1,leave=True):
        dataset_id = [m.get('dataset_id') for m in ls[i:i+BATCH_SIZE]]
        caption = [m.get('caption') for m in ls[i:i+BATCH_SIZE]]
        annotator_id = [m.get('annotator_id') for m in ls[i:i+BATCH_SIZE]]
        original_caption = [m.get('original_caption') for m in ls[i:i+BATCH_SIZE]]
        image_id = [m.get('image_id') for m in ls[i:i+BATCH_SIZE]]
        output_batch = arrange_data_pmd(dataset_id,caption,annotator_id,image_id,original_caption)
        name = '/data/pmd_indonesia/' + file_name
        with open(name, "a") as outfile:
            for batch in output_batch:
                json.dump(batch, outfile)
                outfile.write('\n')