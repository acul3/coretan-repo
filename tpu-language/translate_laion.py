import argparse
import csv
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate
import gc
import jax
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import FlaxMarianMTModel, MarianTokenizer
import glob

all_files = sorted(glob.glob('/data/laion400m/*'))

model = FlaxMarianMTModel.from_pretrained("Wikidepia/marian-nmt-enid", from_pt=True)

tokenizer = MarianTokenizer.from_pretrained("Wikidepia/marian-nmt-enid", source_lang="en")

def generate(params, batch):
      output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], params=params, num_beams=4, max_length=64, early_stopping=True).sequences
      return output_ids

p_generate = jax.pmap(generate, "batch")

p_params = replicate(model.params)

def run_generate(input_str):
  inputs = tokenizer(input_str, return_tensors="jax", padding="max_length", truncation=True, max_length=64)
  p_inputs = shard(inputs.data)
  output_ids = p_generate(p_params, p_inputs)
  output_strings = tokenizer.batch_decode(output_ids.reshape(-1, 64), skip_special_tokens=True, max_length=64)
  return output_strings

def read_tsv_file(tsv_path):
    _df = pd.read_parquet(tsv_path)
    new_df = _df[_df['TEXT'].notna()]
    print("Number of Examples:", new_df.shape[0], "for", tsv_path)
    return new_df

def arrange_data(sample_id,url,text,height,width,license,nsfw,similarity):  # iterates through all the captions and save there translations
    try:
        lis_ = []

        output = run_generate(text)

        for sample_id,url,text,height,width,license,nsfw,similarity in zip(sample_id,url,output,height,width,license,nsfw,similarity):  # add other captions
                lis_.append({"SAMPLE_ID":sample_id,"URL":url,"TEXT":text,"HEIGHT":height,"WIDTH":width,"LICENSE":license,"NSFW":nsfw,"similarity":similarity})

        gc.collect()
        return lis_

    except Exception as e:
        print("errornya",type(text))
        print(e)
        return

BATCH_SIZE = 2048


for i,file in enumerate(tqdm(all_files,desc='iterate fille',position = 0,leave=True)):
    df = read_tsv_file(file)
    name = "part_" + str(i+1) + '.tsv'
    output_file_name = os.path.join('/data/laion_indo_400m', name)
    with open(output_file_name, 'w', newline='') as outtsv:  # creates a blank tsv with headers (overwrites existing file)
        writer = csv.writer(outtsv, delimiter='\t')
        writer.writerow(["SAMPLE_ID","URL","TEXT","HEIGHT","WIDTH","LICENSE","NSFW","similarity"])

    for i in tqdm(range(0,len(df),BATCH_SIZE),desc='Translating',position = 1,leave=True):
        output_batch = arrange_data(list(df["SAMPLE_ID"])[i:i+BATCH_SIZE], list(df["URL"])[i:i+BATCH_SIZE], list(df["TEXT"])[i:i+BATCH_SIZE],list(df["HEIGHT"])[i:i+BATCH_SIZE],list(df["WIDTH"])[i:i+BATCH_SIZE],list(df["LICENSE"])[i:i+BATCH_SIZE],list(df["NSFW"])[i:i+BATCH_SIZE],list(df["similarity"])[i:i+BATCH_SIZE])
        with open(output_file_name, "a", newline='') as f:
            try:
                writer = csv.DictWriter(f, fieldnames=["SAMPLE_ID","URL","TEXT","HEIGHT","WIDTH","LICENSE","NSFW","similarity"], delimiter='\t')
                for batch in output_batch:
                    writer.writerow(batch)
            except Exception as e:
                pass 