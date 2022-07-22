from fileinput import filename
import glob
from lm_dataformat import Reader
from tqdm import tqdm
import json
import gzip
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

all_files = glob.glob('ccnews-id/*')

def get_list(filename):
  rdr = Reader(filename)

  asa = []
  all_sentence_num = []
  for doc in rdr.stream_data():
    if doc is not None:
      text = {"text":doc,"meta":{"set_name":"cc-news","url":None}}
      asa.append(text)
      all_sentence_num.append(len(sent_tokenize(doc)))
  return asa,all_sentence_num

big_list = []
total_sentence = []
for files in tqdm(all_files):
  try:
    a,b = get_list(files)
    big_list.extend(a)
    total_sentence.extend(b)
  except :
    print("An exception occurred",filename)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
a = list(chunks(big_list,7019))
print(len(big_list))
print("total_sentence",sum(total_sentence))
print("saving")

def create_json(lis,name):
    json_lines = [json.dumps(l) for l in lis]
    json_data = '\n'.join(json_lines)
    filename = "cc_news/" + str(name) + ".jsonl"
    with gzip.open(filename, 'wt', encoding='UTF-8') as fout:
        fout.write(json_data)


for i, data in enumerate(tqdm(a)):
    name = "cc-news-" + str(i)
    create_json(data,name)