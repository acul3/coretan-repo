import gzip
import json
import os 
import glob
from tqdm.notebook import tqdm
from urllib.parse import urlparse
import validators
import pickle
from multiprocessing import Pool as ProcessPool
with open('/data/not_in_wiki_list_oscar2109.pkl', 'rb') as f:
    not_in_wiki = pickle.load(f)
from collections import Counter
def check_oscar_not_wiki(all_name):
    pol = []
    print(all_name)
    with gzip.open(open(all_name, "rb"), "rt", encoding="utf-8") as f:
        for line in f:
            if line:
                example = json.loads(line)
                lis = urlparse(example['headers']['warc-target-uri']).netloc
                if lis in not_in_wiki:
                    pol.append(lis)
    #sorted_pol = list(set(pol))
    return pol

if __name__ == '__main__':
    oscar_2109_meta = sorted(glob.glob('/data/OSCAR-2109/packaged/id/*.jsonl.gz'))
    oscar_2109_text = sorted(glob.glob('/data/OSCAR-2109/packaged/id/*.txt.gz'))
    list_domain = []
    with ProcessPool(processes=23) as pool:
        result = pool.map(check_oscar_not_wiki, oscar_2109_meta)
    with open('domain_all_no_wiki.pkl', 'wb') as f:   
        pickle.dump(result, f)