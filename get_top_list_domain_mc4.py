import gzip
import json
import os 
import glob
from tqdm.notebook import tqdm
from urllib.parse import urlparse
import validators
import pickle
from multiprocessing import Pool as ProcessPool
from collections import Counter
def check_oscar_not_wiki(all_name):
    pol = []
    print(all_name)
    with gzip.open(open(all_name, "rb"), "rt", encoding="utf-8") as f:
        for line in f:
            if line:
                example = json.loads(line)
                lis = urlparse(example['url']).netloc
                pol.append(lis)
    #sorted_pol = list(set(pol))
    return pol

if __name__ == '__main__':
    oscar_2109_meta = sorted(glob.glob('/data/cleaned_c4/multilingual/*.json.gz'))
    with ProcessPool(processes=96) as pool:
        result = pool.map(check_oscar_not_wiki, oscar_2109_meta)
    with open('domain_all_mc4.pkl', 'wb') as f:   
        pickle.dump(result, f)