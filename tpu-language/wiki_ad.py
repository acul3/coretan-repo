from dataclasses import asdict
from multiprocessing import Pool
import os 
import glob
import jsonlines
all_files = glob.glob('wiki_id/AA/*')
def process_data(name):
    with jsonlines.open('/data/wiki_id/AA/wiki_00', 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return name

if __name__ == '__main__':
    pool = Pool(os.cpu_count())                         
    pool.map(process_image, all_files)
