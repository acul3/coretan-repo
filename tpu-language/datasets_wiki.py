from datasets import load_dataset

wiki_id = load_dataset("wikipedia", language="id", date="20220620",beam_runner='DirectRunner')