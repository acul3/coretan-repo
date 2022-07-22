from datasets import load_dataset
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("indonesian-nlp/mc4-id", "tiny")

def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000000]["text"]
        for i in range(0, len(raw_datasets["train"]), 1000000)
    )


training_corpus = get_training_corpus()


tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 128100,show_progress=True)
tokenizer.save_pretrained("deberta-id-tokenizer") 