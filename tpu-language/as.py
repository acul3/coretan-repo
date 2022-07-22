from datasets import load_dataset
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
new_tokenizer = AutoTokenizer.from_pretrained("deberta-id-tokenizer")
example = "aku suka kamu dan aku cuma pengen kamu tahu kalau"
example_1 = "i love you and just want you to know that there is no way i w"
print(old_tokenizer.tokenize(example_1))
print(new_tokenizer.tokenize(example))