from fastcore.basics import listify
import unicodedata
import unidecode
from string import punctuation
import html
from itertools import groupby
import fasttext
import re

control_char_regex = re.compile(r'[\r\n\t]+')
url_regex = re.compile(
    r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*')
username_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')

FASTTEXT_MODEL_PATH = '/data/lid.176.bin'
fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)


def fix_html(example):
    "From fastai: 'Fix messy things we've seen in documents'"
    tmp_ls = []
    for e in listify(example['text']):
        e = e.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ').replace(
            '#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
            '\\"', '"').replace('<unk>', ' ').replace(' @.@ ', '.').replace(' @-@ ', '-').replace('...', ' …')
        tmp_ls.append(html.unescape(e))

    example['text'] = tmp_ls
    return example


def remove_control_char(example):
    tmp_ls = []
    for e in listify(example['text']):
        tmp_ls.append(re.sub(control_char_regex, '.', e))

    example['text'] = tmp_ls
    return example


def remove_remaining_control_chars(example):
    tmp_ls = []
    for e in listify(example['text']):
        tmp_ls.append(
            ''.join(ch for ch in e if unicodedata.category(ch)[0] != 'C'))

    example['text'] = tmp_ls
    return example


def remove_unicode_symbols(example):
    tmp_ls = []
    for e in listify(example['text']):
        tmp_ls.append(
            ''.join(ch for ch in e if unicodedata.category(ch)[0] != 'So'))

    example['text'] = tmp_ls
    return example


def standardise_punc(example):
    transl_table = dict([(ord(x), ord(y))
                         for x, y in zip(u"‘’´“”–-",  u"'''\"\"--")])
    tmp_ls = []
    for e in listify(example['text']):
        e = e.translate(transl_table)
        e = re.sub(r"[^a-zA-Z0-9ÖÄÅöäå .,'%&€$=*@+;<>/()!?%:-]", " ", e)
        tmp_ls.append(e)

    example['text'] = tmp_ls
    return example


def remove_news_tags(example):
    tmp_ls = []
    for e in listify(example['text']):
        e = re.sub(r"(<[A-Z].+?>)|(</[A-Z].+?>)", "", e)
        tmp_ls.append(e)

    example['text'] = tmp_ls
    return example


def replace_urls(example):
    filler, tmp_ls = '', []
    for e in listify(example['text']):
        e = re.sub(r"(<a.+?>)|(</a>)|(<ref.+?>)", "", e)
        e = re.sub(url_regex, filler, e)
        tmp_ls.append(e)

    example['text'] = tmp_ls
    return example


def replace_usernames(example):
    filler, tmp_ls = '', []
    for e in listify(example['text']):
        occ = e.count('@')
        for _ in range(occ):
            e = e.replace('@<user>', f'{filler}')
            # replace other user handles by filler
            e = re.sub(username_regex, filler, e)
            # add spaces between, and remove double spaces again
            e = e.replace(filler, f' {filler} ')
            e = ' '.join(e.split())
        tmp_ls.append(e)

    example['text'] = tmp_ls
    return example


def remove_duplicate_words_punctuation(example):
    tmp_ls = []
    for e in listify(example['text']):
        e = re.sub(r'\b(\w+)( \1\b)+', r'\1', e)
        punc = set(punctuation)
        newtext = []
        for k, g in groupby(e):
            if k in punc:
                newtext.append(k)
            else:
                newtext.extend(g)
        e = ''.join(newtext)
        tmp_ls.append(e)

    example['text'] = tmp_ls
    return example


def remove_multi_space(example):
    tmp_ls = []
    for e in listify(example['text']):
        tmp_ls.append(' '.join(e.split()))

    example['text'] = tmp_ls
    return example


def count_alphabet(batch):
    batch['alphabet_len'] = len(re.findall(r'[äÄöÖåÅa-zA-Z]', batch['text']))
    return batch


def count_numbers(batch):
    batch['number_len'] = len(re.findall(r'[0-9]', batch['text']))
    return batch


def count_upper(batch):
    batch['upper_len'] = len(re.findall(r'[ÄÖÅA-Z]', batch['text']))
    return batch


def count_str_len(batch):
    batch['total_len'] = len(batch['text'])
    return batch


def predict_lang(batch):
    pred = fasttext_model.predict(batch['text'])
    batch['predicted_lang'] = pred[0][0]
    batch['predicted_lang_percentage'] = float(pred[1][0])
    return batch


def calculate_alphabet_ratio(batch):
    batch['alphabet_ratio'] = int(
        batch['alphabet_len']) / int(batch['total_len'])
    return batch


def calculate_number_ratio(batch):
    batch['number_ratio'] = int(batch['number_len']) / int(batch['total_len'])
    return batch


def calculate_upper_ratio(batch):
    batch['upper_ratio'] = int(batch['upper_len']) / int(batch['total_len'])
    return batch
