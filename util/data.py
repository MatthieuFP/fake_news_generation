# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""

import os
import random
import re
import pdb
import glob
import pandas as pd
from transformers import GPT2Tokenizer


class Dataset:

    def __init__(self):

        self.metadata = pd.read_csv("./data/news_aggregator/newsCorpora.csv", sep='\t', index_col=0,
                                    names=['title', 'url', 'publisher', 'category', 'story', 'hostname', 'timestamp'])

        self.dataset = glob.glob("./data/articles/*.txt")

        with open('./data/keywords.csv', 'r') as f:
            keywords = f.readlines()
        keywords = [re.sub(r'\n', '', k) for k in keywords]

        self.keywords = {int(keyword.split(',')[0]): keyword.split(',')[1:] for keyword in keywords}
        self.tokenizer = self.config_tokenizer()

    def __getitem__(self, idx):

        path_id = self.dataset[idx]
        news_id = int(re.findall('[0-9]+', os.path.basename(path_id))[0])

        with open(path_id, 'r') as f:
            sample = f.read()
        sample = re.sub(r'[\n]+', r'\n', sample)

        sample_met = self.metadata.loc[news_id]
        keywords = self.keywords[news_id]
        title, cat = sample_met.title, sample_met.category

        inputs, attn_mask = self.process_input(title, sample, cat, keywords, self.tokenizer)

        return inputs, attn_mask

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def config_tokenizer():
        special_tokens = {"bos_token": "[BOS]",
                    "eos_token": "[EOS]",
                    "unk_token": "[UNK]",
                    "pad_token": "[PAD]",
                    "sep_token": "[SEP]",
                    "additional_special_tokens": ['<b>', '<t>', '<e>', '<m>']}
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens(special_tokens)
        return tokenizer

    @staticmethod
    def process_input(title, sample, cat, keywords, tokenizer):

        random.shuffle(keywords)
        keys = ", ".join(keywords)
        inp = "[BOS] " + f"<{cat}> " + title + " [SEP] " + keys + sample + " [EOS]"

        encoded_inp = tokenizer(inp, truncation=True, max_length=1024, padding="max_length", return_tensors="pt")

        return encoded_inp["input_ids"], encoded_inp["attention_mask"]


if __name__ == '__main__':

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    data = Dataset()
    pdb.set_trace()




