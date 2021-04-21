# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""

import os
import torch
import random
import re
import pdb
import glob
import pandas as pd
from transformers import AutoTokenizer


class Dataset:

    def __init__(self, model='gpt2'):

        self.metadata = pd.read_csv("./data/news_aggregator/newsCorpora.csv", sep='\t', index_col=0,
                                    names=['title', 'url', 'publisher', 'category', 'story', 'hostname', 'timestamp'])

        self.dataset = glob.glob("./data/articles/*.txt")
        self.model = model
        assert self.model in ['gpt2', 'bert-base-uncased'], "Model not found : must be gpt2 or bert-base-uncased"

        with open('./data/keywords.csv', 'r') as f:
            keywords = f.readlines()
        keywords = [re.sub(r'\n', '', k) for k in keywords]

        self.keywords = {int(keyword.split(',')[0]): keyword.split(',')[1:] for keyword in keywords}
        self.tokenizer = self.config_tokenizer(model)
        self.cat_2_int = {'b': 0, 't': 1, 'e': 2, 'm': 3}
        self.int_2_cat = {v: k for k, v in self.cat_2_int.items()}

    def __getitem__(self, idx):

        path_id = self.dataset[idx]
        news_id, sample = self.open_file(path_id)

        sample_met = self.metadata.loc[news_id]
        keywords = self.keywords[news_id]
        title, cat = sample_met.title, sample_met.category

        inputs, attn_mask = self.process_input(title, sample, cat, keywords, self.tokenizer, self.model)
        if self.model == 'bert-base-uncased':
            label = torch.LongTensor([self.cat_2_int[cat]])
            inputs = inputs.flatten()
            attn_mask = attn_mask.flatten()
        elif self.model == 'gpt2':
            label = inputs.clone()
        else:
            raise NameError("Unrecognized model")

        return {"input_ids": inputs, "label": label, "attention_mask": attn_mask}

    def __len__(self):
        return len(self.dataset)


    @staticmethod
    def open_file(path_id):
        news_id = int(re.findall('[0-9]+', os.path.basename(path_id))[0])
        with open(path_id, 'r') as f:
            sample = f.read()
        sample = re.sub(r'[\n]+', r'\n', sample)
        return news_id, sample

    @staticmethod
    def config_tokenizer(model):
        tokenizer = AutoTokenizer.from_pretrained(model)

        if model == 'gpt2':
            special_tokens = {"bos_token": "[BOS]",
                              "eos_token": "[EOS]",
                              "unk_token": "[UNK]",
                              "pad_token": "[PAD]",
                              "sep_token": "[SEP]",
                              "additional_special_tokens": ['<b>', '<t>', '<e>', '<m>']}

            tokenizer.add_special_tokens(special_tokens)

        return tokenizer

    @staticmethod
    def process_input(title, sample, cat, keywords, tokenizer, model):

        if model == 'gpt2':
            random.shuffle(keywords)
            keys = ", ".join(keywords)
            inp = "[BOS] " + f"<{cat}> " + " [SEP] " + title + " [SEP] " + keys + " [SEP] " + sample + " [EOS]"
            max_len = 1024
        elif model == 'bert-base-uncased':
            inp = sample
            max_len = 512
        else:
            raise NameError("model unrecognized")

        encoded_inp = tokenizer(inp, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")

        return encoded_inp["input_ids"], encoded_inp["attention_mask"]


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    data = Dataset()
    pdb.set_trace()




