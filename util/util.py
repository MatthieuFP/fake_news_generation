# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""


from transformers import GPT2LMHeadModel, GPT2Config, BertForSequenceClassification, BertConfig


def load_model(tokenizer, device, model_type):
    if model_type == 'gpt2':
        config = GPT2Config.from_pretrained(model_type,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
        model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
        model.resize_token_embeddings(len(tokenizer))
    elif model_type == 'bert-base-uncased':
        config = BertConfig.from_pretrained(model_type, num_labels=4)
        model = BertForSequenceClassification.from_pretrained(model_type, config=config)

    model.to(device)
    return model


