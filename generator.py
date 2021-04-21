# -*- coding: utf-8 -*-

"""
Created on Sat Apr 17 12:47:39 2021

@author: matthieufuteral-peter
"""


import os
import json
import argparse
from logger import logger
import torch
from util import Dataset, load_model, prompt_format
from uuid import uuid4


def generate(prompt, model, args):
    model.eval()
    news = []
    if args.topk:
        sample_generated = model.generate(prompt,
                                          do_sample=True,
                                          min_length=50,
                                          max_length=512,
                                          top_k=args.topk,
                                          top_p=0.6,
                                          temperature=args.temperature,
                                          repetition_penalty=2.0,
                                          num_return_sequences=args.n_sentences
                                          )
    elif args.beam_search:
        sample_generated = model.generate(prompt,
                                          do_sample=True,
                                          min_length=50,
                                          max_length=512,
                                          num_beams=args.beam_search,
                                          repetition_penalty=5.0,
                                          early_stopping=True,
                                          num_return_sequences=1
                                          )
    else:
        raise AttributeError("Either args.beam_search or args.topk have to be set to > 0")

    for i, output in enumerate(sample_generated):
        text = tokenizer.decode(output, skip_special_tokens=True)
        news.append(text)

    return news


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_sentences', type=int, default=10,
                        help="Number of sentences to generate (default: 10)")
    parser.add_argument('--topk', type=int, default=0,
                        help="Generate tokens randomly picking among top k pred (default: 0)")
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="Temperature if topk != 0")
    parser.add_argument('--beam_search', type=int, default=0,
                        help="Length of the beam search (default: 0)")
    parser.add_argument('--keywords', nargs="+",
                        help="keywords to pass to the prompt (Optional)")
    parser.add_argument('--title', type=str,
                        help="Title of the prompt (Optional)")
    parser.add_argument('--cat', type=str, required=True,
                        help="Category of the news (Required)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("news_generated/news", exist_ok=True)
    if os.path.isfile("news_generated/labels.json"):
        with open("news_generated/labels.json", 'r') as fj:
            labels = json.load(fj)
    else:
        labels = dict()

    # Generate ids
    news_ids = [str(uuid4())[:8] for _ in range(args.n_sentences)]
    if args.topk:
        for news_id in news_ids:
            labels[news_id] = args.cat
    elif args.beam_search:
        labels[news_ids[0]] = args.cat
    else:
        raise AttributeError("Either args.beam_search or args.topk have to be set to > 0")

    # Load tokenizer
    tokenizer = Dataset.config_tokenizer(model="gpt2")

    # Load model
    model = load_model(tokenizer=tokenizer, device=device, model_type="gpt2", path_load="models/gpt2/pytorch_model.bin")
    # checkpoint = torch.load("models/gpt2/pytorch_model.bin", map_location=device)
    # model.load_state_dict(checkpoint['model'])

    prompt = torch.Tensor(tokenizer.encode(prompt_format(args.cat, args.title, args.keywords))).unsqueeze(0)
    prompt = prompt.to(device)

    news = generate(prompt, model, args)
    for idx, text in enumerate(news):
        with open(f"news_generated/news/{news_ids[idx]}", "w") as f:
            f.write(text)

    with open("news_generated/labels.json", "w") as fj:
        json.dump(labels, fj)


