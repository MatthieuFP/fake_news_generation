# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""


import os
import argparse
import numpy as np
from logger import logger
import torch
from torch.utils.data import random_split
from util import Dataset, load_model
from uuid import uuid4
from transformers import Trainer, TrainingArguments


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    RUN_ID = str(uuid4())[:4]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    dataset = Dataset()

    # Build validation set
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Load model
    model = load_model(dataset.tokenizer, device)

    # Output dir
    output_dir = f"output_dir/{RUN_ID}"
    os.makedirs(output_dir, exist_ok=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    logger.info("Start training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=dataset.tokenizer
    )







