# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""


import os
import time
import pdb
import argparse
from logger import logger
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from util import Dataset, load_model
from uuid import uuid4
from sklearn.metrics import classification_report
from pprint import pprint


def train(epoch, model, train_loader, use_cuda, train_loss, accumulation_steps, optimizer):

    optimizer.zero_grad()
    model.train()
    train_batch_loss = []
    loss = 0

    for batch_idx, data in tqdm(enumerate(train_loader, 1)):

        inputs, labels, attn_mask = data["input_ids"], data["label"], data["attention_mask"]

        if use_cuda:
            inputs, labels, attn_mask = inputs.cuda(), labels.cuda(), attn_mask.cuda()

        output = model(input_ids=inputs, attention_mask=attn_mask, labels=labels)
        loss += output.loss
        train_batch_loss.append(output.loss.item())

        if not batch_idx % accumulation_steps:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = 0

    train_loss.append(np.mean(train_batch_loss))
    logger.info('Train loss Epoch {} : {}'.format(epoch, train_loss[-1]))

    return model, train_loss


def test(model, test_loader, use_cuda, test_loss, batch_size, mode="val"):

    model.eval()
    test_batch_loss = 0
    correct = 0
    predictions, labs = [], []
    with torch.no_grad():
        for data in tqdm(test_loader):

            inputs, labels, attn_mask = data["input_ids"], data["label"], data["attention_mask"]

            if use_cuda:
                inputs, labels, attn_mask = inputs.cuda(), labels.cuda(), attn_mask.cuda()

            output = model(input_ids=inputs, attention_mask=attn_mask, labels=labels)
            # sum up batch loss
            test_batch_loss += output.loss.item()
            # get the index of the max log-probability

            pred = output.logits.max(1, keepdim=True)[1]
            if mode == "test":
                predictions.append(pred.cpu().numpy().flatten())
                labs.append(labels.cpu().numpy().flatten())
            correct += pred.eq(labels.view_as(pred)).cpu().sum()

    test_batch_loss /= len(test_loader.dataset)
    test_batch_loss *= batch_size
    score = 100. * correct / len(test_loader.dataset)

    test_loss.append(test_batch_loss)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_batch_loss, correct, len(test_loader.dataset),
        score))

    if mode == "test":
        return np.concatenate(predictions), np.concatenate(labs)
    else:
        return test_loss, score.data.item()


def main(model, epochs, train_loader, test_loader, optimizer, use_cuda, accumulation_steps, output_dir_model, patience, args):

    train_loss, test_loss, test_accuracy, epoch_time = [], [], [], []

    for epoch in range(1, epochs + 1):
        logger.info("Epoch {} - Start TRAINING".format(epoch))
        t0 = time.time()

        # Training step
        model, train_loss = train(epoch, model, train_loader, use_cuda, train_loss, accumulation_steps, optimizer)

        # Testing mode - test on the validation set
        test_loss, test_score = test(model, test_loader, use_cuda, test_loss, args.batch_size)
        test_accuracy.append(test_score)

        if epoch == 1:
            logger.info("Save model ... Epoch 1")
            index = 0
            ref_score = test_score
            torch.save(model.state_dict(), os.path.join(output_dir_model, "model.pt"))
        else:
            if test_score > ref_score:
                ref_score = test_score
                index = 0
                torch.save(model.state_dict(), os.path.join(output_dir_model, "model.pt"))
                logger.info("Save model ... Epoch {}".format(epoch))
            else:
                index += 1

        if index > patience:
            print("Stop training... Patience reached")
            break

        time_elapsed = time.time() - t0
        logger.info("Epoch {} - Time elapsed : {}".format(epoch, time_elapsed))
        epoch_time.append(time_elapsed)

    logger.info("Average time per epoch : {}".format(np.mean(epoch_time)))

    return model, train_loss, test_loss, test_accuracy, epoch_time


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_step', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    RUN_ID = str(uuid4())[:4]
    print(f"RUN ID : {RUN_ID}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = True if torch.cuda.is_available() else False

    # Save dir
    output_dir_model = f"output_dir/bert"
    os.makedirs(output_dir_model, exist_ok=True)

    # load model
    dataset = Dataset(model="bert-base-uncased")

    # Build validation set
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=0)

    # Load model
    model = load_model(dataset.tokenizer, device, model_type="bert-base-uncased")
    for params in model.bert.parameters():
        params.requires_grad = False

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    if args.training:
        model, train_loss, test_loss, test_accuracy, epoch_time = main(model, args.epochs, train_loader, test_loader,
                                                                       optimizer, use_cuda, args.gradient_step,
                                                                       output_dir_model, args.patience, args)

        state_dict = torch.load(os.path.join(output_dir_model, 'model.pt'), map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        predictions, labels = test(model, test_loader, use_cuda, [], mode="test")

        pprint(classification_report(labels, predictions))











