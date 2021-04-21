# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""

import os
import pdb
import re
import numpy as np
from logger import logger
import argparse
import matplotlib.pyplot as plt
import random
from util import Dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE


def build_sample(dataset, n_news=0):

    if n_news:
        random.shuffle(dataset.dataset)
        dataset.dataset = dataset.dataset[:n_news]

    category = []
    for path_id in dataset.dataset:
        news_id = int(re.findall('[0-9]+', os.path.basename(path_id))[0])
        category.append(dataset.metadata.loc[news_id].category)

    return dataset.dataset, category


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--n_news', type=int, default=500)
    parser.add_argument('--n_components', type=int, default=256)
    args = parser.parse_args()

    random.seed(42)
    dataset = Dataset()
    corpus, category = build_sample(dataset, n_news=args.n_news)
    labels = {'b': 'b', 'm': 'r', 't': 'g', 'e': 'y'}

    vectorizer = TfidfVectorizer(input='filename')
    X = vectorizer.fit_transform(corpus)

    logger.info("Start SVD")
    svd = TruncatedSVD(n_components=args.n_components)
    X_transformed = svd.fit_transform(X)
    logger.info("Explained variance ratio : {:.4f}".format(svd.explained_variance_ratio_.sum()))

    logger.info("Start TSNE")
    tsne = TSNE(n_components=2)
    X_embedded = tsne.fit_transform(X_transformed)

    fig, ax = plt.subplots()
    for g in labels.keys():
        idx = np.where(np.array(category) == g)[0]
        ax.scatter(X_embedded[idx, 0], X_embedded[idx, 1], c=labels[g], label=g)

    ax.legend()
    plt.show()




