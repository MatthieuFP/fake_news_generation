# -*- coding: utf-8 -*-

"""
Created on Sat Apr 17 12:47:39 2021

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
from util import Dataset, GeneratedData
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
    parser.add_argument('--generated', action='store_true')
    args = parser.parse_args()

    random.seed(42)
    labels = {'b': 'b', 'm': 'r', 't': 'g', 'e': 'y'}
    vectorizer = TfidfVectorizer(input='filename')
    svd = TruncatedSVD(n_components=args.n_components)

    if not args.generated:
        logger.info(f"Start LSA with original data : {args.n_news}")
        dataset = Dataset()
        corpus, category = build_sample(dataset, n_news=args.n_news)

    else:
        logger.info("Start LSA with generated data")
        dataset = GeneratedData()
        corpus = dataset.dataset
        category = []
        for path_id in dataset.dataset:
            news_id = os.path.basename(path_id)[:-4]
            category.append(dataset.labels[news_id])

    X = vectorizer.fit_transform(corpus)

    logger.info("Start SVD")
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

    os.makedirs("plots", exist_ok=True)

    if args.generated:
        fig.savefig("plots/tsne_lsa_generated.png")
    else:
        fig.savefig("plots/tsne_lsa.png")
        
    plt.show()


