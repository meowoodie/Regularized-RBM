#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script includes various of validation approaches for evaluating the
performance of embedding vectors.
"""
from __future__ import print_function

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def eval_by_cosine(vectors, labels,
    label_inds=range(56), top_k=20,
    type="mat_rate"): # "mat_rate", "avg_rate"
    """
    Evaluate embedding quality by measuring pairwise cosine similarities between
    embedding vectors.
    """
    # numeralize labels
    label_set = list(set(labels))
    labels    = np.array([ label_set.index(label) for label in labels ])
    # calculate pairwise cosine distance for input vectors
    sim_mat   = cosine_similarity(vectors, dense_output=True)
    # sort similarity matrix by rows,
    # each row represents orders of similarities against other vectors.
    order_mat = sim_mat.argsort(axis=1)
    # trim the matrix by removing the last column (similarity against itself)
    order_mat = np.flip(order_mat[:, 0:-1], axis=1)
    # convert order_mat to label_mat which only consists of zeros (uncorrelated)
    # and ones (correlated) i.e. the ground truth.
    label_mat = np.zeros(order_mat.shape)
    for i in range(order_mat.shape[0]):
        label_mat[i, :] = np.array([
            1 if labels[j] == labels[i] else 0 # 1 if labels are same, otherwise 0
            for j in order_mat[i, :] ])        # for all vectors in matrix
    # calculate hit rate according to different evaluation methods
    if type == "mat_rate":
        hit_rate = label_mat[label_inds, 0:top_k].sum() / label_mat[label_inds, :].sum()
    elif type == "avg_rate":
        hit_rate = (label_mat[label_inds, 0:top_k].sum(axis=1) / label_mat[label_inds, :].sum(axis=1)).mean()

    return hit_rate



if __name__ == "__main__":
    label_path = "data/new.info.txt"
    labels     = []
    with open(label_path, "r") as fhandler:
		for line in fhandler:
			doc_ind  = line.strip().split("\t")[0]
			catagory = line.strip().split("\t")[1]
			labels.append(catagory)

    vectors = np.loadtxt("resource/embeddings/sub.2k.corpus.txt", delimiter=",")
    eval_by_cosine(vectors, labels[0:256], type="avg_rate")
