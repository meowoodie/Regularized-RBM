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
    # remove diagonal elements (similarity against itself) of similarity matrix
    sim_mat   = sim_mat[~np.eye(sim_mat.shape[0],dtype=bool)].reshape(sim_mat.shape[0],-1)
    # sort similarity matrix by rows,
    # each row represents orders of similarities against other vectors.
    order_mat = sim_mat.argsort(axis=1)
    # flip the order matrix to make the elements with largest similarities be the first ones.
    order_mat = np.flip(order_mat, axis=1)
    # convert order_mat to label_mat which only consists of zeros (uncorrelated)
    # and ones (correlated) i.e. the ground truth.
    label_mat = np.zeros(order_mat.shape)
    for i in range(order_mat.shape[0]):
        label_mat[i, :] = np.array([
            1 if labels[j] == labels[i] else 0 # 1 if labels are same, otherwise 0
            for j in order_mat[i, :] ])        # for all vectors in matrix
    # calculate hit rate according to different evaluation methods
    if type == "mat_rate":
        score = label_mat[label_inds, 0:top_k].sum() / label_mat[label_inds, :].sum()
    elif type == "avg_rate":
        score = (label_mat[label_inds, 0:top_k].sum(axis=1) / label_mat[label_inds, :].sum(axis=1)).mean()
    elif type == "f_measure":
        P = label_mat[label_inds, 0:top_k].sum(axis=1) / top_k
        R = label_mat[label_inds, 0:top_k].sum(axis=1) / label_mat[label_inds, :].sum(axis=1)
        F = 2 * P * R / (P + R + 1e-5)
        score = (F * label_mat[label_inds, :].sum(axis=1)).sum() / label_mat[label_inds, :].sum()
    return score



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

    # top_k=20, 200 random cases with noise term: [0,    10,   20,   50,   100,  200,  500,  1000, 2000, 3000, 4000, 5000]
    # [0.5454931972789115, 0.5110544217687075, 0.51828231292517, 0.5386904761904762, 0.48724489795918363, 0.453656462585034, 0.3596938775510204, 0.3150510204081632, 0.20535714285714285, 0.14285714285714285, 0.13732993197278912, 0.13903061224489796]
