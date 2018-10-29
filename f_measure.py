#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the main script for evaluating method by F-measure (F_1 score).
"""
from __future__ import print_function

import sys
import numpy as np
from utils.mat2img import mat2img
from utils.vec2tsne import vec2tsne
from utils.eval4vec import eval_by_cosine

label_path  = "data/2000+56.dataset/new.info.txt"
label_inds  = range(56)
Ks          = [20, 40, 80, 100]
# load labels
labels = []
with open(label_path, "r") as fhandler:
    for line in fhandler:
        doc_ind  = line.strip().split("\t")[0]
        catagory = line.strip().split("\t")[1]
        labels.append(catagory)
embeddings = np.loadtxt("/Users/woodie/Desktop/workspace/Event-Series-Detection/resource/embeddings/svd.txt", delimiter=",")
# embeddings = np.loadtxt("resource/embeddings/reg.1e-3.lr.1e-3.2k.recon.txt", delimiter=",")
# embeddings = np.loadtxt("resource/embeddings/2k.embeddings.lam0e+00.lr1e-03.t1e-02.epoch20.txt", delimiter=",")
scores = [ eval_by_cosine(embeddings, labels, label_inds=label_inds, top_k=k, type="f_measure") for k in Ks ]
print(scores)

# # RBM with regularization
# [0.15831824908807041, 0.14106623564416126, 0.1094700518782232, 0.096323801425203318]
# # regular RBM
# [0.076563426203208898, 0.072932631783345955, 0.063187276466536366, 0.055808217241579479]
# # LDA
# [0.054438022759614353, 0.065424181538917603, 0.0711249791059681, 0.069326223352732738]
# # Autoencoder
# [0.020288035440106472, 0.014952014571509634, 0.012322640429558076, 0.012212399671753981]
