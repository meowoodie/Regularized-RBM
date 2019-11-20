#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the main script for cross validation with RegRBM.
"""
from __future__ import print_function

import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
from gensim.matutils import corpus2dense
from gensim import corpora

# from utils.mat2img import mat2img
# from utils.vec2tsne import vec2tsne
# from utils.eval4vec import eval_by_cosine

from rbm import RegRBM
from sklearn.model_selection import KFold

if __name__ == "__main__":
    dict_name   = "resource/dict/2k.bigram.dict"
    corpus_name = "resource/corpus/2k.bigram.doc.tfidf.corpus"
    log_lams    = np.linspace(-10, 0, num=101)[1:]
    lams        = np.exp(log_lams)

    ngram_dict   = corpora.Dictionary.load(dict_name)
    corpus_tfidf = corpora.MmCorpus(corpus_name)

    # get corpus matrix
    data_x = corpus2dense(corpus_tfidf, num_terms=len(ngram_dict)).transpose()
    n_x    = data_x.shape[1]

    # k-fold cross validation
    kf = KFold(n_splits=5)

    errs  = []
    zeros = []
    for lam in lams:
        # init k-fold scores
        k_fold_err  = []
        k_fold_zero = []
        for train_index, test_index in kf.split(data_x):
            # get (k-1)-fold train data and 1-fold test data
            x_train, x_test = data_x[train_index], data_x[test_index]
            # fit RegRBM and get score
            rbm = RegRBM(n_visible=n_x, n_hidden=1000, t=1e-2, lam=lam, \
                         learning_rate=1e-2, momentum=0.95, err_function="mse", \
                         sample_visible=False, sigma=1.)
            rbm.fit(x_train, n_epoches=8, batch_size=20, \
                    shuffle=True, verbose=True)
            err  = rbm.get_err(x_test)
            zero = rbm.get_zero(x_test)
            # append to k_fold_score list
            k_fold_err.append(err)
            k_fold_zero.append(zero)
        # append to score list
        errs.append(k_fold_err)
        zeros.append(k_fold_zero)

    # save score
    np.savetxt("resource/new_cv_errs.txt", np.array(errs), delimiter=",")
    np.savetxt("resource/new_cv_zeros.txt", np.array(zeros).astype(int), delimiter=",")
