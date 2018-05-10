#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the main script for testing algorithm on real dataset, which includes
helpful function for data preparing, model training, and visualizing.
"""
import numpy as np

from gensim import corpora
from gensim.matutils import corpus2dense

from utils.mat2img import mat2img
from utils.vec2tsne import vec2tsne
from utils.eval4vec import eval_by_cosine

from rbm import GBRBM
from rbm import RegRBM

if __name__ == "__main__":
    dict_name   = "resource/dict/2k.bigram.dict"
    corpus_name = "resource/corpus/2k.bigram.doc.tfidf.corpus"
    info_name   = "data/2000+56.dataset/new.info.txt"

    ngram_dict   = corpora.Dictionary.load(dict_name)
    corpus_tfidf = corpora.MmCorpus(corpus_name)

    # get corpus matrix
    data_x = corpus2dense(corpus_tfidf, num_terms=len(ngram_dict)).transpose()
    n_x    = data_x.shape[1]

    print(data_x.shape)

    t   = 1e-2
    lam = 1e-3
    lr  = 1e-3
    n_epoches = 25

    rbm = RegRBM(n_visible=n_x, n_hidden=1000, t=t, lam=lam, \
                 learning_rate=lr, momentum=0.95, err_function="mse", \
                 sample_visible=False, sigma=1.)
    errs, zeros = rbm.fit(data_x, n_epoches=n_epoches, batch_size=20, \
                          shuffle=True, verbose=True)

    embeddings = rbm.transform(data_x).round().astype(int)

    # save results
    np.savetxt("resource/errors.lam%1.e.lr%1.e.t%1.e.epoch%d.txt" % (lam, lr, t, n_epoches), errs, delimiter=',')
    np.savetxt("resource/zeros.lam%1.e.lr%1.e.t%1.e.epoch%d.txt" % (lam, lr, t, n_epoches), zeros, delimiter=',')
    # np.savetxt("resource/embeddings/2k.embeddings.reg.1e-3.lr.1e-3.txt", embeddings, delimiter=',')
    # vec2tsne(info_name, "results/test.pdf", vectors=embeddings, n=2)

    # # get labels info
    # labels = None
    # with open(info_name, "r") as f:
    #     # fetch labels
    #     labels = np.array([ line.strip().split("\t")[1] for line in f ])
    # label_set = list(set(labels))
    # n_y       = len(set(labels))
    # data_y    = []
    # for label in labels:
    #     d = np.zeros(n_y)
    #     d[label_set.index(label)] = 1.
    #     data_y.append(d)
    # data_y    = np.array(data_y)
    #
    # print(data_y.shape)

    # rbm = SemiSupervRBM(n_y=n_y, n_x=n_x, n_h=1000, alpha=.5, batch_size=20, \
    #                     learning_rate=.01, momentum=0.95, err_function='mse', \
    #                     sample_visible=False)
    # rbm.fit(data_x, data_y, n_epoches=100, shuffle=True)
    # embeddings = rbm.transform(data_x).round().astype(int)
