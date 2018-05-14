#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the main script for generating embeddings by RegRBM.
"""
from __future__ import print_function

import numpy as np

from gensim import corpora
from gensim.matutils import corpus2dense

from utils.plotter import matrix_plotter
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

    # data_x[data_x < 1e-10] = 1e-10
    # matrix_plotter(-1 * np.log(data_x))

    print(data_x.shape)

    t   = 1e-2
    lam = 1e-3
    lr  = 1e-3
    n_epoches = 20

    rbm = RegRBM(n_visible=n_x, n_hidden=1000, t=t, lam=lam, \
                 learning_rate=lr, momentum=0.95, err_function="mse", \
                 sample_visible=False, sigma=1.)
    errs, zeros = rbm.fit(data_x, n_epoches=n_epoches, batch_size=20, \
                          shuffle=True, verbose=True)

    embeddings   = rbm.transform(data_x).round().astype(int)
    nonzero_vars = rbm.get_nonzero_vars(data_x)

    # save results
    np.savetxt("resource/vars.lam%1.e.lr%1.e.t%1.e.epoch%d.txt" % (lam, lr, t, n_epoches), nonzero_vars, delimiter=',')
    np.savetxt("resource/errors.lam%1.e.lr%1.e.t%1.e.epoch%d.txt" % (lam, lr, t, n_epoches), errs, delimiter=',')
    np.savetxt("resource/zeros.lam%1.e.lr%1.e.t%1.e.epoch%d.txt" % (lam, lr, t, n_epoches), zeros, delimiter=',')
    np.savetxt("resource/embeddings/2k.embeddings.lam%1.e.lr%1.e.t%1.e.epoch%d.txt" % (lam, lr, t, n_epoches), embeddings, delimiter=',')
    # vec2tsne(info_name, "results/test.pdf", vectors=embeddings, n=2)
