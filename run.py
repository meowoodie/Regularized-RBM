#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the main script for testing algorithm on real dataset, which includes
helpful function for data preparing, model training, and visualizing.
"""
import numpy as np

from gensim import corpora
from gensim.matutils import corpus2dense

from rbm import SemiSupervRBM

if __name__ == "__main__":
    dict_name   = "resource/dict/2k.bigram.dict"
    corpus_name = "resource/corpus/2k.bigram.doc.tfidf.corpus"
    info_name   = "data/new.info.txt"

    ngram_dict   = corpora.Dictionary.load(dict_name)
    corpus_tfidf = corpora.MmCorpus(corpus_name)

    # get corpus matrix
    data_x = corpus2dense(corpus_tfidf, num_terms=len(ngram_dict)).transpose()
    n_x    = data_x.shape[1]
    print(data_x.shape)

    # get labels info
    labels = None
    with open(info_name, "r") as f:
        # fetch labels
        labels = np.array([ line.strip().split("\t")[1] for line in f ])
    label_set = list(set(labels))
    n_y       = len(set(labels))
    data_y    = []
    for label in labels:
        d = np.zeros(n_y)
        d[label_set.index(label)] = 1
        data_y.append(d)
    data_y    = np.array(data_y)

    rbm = SemiSupervRBM(n_y=n_y, n_x=n_x, n_h=1000, alpha=0.1, \
                        learning_rate=0.005, momentum=0.95, err_function='mse', \
                        sample_visible=False)
    rbm.fit(data_x, data_y, n_epoches=10, batch_size=100, shuffle=True)
    embeddings = rbm.transform(data_x).round().astype(int)

    # save embeddings
    file_name="new.2k.corpus"
    np.savetxt("resource/embeddings/%s.txt" % file_name, embeddings, delimiter=',')
