#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""
import numpy as np

from gensim import corpora
from gensim.matutils import corpus2dense

from plotter.mat2img import mat2img
from tfrbm import GBRBM

def gbrbm_embeddings(input_data,
    n_visible, n_hidden, lr=0.05, n_epoches=10, batch_size=20):
    # init rbm object
    rbm = GBRBM(n_visible=n_visible, n_hidden=n_hidden, \
                learning_rate=lr, momentum=0.95, err_function='mse', \
                use_tqdm=False, sample_visible=False, sigma=1)
    rbm.fit(input_data, n_epoches=n_epoches, batch_size=batch_size, \
            shuffle=True, verbose=True)
    embeddings = rbm.transform(input_data).round().astype(int)
    return rbm, embeddings



if __name__ == "__main__":
    dict_name   = "resource/dict/trigram_dict"
    corpus_name = "resource/corpus/trigram.doc.tfidf.corpus"
    info_name   = "data/56+446.info.txt"

    ngram_dict   = corpora.Dictionary.load(dict_name)
    corpus_tfidf = corpora.MmCorpus(corpus_name)

    # get corpus matrix
    dense_corpus = corpus2dense(corpus_tfidf, num_terms=len(ngram_dict)).transpose()
    print(dense_corpus.shape)

    # get location matrix
    location = None
    with open(info_name, "r") as f:
        # fetch location
        location = [ map(float, line.strip().split("\t")[3:]) for line in f ]
        location = np.array(location)
        # normalization
        max_vals = location.max(axis=0) # get max values for each of columns
        min_vals = location.min(axis=0) # get min values for each of columns
        location = (location - min_vals) / (max_vals - min_vals)

    # concatenate corpus and location into one single matrix
    input_data = np.concatenate((dense_corpus, location), axis=1)

    # generate gbrbm embeddings and get`fitted model
    rbm, embeddings = gbrbm_embeddings(input_data, n_visible=len(ngram_dict)+2,
        n_hidden=1000, n_epoches=10)

    # plot weight matrix of rbm
    weights_matrix, vbias_array, hbias_array = rbm.get_weights()
    bias_matrix = np.row_stack((vbias_array[-50:], hbias_array[-50:]))
    mat2img(bias_matrix)

    # save embeddings
    file_name="corpus_location"
    np.savetxt("resource/embeddings/%s.txt" % file_name, embeddings, delimiter=',')
