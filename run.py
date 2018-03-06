#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""
import numpy as np

from gensim import corpora
from gensim.matutils import corpus2dense

from plotter.mat2img import mat2img
from tfrbm import GBRBM
from superv_rbm import NNSupervRBM

def gbrbm_embeddings(input_data, n_visible, n_hidden,
    lr=0.05, n_epoches=10, batch_size=20):
    # init rbm object
    rbm = GBRBM(n_visible=n_visible, n_hidden=n_hidden, \
                learning_rate=lr, momentum=0.95, err_function='mse', \
                use_tqdm=False, sample_visible=False, sigma=1)
    rbm.fit(input_data, n_epoches=n_epoches, batch_size=batch_size, \
            shuffle=True, verbose=True)
    embeddings = rbm.transform(input_data).round().astype(int)
    return rbm, embeddings

def superv_gbrbm_embeddings(input_data, label_data, n_visible, n_hidden,
    init_w=None, init_vbias=None, init_hbias=None,
    superv_lr=0.1, n_epoches=10, batch_size=20):
    target_set = list(set(label_data))
    print(target_set)
    superv_rbm = NNSupervRBM(n_visible=n_visible, n_hidden=n_hidden, target_set=target_set, \
                             init_w=init_w, init_vbias=init_vbias, init_hbias=init_hbias, \
                             superv_lr=superv_lr)
    superv_rbm.superv_fit(input_data, label_data,
                          n_epoches=n_epoches, batch_size=batch_size, shuffle=True)
    embeddings = superv_rbm.transform(input_data).round().astype(int)
    return superv_rbm, embeddings



if __name__ == "__main__":
    dict_name   = "resource/dict/trigram_dict"
    corpus_name = "resource/corpus/trigram.doc.tfidf.corpus"
    info_name   = "data/56+446.info.txt"

    alpha       = 0.2

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
        location = alpha * (location - min_vals) / (max_vals - min_vals)

    # concatenate corpus and location into one single matrix
    input_data = np.concatenate((dense_corpus, location), axis=1)

    # get labels info
    label_data = None
    with open(info_name, "r") as f:
        # fetch labels
        label_data = np.array([ line.strip().split("\t")[1] for line in f ])

    # generate gbrbm embeddings and get fitted model
    rbm, embeddings = gbrbm_embeddings(input_data,
        n_visible=len(ngram_dict)+2,
        n_hidden=1000, n_epoches=10)

    # remove random data records
    indices = [ ind for ind in range(len(label_data)) if label_data[ind] != "random" ]
    init_w, init_vbias, init_hbias = rbm.get_weights()
    new_input_data = input_data[indices]
    new_label_data = label_data[indices]
    superv_rbm, new_embeddings = superv_gbrbm_embeddings(
        new_input_data, new_label_data,
        n_visible=len(ngram_dict)+2, n_hidden=1000,
        init_w=init_w, init_vbias=init_vbias, init_hbias=init_hbias,
        superv_lr=0.005, n_epoches=10, batch_size=20)

    # # plot weight matrix of rbm
    # weights_matrix, vbias_array, hbias_array = rbm.get_weights()
    # bias_matrix = np.row_stack((vbias_array[-50:], hbias_array[-50:]))
    # mat2img(bias_matrix)
    #
    # save embeddings
    file_name="corpus_location_supervised"
    np.savetxt("resource/embeddings/%s.txt" % file_name, new_embeddings, delimiter=',')
