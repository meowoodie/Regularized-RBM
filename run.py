#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""
import numpy as np

from gensim import corpora
from gensim.matutils import corpus2dense

from tfrbm import GBRBM

def gbrbm_embeddings(input_data,
    n_visible, n_hidden, file_name,
    lr=0.05, n_epoches=10, batch_size=20):
    # Init rbm object
    rbm = GBRBM(n_visible=n_visible, n_hidden=n_hidden, \
                learning_rate=lr, momentum=0.95, err_function='mse', \
                use_tqdm=False, sample_visible=False, sigma=1)
    rbm.fit(input_data, n_epoches=n_epoches, batch_size=batch_size, \
            shuffle=True, verbose=True)
    embeddings = rbm.transform(input_data).round().astype(int)
    np.savetxt("resource/embeddings/%s.txt" % file_name, embeddings, delimiter=',')



if __name__ == "__main__":
    dict_name   = "resource/dict/trigram_dict"
    corpus_name = "resource/corpus/trigram.doc.tfidf.corpus"
    info_name   = "data/56+446.info.txt"

    ngram_dict   = corpora.Dictionary.load(dict_name)
    corpus_tfidf = corpora.MmCorpus(corpus_name)

    dense_corpus = corpus2dense(corpus_tfidf, num_terms=len(ngram_dict)).transpose()
    print(dense_corpus.shape)

    location = None
    with open(info_name, "r") as f:
        location = [ map(float, line.strip().split("\t")[3:]) for line in f ]
        location = np.array(location)

    input_data = np.concatenate((dense_corpus, location), axis=1)

    gbrbm_embeddings(input_data, n_visible=len(ngram_dict)+2, n_hidden=1000,
        file_name="corpus_location", n_epoches=10)
