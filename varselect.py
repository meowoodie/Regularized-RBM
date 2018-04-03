#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the main script for testing algorithm on real dataset, which includes
helpful function for data preparing, model training, and visualizing.
"""
from __future__ import print_function
from gensim.corpora.mmcorpus import MmCorpus
from gensim.matutils import corpus2dense
from gensim import corpora
import numpy as np
import random
import arrow
import sys

from corpus import corpus_by_documents, sub_dictionary
from tfrbm import GBRBM
# from rbm import SemiSupervRBM
# from rbm import GBRBM

# row ids of random documents
RANOM_INDICES = [
    565,  1332, 1969, 1160, 2019, 1077, 1917,  910, 1024,  816, 1830,  382,  999, 1748, 1759, 1525,  757,
    1392, 1744, 1236,  751,  887, 1278, 1421, 1538, 1279, 2047, 1089, 1412,  211, 1655,  327, 1963, 1343,
    2050, 302,  1144,  978, 1747, 1265,  258, 1943, 881,  943, 1143,  472,  819,  479, 1542, 1398,  489,
    348,  1859,   73, 1615,  256,  727, 1590,  205, 1078, 207, 1288,  776,  931, 1241,  305, 1534, 1714,
    1346,  861,  756, 1510, 1246, 1222,  821, 1436, 1797, 1272, 1030, 1296,  696,  321, 1301,  904, 1046,
    1749,  968,  395,  243,  612,   88,  309,  411, 1583, 1448, 1813, 1427, 2003, 1003, 1389,  116,  404,
    1891, 1414,  361,  345,  196,  341,  991,  729, 1388, 1259,  450,  437,  764, 1076, 1690,  505,  891,
    1721,  844,  853,  522,  471,  962, 1298,  286,  682,  468, 1792,  391,   93, 1057, 1459,  475, 1235,
    1846, 1990, 1114, 1937, 1536,  223,  973,  863, 1335, 1422, 1992,  710, 1066,  752, 1670,  871,  584,
    534, 1367,  191,  969, 1417,  579, 1945,  511,   75, 1672, 1456, 1239, 1709,  615,  920,  540, 1519,
    1807, 1707,  809,   86,  365,  136,  996, 1826,  148, 1377, 1486, 1375, 1975, 1743,  332, 1930, 1625,
    1247,  545,  453,  717, 1880, 1022, 1349,  523, 1993, 1175, 1639, 1867,  344]
# since above hardcode data comes from R, which means array starts from 1 instead of 0
RANOM_INDICES = [ ind-1 for ind in RANOM_INDICES ]
# row ids of labeled documents
LABEL_INDICES = np.arange(0,56,1).tolist()

# key ids of terms in labeld documents
BURGLARY_TERMS = [162, 950, 960, 1304, 1698, 1709, 1716, 2412, 2701, 2795, 3388, 3413, 3624, 3726, 4243, 4468, 4760, 4932, 5271, 5476, 5525, 5762, 5820, 5920, 6853]
PEDROB_TERMS   = [1076, 1908, 3067, 4079, 6471]
ADAMS_TERMS    = [1025, 1697, 1754, 1891, 3501, 3551, 3866, 3914, 5304, 6314]
MORRI_TERMS    = [4313, 5299, 5653, 6023, 6040, 6134, 6855, 7021]
TUCKR_TERMS    = [226, 477, 484, 1499, 1943, 1946, 3067, 3143, 3621, 4096, 4313, 4433, 4874, 6134, 6207]
TODD_TERMS     = [234, 1433, 3058, 4313, 4981, 5513, 5661, 6288, 6801, 7019]

# row ids of preserved documents in corpus
PRESERV_DOCS  = RANOM_INDICES + LABEL_INDICES
# key ids of preserved terms in specified vocabulary
PRESERV_TERMS = BURGLARY_TERMS + PEDROB_TERMS + ADAMS_TERMS + MORRI_TERMS + TUCKR_TERMS + TODD_TERMS

if __name__ == "__main__":
    N            = 2  # N for n-gram
    n_noise_term = 10 # number of noise ngram terms
    # path for resource
    dict_name   = "resource/dict/2k.bigram.dict"
    corpus_name = "resource/corpus/2k.bigram.doc.tfidf.corpus"

    # load existing dictionary (or creat a new dictionary from scratch)
    # code for creating new dictionary ...
    ngram_dict    = corpora.Dictionary.load(dict_name)
    # select key ides of some random ngram terms from loaded dictionary as dictionary noise
    random_terms = list(set(ngram_dict.keys()) - set(PRESERV_TERMS))
    noise_terms  = random.sample(random_terms, n_noise_term)
    print("[%s] [Var Select] %d noise terms has been added: %s" % \
         (arrow.now(), len(noise_terms), [ngram_dict[key] for key in noise_terms]), file=sys.stderr)

    # # shrink dictionary to a subset in accordance with PRESERV_TERMS
    # sub_ngram_dict = sub_dictionary(ngram_dict, PRESERV_TERMS, by_key=True)

    # load existing corpus
    corpus       = MmCorpus(corpus_name)
    dense_corpus = corpus2dense(corpus, num_terms=len(ngram_dict)).transpose()
    print("[%s] [Var Select] raw corpus has been loaded with size (%d, %d)" % \
         (arrow.now(), dense_corpus.shape[0], dense_corpus.shape[1]), file=sys.stderr)
    # slice the corpus by PRESERV_TERMS and corpus
    # (remove columns which are not included in PRESERV_TERMS)
    # noted: indexing arrays could not be broadcast together
    # e.g. dense_corpus[PRESERV_DOCS, PRESERV_TERMS]
    corpus_slice = dense_corpus[:, PRESERV_TERMS + noise_terms]
    corpus_slice = corpus_slice[PRESERV_DOCS, :]
    print("[%s] [Var Select] corpus has been sliced with size (%d, %d)" % \
         (arrow.now(), corpus_slice.shape[0], corpus_slice.shape[1]), file=sys.stderr)

    rbm = GBRBM(n_visible=corpus_slice.shape[1], n_hidden=30, \
                learning_rate=.05, momentum=0.95, err_function='mse', \
                use_tqdm=False, sample_visible=False, sigma=1.)
    rbm.fit(corpus_slice, n_epoches=30, batch_size=10, \
            shuffle=True, verbose=True)
    embeddings = rbm.transform(corpus_slice).round().astype(int)

    # save embeddings
    file_name="sub.2k.corpus"
    np.savetxt("resource/embeddings/%s.txt" % file_name, embeddings, delimiter=',')
