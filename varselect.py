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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import arrow
import sys

from corpus import corpus_by_documents, sub_dictionary
from utils.mat2img import mat2img
from utils.vec2tsne import vec2tsne
from utils.eval4vec import eval_by_cosine
from rbm import GBRBM
# from rbm import SemiSupervRBM

# row ids of random documents
# RANOM_INDICES = [
#     565,  1332, 1969, 1160, 2019, 1077, 1917,  910, 1024,  816, 1830,  382,  999, 1748, 1759, 1525,  757,
#     1392, 1744, 1236,  751,  887, 1278, 1421, 1538, 1279, 2047, 1089, 1412,  211, 1655,  327, 1963, 1343,
#     2050, 302,  1144,  978, 1747, 1265,  258, 1943, 881,  943, 1143,  472,  819,  479, 1542, 1398,  489,
#     348,  1859,   73, 1615,  256,  727, 1590,  205, 1078, 207, 1288,  776,  931, 1241,  305, 1534, 1714,
#     1346,  861,  756, 1510, 1246, 1222,  821, 1436, 1797, 1272, 1030, 1296,  696,  321, 1301,  904, 1046,
#     1749,  968,  395,  243,  612,   88,  309,  411, 1583, 1448, 1813, 1427, 2003, 1003, 1389,  116,  404,
#     1891, 1414,  361,  345,  196,  341,  991,  729, 1388, 1259,  450,  437,  764, 1076, 1690,  505,  891,
#     1721,  844,  853,  522,  471,  962, 1298,  286,  682,  468, 1792,  391,   93, 1057, 1459,  475, 1235,
#     1846, 1990, 1114, 1937, 1536,  223,  973,  863, 1335, 1422, 1992,  710, 1066,  752, 1670,  871,  584,
#     534, 1367,  191,  969, 1417,  579, 1945,  511,   75, 1672, 1456, 1239, 1709,  615,  920,  540, 1519,
#     1807, 1707,  809,   86,  365,  136,  996, 1826,  148, 1377, 1486, 1375, 1975, 1743,  332, 1930, 1625,
#     1247,  545,  453,  717, 1880, 1022, 1349,  523, 1993, 1175, 1639, 1867,  344]
RANOM_INDICES = [
    208,1704,1953,607,408,137,426,1348,115,86,851,1688,817,826,596,942,977,1142,
    1389,293,502,1629,263,1473,500,599,1066,442,936,1390,544,1828,1807,1671,1314,
    1919,1432,1726,824,837,1223,1109,1984,456,1721,255,814,159,367,1581,839,774,
    890,696,1144,1959,1154,1185,1315,1669,1524,912,1970,1146,639,378,345,1523,
    755,1545,675,1448,1242,809,678,1905,1585,1419,372,1636,1110,1754,307,535,
    1446,571,175,1749,794,264,1356,364,383,768,1166,1013,1531,777,661,723,987,
    1667,933,1702,1351,1998,1922,439,347,1011,1522,617,1401,227,247,1174,2056,
    1295,1243,1809,1306,1731,1647,1838,1578,1304,1153,1297,1563,1134,1568,470,
    1890,394,99,1728,1625,1263,1697,1176,627,1631,1182,2051,689,1149,862,1678,
    919,858,556,1973,258,224,717,1360,1355,192,860,530,993,1044,256,744,840,1014,
    1680,1816,968,1255,1561,545,855,1454,1653,321,606,821,375,1375,1247,609,365,
    581,1550,996,1070,538,1609,726,732,1551,1289,157,509,465,801,1870,352,505]

# since above hardcode data comes from R, which means array starts from 1 instead of 0
RANOM_INDICES = [ ind-1 for ind in RANOM_INDICES ]
# row ids of labeled documents
LABEL_INDICES = np.arange(0,69,1).tolist() # np.arange(0,56,1).tolist()

# key ids of terms in labeld documents
# BURGLARY_TERMS = [162, 950, 960, 1304, 1698, 1709, 1716, 2412, 2701, 2795, 3388, 3413, 3624, 3726, 4243, 4468, 4760, 4932, 5271, 5476, 5525, 5762, 5820, 5920, 6853]
# PEDROB_TERMS   = [1076, 1908, 3067, 4079, 6471]
# ADAMS_TERMS    = [1025, 1697, 1754, 1891, 3501, 3551, 3866, 3914, 5304, 6314]
# MORRI_TERMS    = [4313, 5299, 5653, 6023, 6040, 6134, 6855, 7021]
# TUCKR_TERMS    = [226, 477, 484, 1499, 1943, 1946, 3067, 3143, 3621, 4096, 4313, 4433, 4874, 6134, 6207]
# TODD_TERMS     = [234, 1433, 3058, 4313, 4981, 5513, 5661, 6288, 6801, 7019]
BURGLARY_TERMS = [9,180,1715,1726,1733,2438,2731,3080,3767,4289,4778,4987,5540,5589,5829,5888,6927]
PEDROB_TERMS   = [643,1088]
ADAMS_TERMS    = [1037,1771,1908,3100,6623]
MORRI_TERMS    = [540]
TUCKR_TERMS    = [1516,2583,2788,2957,3104,4188,5250,6492,6805]
TODD_TERMS     = [5140,5577]


# row ids of preserved documents in corpus
PRESERV_DOCS  = LABEL_INDICES + RANOM_INDICES
# key ids of preserved terms in specified vocabulary
PRESERV_TERMS = [] # BURGLARY_TERMS + PEDROB_TERMS + ADAMS_TERMS + MORRI_TERMS + TUCKR_TERMS + TODD_TERMS

def plot_rates(df, time_name="Number of Noise Terms", value_name="Hit Rate", \
               unit_name="Iteration Id", condition_name="Number of Results",
               plot_path="results/hit_rates.pdf"):
    # plot as a pdf file
    with PdfPages(plot_path) as pdf:
        fig, ax = plt.subplots(1, 1)
        sns.tsplot(time=time_name, value=value_name, \
                   unit=unit_name, condition=condition_name, data=df)
        pdf.savefig(fig)

def exp_variable_selection(dict_name, corpus_name, N=2, n_noise_term=10, n_epoches=20, \
                           learning_rate=.001, batch_size=30, n_hidden=50):
    """
    Main function for selecting variables and calculating embeddings for selected
    embeddings vectors by using vanilla RBM.
    """
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
    # mat2img(np.log(corpus_slice))

    rbm = GBRBM(n_visible=corpus_slice.shape[1], n_hidden=n_hidden, \
                learning_rate=learning_rate, momentum=0.95, err_function='mse', \
                use_tqdm=False, sample_visible=False, sigma=1.)
    rbm.fit(corpus_slice, n_epoches=n_epoches, batch_size=batch_size, \
            shuffle=True, verbose=True)
    embeddings = rbm.transform(corpus_slice).round().astype(int)
    # w, vbias, hbias = rbm.get_weights()
    # mat2img(w)
    return corpus_slice, embeddings

    # # save embeddings
    # file_name="sub.2k.corpus"
    # np.savetxt("resource/embeddings/%s.txt" % file_name, embeddings, delimiter=',')

if __name__ == "__main__":

    params = {
        "n_noise_term":  [(i+1)*200 for i in range(4)+2], # [0,    5,    10,   15,   20,   25,   30,   35,   40,   45,   50],
        "n_epoches":     [100 for i in range(4)] + [200 for i in range(0)], # [100,  100,  100,  100,  200,  200,  200,  200,  200,  200,  200],
        "learning_rate": [1e-2 for i in range(4)], # [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
        "batch_size":    [30 for i in range(4)], # [30,   30,   30,   30,   30,   30,   30,   30,   30,   30,   30],
        "n_hidden":      [50 for i in range(4)], # [50,   50,   50,   50,   50,   50,   50,   50,   50,   50,   50]
    }
    N            = 2  # N for n-gram
    Ks           = [20, 40, 60, 80]
    iters        = 100
    label_inds   = range(69) # range(56)
    # path for resource
    dict_name   = "resource/dict/2069.bigram.dict"
    corpus_name = "resource/corpus/2069.bigram.doc.tfidf.corpus"
    label_path  = "data/2000+69.dataset/2069.info.txt"

    # load labels
    labels = []
    with open(label_path, "r") as fhandler:
        for line in fhandler:
            doc_ind  = line.strip().split("\t")[0]
            catagory = line.strip().split("\t")[1]
            labels.append(catagory)

    # EXP RATE PLOTTING
    # raw experiment results
    exp_data = {
        "Number of Results": [], "Hit Rate": [],
        "Iteration Id": [], "Number of Noise Terms": []}
    # iteratively repeat the same experiments multiple times
    for j in range(iters):
        print("calculation iter %d..." % j, file=sys.stderr)
        # iteratively do experiments over all the parameters
        for i in range(len(params.values()[0])):
            # exp: variable selection
            corpus_slice, embeddings = exp_variable_selection(
                dict_name, corpus_name, n_hidden=params["n_hidden"][i],
                N=2, n_noise_term=params["n_noise_term"][i], n_epoches=params["n_epoches"][i],
                learning_rate=params["learning_rate"][i], batch_size=params["batch_size"][i])

            hit_rates = [
                eval_by_cosine(embeddings, labels, label_inds=label_inds, top_k=k, type="avg_rate")
                for k in Ks ]

            exp_data["Number of Results"]     += Ks
            exp_data["Hit Rate"]              += hit_rates
            exp_data["Iteration Id"]          += [ j for ki in range(len(Ks)) ]
            exp_data["Number of Noise Terms"] += [ params["n_noise_term"][i] for ki in range(len(Ks)) ]

    exp_df = pd.DataFrame(data=exp_data)
    exp_df.to_pickle("df_%d_to_%d_iter_%d" % (params["n_noise_term"][0], params["n_noise_term"][1], iters))
    # exp_df = pd.read_pickle("exp_data_frame")
    plot_rates(exp_df)

    # # UNIT TEST ON EXP_VARIABLE_SELECTION
    # # parameters
    # n_hidden = 50
    # n_noise  = 0
    # n_epoch  = 150
    # n_batch  = 30
    # # name of the plot
    # plot_name = "2069_hid%d_noise%d_epoch%d_bat%d" % \
    #     (n_hidden, n_noise, n_epoch, n_batch)
    # # exp: variable selection
    # corpus_slice, embeddings = exp_variable_selection(
    #     dict_name, corpus_name, n_hidden=n_hidden,
    #     N=2, n_noise_term=n_noise, n_epoches=n_epoch,
    #     learning_rate=1e-3, batch_size=n_batch)
    # # path of the plot
    # plot_path = "results/%s.pdf" % plot_name
    # # plot the embeddings results
    # # vec2tsne(label_path, plot_path, vectors=embeddings, n=2)
    # hit_rate = eval_by_cosine(embeddings, labels, label_inds=label_inds, top_k=10, type="avg_rate")
    # print(hit_rate)
