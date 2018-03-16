#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import sys
import argparse
import numpy as np
import random

np.random.seed(100)

n_total      = 155991
n_sampling   = 2000
rand_indices = random.sample(range(n_total), n_sampling)


with open("data/new_dataset/e9999.unlabeled.corpus.txt", "r") as fr1,\
     open("data/new_dataset/rand2k.e9999.unlabeled.corpus.txt", "w") as fw1,\
     open("data/new_dataset/e9999.unlabeled.info.txt", "r") as fr2,\
     open("data/new_dataset/rand2k.e9999.unlabeled.info.txt", "w") as fw2:
    ind = 0
    for line in fr1:
        line = line.strip("\n")
        if ind in rand_indices:
            fw1.write(line+"\n")
        ind += 1

    ind = 0
    for line in fr2:
        line = line.strip("\n")
        if ind in rand_indices:
            fw2.write(line+"\n")
        ind += 1

# indices = []
#
# ind = 0
# with open("data/new_dataset/unlabeled.info.txt", "r") as fr,\
#      open("data/new_dataset/e9999.unlabeled.info.txt", "w") as fw:
#     for line in fr:
#         catagory = line.strip("\n").split("\t")[1].strip()
#         if catagory != "9999" and catagory != "":
#             indices.append(ind)
#             fw.write(line)
#         ind += 1
# ind = 0
# with open("data/new_dataset/unlabeled.corpus.txt", "r") as fr,\
#      open("data/new_dataset/unlabeled.e9999.corpus.txt", "w") as fw:
#     for line in fr:
#         line = line.strip("\n")
#         if ind in indices:
#             fw.write(line+"\n")
#         ind += 1
