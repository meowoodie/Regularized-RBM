#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import numpy as np
import tensorflow as tf

from model import Skipgram
from context import Context

if __name__ == "__main__":
    context_file_path = "data/info.txt"

    c  = Context(context_file_path)
    sc = c.spatial_context(delta=0.02)
    # sc = c.true_context()

    for pair in sc[0:5]:
        print pair

    input_data, label_data = c.target_context_pairs(sc)
    init_embed = np.loadtxt("resource/embeddings/docs/trigram-tfidf-vecs.txt", delimiter=",")
    event_size = init_embed.shape[0]
    embed_size = init_embed.shape[1]
    print "embed size", embed_size
    print "data size", len(input_data)

    event2vec = Skipgram(event_size, embed_size, init_embed,
                          batch_size=128, num_sampled=128, iters=100000, display_step=10)
    with tf.Session() as sess:
        event2vec.train(sess, input_data, label_data)
        embeddings = event2vec.get_embeddings(sess)
        np.savetxt("resource/embeddings/events/spatial-trigram-tfidf-vecs.txt", embeddings, delimiter=',')
        print embeddings
        print embeddings.shape
