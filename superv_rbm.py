#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
model wrappers that focus on the sparsification of model weights by supervised
learning the groud truth of real data. For here two implementations for RBM
(Restricted Boltzmann Machine) have been provided, including using simple neural
network and skip-gram for supervising the learning of RBM.
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
from tfrbm import GBRBM

class SupervRBM(GBRBM):
    """
    An abstract class for supervised learning all kinds of RBMs, it provides a
    framework for learning specific RBM with target labeling (self.t). You have
    to override the function '_initialize_vars', 'partial_superv_fit' and
    'get_superv_err' to implement the structure of your supervised model.
    Usually the structure could be a simple neural network or skipgram which
    takes the transformation (hidden output) of visible layer in RBM as input,
    and minimize the error of prediction of target labeling.
    """

    def __init__(self, n_visible, n_hidden, target_set,
        init_w=None, init_vbias=None, init_hbias=None, superv_lr=0.1,
        sample_visible=False, sigma=1, **kwargs):
        # initialize additional input variables
        self.target_set = target_set
        self.init_w     = init_w
        self.init_vbias = init_vbias
        self.init_hbias = init_hbias
        self.superv_lr  = superv_lr
        # initialize the computational graph of GBRBM
        GBRBM.__init__(self, n_visible, n_hidden, sample_visible, sigma, **kwargs)

    def _initialize_vars(self):
        GBRBM._initialize_vars(self)
        # initialize the weights
        if (self.init_w is not None) and \
           (self.init_vbias is not None) and \
           (self.init_hbias is not None):
            self.set_weights(self.init_w, self.init_vbias, self.init_hbias)
        # define your structure of supervised model below.

    def partial_superv_fit(self, batch_x, batch_t):
        """
        An abstract method that needs to be overrided in subclass for partially
        supervised learning batch data. In addition to batch_x, batch_t is
        provided as target labelings.
        """
        pass

    def get_superv_acc(self, batch_x, batch_y):
        """
        An abstract method that needs to be overrided in subclass for getting
        partially accuracy for supervised learning batch data. In addition to
        batch_x, batch_t is provided as target labelings.
        """
        pass

    def superv_fit(self, data_x, data_t,
        n_epoches=10, batch_size=10, shuffle=True):
        """
        A customized fitting method for supervised learning RBM. There are only
        several minor changes compared to 'fit' method in tfrbm in order to
        incorporate target labelings.
        """
        assert n_epoches  > 0
        assert batch_size > 0
        assert len(data_t) == len(data_x)

        # number of data records
        n_data    = data_x.shape[0]
        # number of batches
        n_batches = n_data / batch_size + (0 if n_data % batch_size == 0 else 1)

        # prepare for shuffling the dataset
        if shuffle:
            data_x_cpy = data_x.copy()
            data_t_cpy = data_t.copy()
            inds = np.arange(n_data)
        else:
            data_x_cpy = data_x
            data_t_cpy = data_t

        # logging the training errors
        # errs = []
        # iterate training epoches
        for e in range(n_epoches):
            # shuffle dataset
            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]
                data_t_cpy = data_t_cpy[inds]

            # init the array of errors of each epoches
            epoch_errs = np.zeros((n_batches,))
            epoch_accs = np.zeros((n_batches,))
            epoch_ind  = 0
            # iterate each batch of dataset
            for b in range(n_batches):
                batch_x = data_x_cpy[b*batch_size:(b+1)*batch_size]
                batch_t = data_t_cpy[b*batch_size:(b+1)*batch_size]
                self.partial_superv_fit(batch_x, batch_t) # supervised fitting partially
                batch_err = self.get_err(batch_x)         # get errors after one batch training
                batch_acc = self.get_superv_acc(batch_x, batch_t)
                epoch_errs[epoch_ind] = batch_err
                epoch_accs[epoch_ind] = batch_acc
                epoch_ind += 1

            # get mean of errors in this epoch
            err_mean = epoch_errs.mean()
            acc_mean = epoch_accs.mean()
            print("Epoch: {:d}".format(e), file=sys.stderr)
            print("Train error: {:.4f}".format(err_mean), file=sys.stderr)
            print("Train acc: {:.4f}".format(acc_mean), file=sys.stderr)
            # errs = np.hstack([errs, epoch_errs])
        # return errs



class NNSupervRBM(SupervRBM):
    """
    Using simple neural network as supervised structure to learn RBM.
    Noted: please do not include any random cases (without target labelings) in
    dataset.
    """

    def __init__(self, n_visible, n_hidden, target_set,
        init_w=None, init_vbias=None, init_hbias=None, superv_lr=0.1,
        sample_visible=False, sigma=1, **kwargs):
        # initialize the computational graph of GBRBM
        SupervRBM.__init__(self, n_visible, n_hidden, target_set,
            init_w, init_vbias, init_hbias, superv_lr,
            sample_visible, sigma, **kwargs)

    def _initialize_vars(self):
        SupervRBM._initialize_vars(self)
        # initialize additional input variables for supervised structure
        self.t = tf.placeholder(tf.float32, None)
        # supervised structure
        n_hidden_1 = 256
        embeddings = self.compute_hidden
        # hidden fully connected layer with 256 neurons
        layer_1    = tf.layers.dense(embeddings, n_hidden_1)
        # add any number of layers as you want here
        out_layer  = tf.layers.dense(layer_1, len(self.target_set))
        pred_class = tf.argmax(out_layer, axis=1)
        pred_prob  = tf.nn.softmax(out_layer)
         # define loss and optimizer
        self.superv_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=out_layer, labels=tf.cast(self.t, dtype=tf.int32)))
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.superv_lr)
        self.superv_optimizer = optimizer.minimize(
            self.superv_loss,
            global_step=tf.train.get_global_step())
        # evaluate the accuracy of the model
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(pred_class, tf.cast(self.t, tf.int64)), tf.float32))

    def partial_superv_fit(self, batch_x, batch_t):
        """
        """
        # TODO: is this step necessary for updating weights in rbm?
        # unsupervised update weights of rbm by normal fitting method
        self.partial_fit(batch_x)
        # convert batch_t from labelings to enumerate values
        new_batch_t = np.array([ self.target_set.index(t) for t in batch_t ])
        # supervised optimization
        self.sess.run(self.superv_optimizer,
            feed_dict={self.x: batch_x, self.t: new_batch_t})

    def get_superv_acc(self, batch_x, batch_t):
        """
        """
        # convert batch_t from labelings to enumerate values
        new_batch_t = np.array([ self.target_set.index(t) for t in batch_t ])
        return self.sess.run(self.acc,
            feed_dict={self.x: batch_x, self.t: new_batch_t})

# if __name__ == "__main__":



        # convert batch_t from labelings to one-hot vectors
        # new_batch_t = np.zeros((len(batch_t), len(self.target_set)))
        # for ind in range(len(batch_t)):
        #     new_batch_t[ind][self.target_set.index(batch_t[ind])]

        # only keep data with target labelings in this batch
        # indices = [ ind
        #     for ind in range(len(batch_t))
        #     if batch_t[ind] in self.target_set ]
        # batch_x     = batch_x[indices]
        # batch_t     = batch_t[indices]
        # new_batch_t = np.zeros((len(indices), len(self.target_set)))
        # for ind in range(len(indices)):
        #     new_batch_t[ind][self.target_set.index(batch_t[ind])]
