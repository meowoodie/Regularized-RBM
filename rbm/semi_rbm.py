#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script contains class SupervMixedRBM and a simple example of its
implementation.
"""

from __future__ import print_function

from .util import tf_xavier_init
import tensorflow as tf
import numpy as np
import sys

# TODO: comment this line in production environment
# tf.set_random_seed(100)

class SemiSupervRBM:
    """
    """

    def __init__(self,
                 n_y, # number of bernoulli visible units
                 n_x, # number of gaussian visible units
                 n_h, # number of bernoulli hidden units
                 sample_visible=False,
                 learning_rate=0.01,
                 momentum=0.95,
                 xavier_const=1.0,
                 err_function='mse'):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        # configurations
        self.n_y = n_y
        self.n_x = n_x
        self.n_h = n_h
        self.learning_rate  = learning_rate
        self.momentum       = momentum
        self.sample_visible = sample_visible

        # input parameters
        self.x = tf.placeholder(tf.float32, [None, self.n_bvisible])
        self.y = tf.placeholder(tf.float32, [None, self.n_gvisible])
        self.h = tf.placeholder(tf.float32, [None, self.n_hidden])

        # variables of mixed-rbm
        self.y_w = tf.Variable(tf_xavier_init(self.n_y, self.n_h, const=xavier_const), dtype=tf.float32)
        self.x_w = tf.Variable(tf_xavier_init(self.n_x, self.n_h, const=xavier_const), dtype=tf.float32)
        self.y_b = tf.Variable(tf.zeros([self.n_y]), dtype=tf.float32)
        self.x_b = tf.Variable(tf.zeros([self.n_x]), dtype=tf.float32)
        self.h_b = tf.Variable(tf.zeros([self.n_h]), dtype=tf.float32)
        self.x_sigma = 1. # TODO: change fixed sigma to tensor variable

        # variables of weights updates
        self.delta_y_w = tf.Variable(tf.zeros([self.n_bvisible, self.n_hidden]), dtype=tf.float32)
        self.delta_x_w = tf.Variable(tf.zeros([self.n_gvisible, self.n_hidden]), dtype=tf.float32)
        self.delta_y_b = tf.Variable(tf.zeros([self.n_bvisible]), dtype=tf.float32)
        self.delta_x_b = tf.Variable(tf.zeros([self.n_gvisible]), dtype=tf.float32)
        self.delta_h_b = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.update_weights  = None
        self.update_deltas   = None
        self.compute_hidden  = None
        self.compute_visible = None
        self.compute_visible_from_hidden = None

        self._initialize_vars()

        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None

        # if err_function == 'cosine':
        #     x1_norm = tf.nn.l2_normalize(self.x, 1)
        #     x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
        #     cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
        #     self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        # else:
        #     self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))

        # init all defined variables before training
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_vars(self):
        """
        This function defines conditional probability of h|v, and reconstruction
        conditional probability of v|h and h|v.
        """

        hidden_p         = tf.nn.sigmoid(tf.matmul(self.bx, self.bw) + tf.matmul(self.gx, self.gw / self.gvisible_sigma) + self.hidden_bias)
        bvisible_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.bw)) + self.bvisible_bias)
        gvisible_recon_p = tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.gw)) + self.gvisible_bias
        # give gvisible_recon_p a gaussian random noise if sample_visible is set True
        if self.sample_visible:
            gvisible_recon_p = sample_gaussian(gvisible_recon_p, self.gvisible_sigma)

        hidden_recon_p = tf.nn.sigmoid(tf.matmul(bvisible_recon_p, self.bw) + tf.matmul(gvisible_recon_p, self.gw) + self.hidden_bias)

        positive_grad  = tf.matmul(tf.transpose(self.x), hidden_p)
        negative_grad  = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        # training momentum
        def f(x_old, x_new):
            return self.momentum * x_old + \
                   self.learning_rate * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])

        delta_w_new            = f(self.delta_w, positive_grad - negative_grad)
        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_p, 0))
        delta_hidden_bias_new  = f(self.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias
        self.compute_visible_from_hidden = tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias

    def get_err(self, batch_x):
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x):
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.x: batch_x})

    def fit(self,
        data_x, data_t,
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
                self.partial_fit(batch_x, batch_t) # supervised fitting partially
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

    def get_weights(self):
        return self.sess.run(self.w),\
            self.sess.run(self.bvisible_bias),\
            self.sess.run(self.gvisible_bias),\
            self.sess.run(self.hidden_bias)

    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w':  self.w,
                                name + '_bv': self.bvisible_bias,
                                name + '_gv': self.gvisible_bias,
                                name + '_h':  self.hidden_bias})
        return saver.save(self.sess, filename)

    def set_weights(self, w, bvisible_bias, gvisible_bias, hidden_bias):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.bvisible_bias.assign(bvisible_bias))
        self.sess.run(self.gvisible_bias.assign(gvisible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w':  self.w,
                                name + '_bv': self.bvisible_bias,
                                name + '_gv': self.gvisible_bias,
                                name + '_h':  self.hidden_bias})
        saver.restore(self.sess, filename)
