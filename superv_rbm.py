#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
model wrappers that focus on the sparsification of model weights by supervised
learning the groud truth of real data. For here two implementations for RBM
(Restricted Boltzmann Machine) have been provided, including using simple neural
network and skip-gram for supervising the learning of RBM.
"""

from __future__ import print_function

class SupervRBM(GBRBM):
    """
    An abstract class for supervised learning all kinds of RBMs, it provides a
    framework for learning specific RBM with target labeling (self.t). You have
    to override the function '__init__' and 'partial_superv_fit' to implement
    the structure of your supervised model. Usually the structure could be a
    simple neural network or skipgram which takes the transformation (hidden
    output) of visible layer in RBM as input, and minimize the error of
    prediction of target labeling.
    """

    def __init__(self, n_visible, n_hidden, target_set,
        init_w=None, init_vbias=None, init_hbias=None,
        sample_visible=False, sigma=1, **kwargs):
        # initialize the computational graph of GBRBM
        GBRBM.__init__(self, n_visible, n_hidden, sample_visible, sigma, **kwargs)
        # initialize the weights
        if init_w and init_vbias and init_hbias:
            self.set_weights(init_w, init_vbias, init_hbias)
        # initialize additional input variables
        self.target_set = target_set
        # define your structure of supervised model below.

    def partial_superv_fit(self, batch_x, batch_t):
        """
        An abstract method that needs to be overrided in subclass for partially
        supervised learning batch data. In addition to batch_x, batch_t is
        provided as target labelings.
        """
        pass

    def superv_fit(self, data_x, data_t,
        n_epoches=10, batch_size=10,
        shuffle=True, verbose=True):
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
        errs = []
        # iterate training epoches
        for e in range(n_epoches):
            # shuffle dataset
            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]
                data_t_cpy = data_t_cpy[inds]

            # init the array of errors of each epoches
            epoch_errs     = np.zeros((n_batches,))
            epoch_errs_ind = 0
            # iterate each batch of dataset
            for b in range(n_batches):
                batch_x = data_x_cpy[b*batch_size:(b+1)*batch_size]
                batch_t = data_t_cpy[b*batch_size:(b+1)*batch_size]
                self.partial_superv_fit(batch_x, batch_t) # supervised fitting partially
                batch_err = self.get_err(batch_x)         # get errors after one batch training
                epoch_errs[epoch_errs_ind] = batch_err
                epoch_errs_ind += 1

            # get mean of errors in this epoch
            err_mean = epoch_errs.mean()
            print("Epoch: {:d}".format(e), file=sys.stderr)
            print("Train error: {:.4f}".format(err_mean), file=sys.stderr)
            errs = np.hstack([errs, epoch_errs])

        return errs



class NNSupervRBM(SupervRBM):
    """
    Using simple neural network as supervised structure to learn RBM.
    Noted: discard random cases (without target labelings) in dataset.
    """

    def __init__(self, n_visible, n_hidden, target_set,
        init_w, init_vbias, init_hbias,
        sample_visible=False, sigma=1, **kwargs):
        # initialize the computational graph of GBRBM
        SupervRBM.__init__(self, n_visible, n_hidden, target_set,
            init_w, init_vbias, init_hbias, sample_visible, sigma, **kwargs)
        # initialize additional input variables
        self.t = tf.placeholder(tf.float32, [None, len(self.target_set)])


    def partial_superv_fit(self, batch_x, batch_t):
        """
        """
        # TODO: is this step necessary for updating weights in rbm?
        # unsupervised update weights of rbm by normal fitting method
        self.partial_fit(batch_x)

        new_batch_t = np.zeros((len(batch_t), len(self.target_set)))





        # only keep data with target labelings in this batch
        # indices = [ ind
        #     for ind in range(len(batch_t))
        #     if batch_t[ind] in self.target_set ]
        # batch_x     = batch_x[indices]
        # batch_t     = batch_t[indices]
        # new_batch_t = np.zeros((len(indices), len(self.target_set)))
        # for ind in range(len(indices)):
        #     new_batch_t[ind][self.target_set.index(batch_t[ind])]
