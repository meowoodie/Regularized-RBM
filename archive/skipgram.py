#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""
import sys
import math
import arrow
import numpy as np
import tensorflow as tf

class Skipgram(object):
    """
    """

    def __init__(self, num_events, embedding_size, init_embeddings,
                 batch_size=128, num_sampled=64, lr=0.1, iters=1000, display_step=1):
        self.num_events     = num_events
        self.embedding_size = embedding_size
        self.batch_size     = batch_size
        self.iters          = iters
        self.display_step   = display_step

        #TODO: let tfidf of corpus be init embeddings
        self.embeddings = tf.Variable(init_embeddings, dtype=tf.float32)
            # tf.random_uniform(
            #     [self.num_events, self.embedding_size],
            #     -1.0, 1.0))
        self.nce_weights = tf.Variable(
            tf.truncated_normal(
                [self.num_events, self.embedding_size],
                stddev=1.0/math.sqrt(self.embedding_size)))
        self.nce_biases  = tf.Variable(
            tf.zeros([self.num_events]))

        # Placeholders for inputs
        # - target events
        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        # - context events
        self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Build mapping between embeddings and inputs
        embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

        # Compute the NCE loss, using a sample of the negative labels each time.
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=self.nce_weights,
                           biases=self.nce_biases,
                           labels=self.train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=self.num_events))

        # Optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm

    def get_embeddings(self, sess, pretrained=True):
        """
        Get normalized embeddings from well-fitted model
        """
        # Set pretrained variable if it was existed
        if not pretrained:
			init = tf.global_variables_initializer()
			sess.run(init)

        return sess.run(self.normalized_embeddings)

    def train(self, sess, input_data, label_data, pretrained=False):
        """
        Train Event2vec Model
        """
        # Set pretrained variable if it was existed
        if not pretrained:
			init = tf.global_variables_initializer()
			sess.run(init)

        step        = 1 # the step of the iteration
        start_index = 0 # the index of the start row of the batch
		# Keep training until reach max iterations
        while step * self.batch_size <= self.iters:
            # Fetch the next batch of the input data (q, d, y)
            # And update the start_indext in order to prepare for the next batch
            batch_input_data, batch_label_data, start_index = self._next_batch(input_data, label_data, start_index)
            # Run optimization
            sess.run(self.optimizer, feed_dict={
                self.train_inputs: batch_input_data,
                self.train_labels: batch_label_data})
            if step % self.display_step == 0:
                # Calculate batch loss and accuracy
                train_loss = sess.run(self.loss, feed_dict={
                    self.train_inputs: batch_input_data,
                    self.train_labels: batch_label_data})
                print >> sys.stderr, "[%s] Iter: %d\tTrain Loss: %.5f" % \
                    (arrow.now(), (step * self.batch_size), train_loss)
            step += 1
        print >> sys.stderr, "[%s] Optimization Finished!" % arrow.now()

    def _next_batch(self, input_data, label_data, start_index):
        """
		Next Batch

		This is a private method for fetching a batch of data from the integral input data.
		Each time you call this method, it would return the next batch in the dataset by indicating
		the start index. Which means you have to keep the return start index of every invoking,
		and pass it to the next invoking for getting the next batch correctly.
		"""
        # total number of rows of the input data
        # num_seq, num_action, num_feature = np.shape(input_data)
        # num_events, num_feature = np.shape(input_data)
        num_data = len(input_data)
        # start index of the row of the input data
        start_event = start_index % num_data
        # end index of the row of the input data
        end_event   = (start_event + self.batch_size) % num_data
        # if there is not enought data left in the dataset for generating an integral batch,
        # then top up this batch by traversing back to the start of the dataset.
        if end_event < start_event:
            batch_input_data = np.append(input_data[start_event: num_data], input_data[0: end_event], axis=0).astype(np.float32)
            batch_label_data = np.append(label_data[start_event: num_data], label_data[0: end_event], axis=0).astype(np.float32)
        else:
            batch_input_data = input_data[start_event: end_event].astype(np.float32)
            batch_label_data = label_data[start_event: end_event].astype(np.float32)
        # Update the start index
        start_index += self.batch_size
        return batch_input_data, batch_label_data, start_index

if __name__ == "__main__":

    from context import Context

    c  = Context("data/info.txt")
    sc = c.spatial_context(delta=0.05)
    input_data, label_data = c.target_context_pairs(sc)
    print len(input_data)
    print len(label_data)

    input_data = np.array([0, 0, 2, 2, 3, 3])
    label_data = np.array([[1], [2], [1], [3], [2], [4]])
    model      = Skipgram(5, 3, batch_size=2, num_sampled=2, iters=10, display_step=1)
    with tf.Session() as sess:
        model.train(sess, input_data, label_data)
