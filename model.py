from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim

import tfplot
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from ops import *
from util import log

class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.input_height = self.config.data_info[0]
        self.input_width = self.config.data_info[1]
        self.num_class = self.config.data_info[2]
        self.c_dim = self.config.data_info[3]
        self.visualize_shape = self.config.visualize_shape
        self.conv_info = self.config.conv_info
        self.activation = self.config.activation

        # create placeholders for the input
        self.image = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[self.batch_size, self.input_height, self.input_width, self.c_dim],
        )
        self.label = tf.placeholder(
            name='label', dtype=tf.float32, shape=[self.batch_size, self.num_class],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.image: batch_chunk['image'], # [B, h, w, c]
            self.label: batch_chunk['label'], # [B, n]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd

    def build(self, is_train=True):

        n = self.num_class
        num_layer = 6
        conv_info = self.conv_info
        visualize_shape = self.visualize_shape # [# of layers, 2]

        # build loss and accuracy {{{
        def build_loss(logits, labels):
            # Cross-entropy loss            
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.label,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy
        # }}}

        # Classifier: takes images as input and tries to output class label [B, n]
        def C(img, activation, scope='Classifier'):
            with tf.variable_scope(scope) as scope:
                print ('\033[93m'+scope.name+'\033[0m')
                c_1 = conv2d(img, conv_info[0], is_train, activation, name='c_1_conv')
                c_1 = slim.dropout(c_1, keep_prob=0.5, is_training=is_train, scope='c_1_conv/')
                print (scope.name, c_1)
                c_2 = conv2d(c_1, conv_info[1], is_train, activation, name='c_2_conv')
                c_2 = slim.dropout(c_2, keep_prob=0.5, is_training=is_train, scope='c_2_conv/')
                print (scope.name, c_2)
                c_3 = conv2d(c_2, conv_info[2], is_train, activation, name='c_3_conv')
                c_3 = slim.dropout(c_3, keep_prob=0.5, is_training=is_train, scope='c_3_conv/')
                print (scope.name, c_3)
                c_4 = fc(tf.reshape(c_3, [self.batch_size, -1]), 16*n, is_train, activation, name='c_4_fc')
                print (scope.name, c_4)
                c_5 = fc(tf.reshape(c_4, [self.batch_size, -1]), 4*n, is_train, activation, name='c_5_fc')
                print (scope.name, c_5)
                c_6 = fc(c_5, n, is_train, activation, name='c_6_fc')
                print (scope.name, c_6)
                assert c_6.get_shape().as_list() == [self.batch_size, n], c_6.get_shape().as_list()
                return c_1, c_2, c_3, c_4, c_5, c_6

        h_1, h_2, h_3, h_4, h_5, h_6 = C(self.image, self.activation, scope='Classifier')
        h_all = [h_1, h_2, h_3, h_4, h_5, h_6]
        self.loss, self.accuracy = build_loss(h_6, self.label)

        # Add summaries
        def draw_act_vis(h, grid_shape):
            fig, ax = tfplot.subplots(figsize=(4, 4))
            i = ax.imshow(h.reshape(grid_shape))
            fig.colorbar(i)
            return fig

        def draw_act_hist(h, grid_shape):
            # import ipdb; ipdb.set_trace()
            # fig, ax = tfplot.subplots(figsize=(4, 4))
            fig = plt.figure()
            n, bins, patches = plt.hist(np.reshape(h, [grid_shape[0]*grid_shape[1]]), 50, normed=1, facecolor='blue', alpha=0.75)
            plt.xlabel('Activation values')
            plt.ylabel('Probability')
            plt.grid(True)
            plt.show() 
            return fig

        for l in range(num_layer-4):
            i = l+3
            shape = tf.tile(tf.expand_dims(visualize_shape[i, :], 0), [self.batch_size, 1])
            tfplot.summary.plot_many('visualization/h'+str(i), 
                                     draw_act_vis, [h_all[i], shape], 
                                     max_outputs=1, 
                                     collections=["plot_summaries"])
            tfplot.summary.plot_many('histogram/h'+str(i), 
                                     draw_act_hist, [h_all[i], shape], 
                                     max_outputs=1, 
                                     collections=["plot_summaries"])

        tf.summary.scalar("loss/accuracy", self.accuracy)
        tf.summary.scalar("loss/cross_entropy", self.loss)
        print ('\033[93mSuccessfully loaded the model.\033[0m')
