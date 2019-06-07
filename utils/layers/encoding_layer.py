#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from .layer import Layer
import numpy as np

class MeanEncoder(Layer):
    def __init__(self, vocab_size, emb_size, region_radius, trainable=True, name="mean_encoder",
                 initializer=None, **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._vocab_size = vocab_size
        self._emb_size = emb_size
        self._region_radius = region_radius
        self._paddings = tf.constant([[0, 0], [self._region_radius, self._region_radius]])
        self._bias = np.tile(np.array([i for i in range(region_radius*2+1)]),(1,1,1))
        self._W = tf.get_variable(name + '_W', shape=[(vocab_size-1)*(region_radius*2+1), emb_size],
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                  trainable=trainable)

    def _forward(self, seq):
        region_size = 2*self._region_radius + 1
        padded_seq = tf.pad(seq, self._paddings, "CONSTANT")
        s = tf.shape(seq)
        r = tf.range(region_size)
        align_emb = tf.map_fn(lambda i: padded_seq[:,i:i+s[1]], r)
        align_emb = tf.transpose(align_emb, perm=[1, 2, 0])*region_size + self._bias
        W = tf.concat((tf.zeros(shape=[region_size, self._emb_size]), self._W), 0)
        # [batch_size, max_len, region_radius, emb_size]
        align_emb = tf.nn.embedding_lookup(W, align_emb)
        trigram_emb = tf.div_no_nan(tf.reduce_sum(align_emb,axis=2), tf.count_nonzero(seq, axis=1, dtype=tf.float32, keepdims=True))
        h = tf.tanh(trigram_emb) 
        return h


class RegionEncoder(Layer):
    def __init__(self, vocab_size, emb_size, region_radius, trainable=True, name="region_encoder",
                 initializer=None, **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._vocab_size = vocab_size
        self._emb_size = emb_size
        self._region_radius = region_radius
        self._paddings = tf.constant([[0, 0], [self._region_radius, self._region_radius]])
        self._bias = np.tile(np.array([i for i in range(region_radius*2+1)]),(1,1,1))
        self._W = tf.get_variable(name + '_W', shape=[vocab_size-1, emb_size],
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                  trainable=trainable)
        # Context matrix
        self._U = tf.get_variable(name + '_U',
                     shape=[(vocab_size-1)*(region_radius*2+1), emb_size],
                     dtype=tf.float32,
                     trainable=trainable)

    def _forward(self, seq):
        region_size = 2*self._region_radius + 1
        padded_seq = tf.pad(seq, self._paddings, "CONSTANT")
        s = tf.shape(seq)
        r = tf.range(region_size)
        align_emb = tf.map_fn(lambda i: padded_seq[:,i:i+s[1]], r)
        align_emb = tf.transpose(align_emb, perm=[1, 2, 0])*region_size + self._bias
        # Word Embedding
        W = tf.concat((tf.zeros(shape=[1, self._emb_size]), self._W), 0)
        trigram_emb = tf.nn.embedding_lookup(W, seq)
        # Context-Word Embedding
        U = tf.concat((tf.zeros(shape=[region_size, self._emb_size]), self._U), 0)
        align_emb = tf.nn.embedding_lookup(U, align_emb)
        trigram_emb = tf.expand_dims(trigram_emb, 2)
        # [batch_size, max_len, region_radius, emb_size]
        projected_emb = align_emb * trigram_emb
        h = tf.reduce_max(projected_emb, axis=2)
        return h


class CNNEncoder(Layer):
    def __init__(self, vocab_size, emb_size, groups, filters, kernel_size,
                 trainable=True, name="cnn_encoder", initializer=None, emb=None, **kwargs):
        Layer.__init__(self, name, **kwargs)
        if emb is None:
            self._W = tf.get_variable(name + '_W', shape=[vocab_size - 1, emb_size],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                      trainable=trainable)
        self._emb = emb
        self._emb_size = emb_size
        self._groups = groups
        self._filters = filters
        self._kernel_size = kernel_size
        self._name = name
        self._conv_layers = []
        for i in range(self._groups):
            self._conv_layers.append(tf.layers.Conv1D(kernel_size=self._kernel_size[i],
                                      filters=self._filters[i],
                                      strides=1,
                                      padding='same',
                                      activation=tf.tanh,
                                      use_bias=False,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                      name=self._name+'_'+str(i)))

    def _forward(self, seq):
        nwords = tf.count_nonzero(seq, axis=-1, dtype=tf.float32, keepdims=True)
        if self._emb is None:
            W = tf.concat((tf.zeros(shape=[1, self._emb_size]), self._W), 0)
            # [batch_size, max_len, emb_size]
            char_emb = tf.nn.embedding_lookup(W, seq)
        else:
            char_emb = self._emb(seq, zero_forward=True)
        h = []
        for i in range(self._groups):
            # [batch_size, max_len, filters[i]]
            h.append(self._conv_layers[i](char_emb))
        h = tf.concat(h, axis=-1)
#        if self._aggregation == 'mean':
        h = tf.div_no_nan(tf.reduce_sum(h, axis=2), nwords)
#        elif self._aggregation == 'rnn':
#            # Forwarding LSTM
#            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(tf.shape(h)[-1])
#            _, (_, results_fw) = lstm_cell_fw(h, dtype=tf.float32, sequence_length=nwords)
##            output_fw = tf.layers.dropout(output_fw, rate=config.dropout_rate, training=training)
#            h = results_fw

        return h


def main():
    """main"""
    pass


if '__main__' == __name__:
    main()
