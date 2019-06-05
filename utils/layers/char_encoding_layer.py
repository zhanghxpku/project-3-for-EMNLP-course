#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from .layer import Layer


class TrigramEmbeddingLayer(Layer):
    def __init__(self, trigram_size, emb_size, trainable=True, name="tri_embedding",
                 initializer=None, aggregation='mean', **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
#        self.__aggregation = aggregation
        self._W = tf.get_variable(name + '_W', shape=[trigram_size - 1, emb_size],
                                  initializer=tf.initializers.glorot_uniform(),
                                  trainable=trainable)
        
    def _forward(self, seq):
        W = tf.concat((tf.zeros(shape=[1, self._emb_size]), self._W), 0)
        trigram_emb = tf.nn.embedding_lookup(W, seq)
#        if self.__aggregation == 'mean':
        emb = tf.div_no_nan(tf.reduce_sum(trigram_emb, axis=2), tf.count_nonzero(seq, axis=2, dtype=tf.float32, keepdims=True))
#        elif self.__aggregation == 'rnn':
#            nwords_char = tf.count_nonzero(seq, axis=-1, dtype=tf.int32)
#            # Forwarding LSTM
#            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(tf.shape(emb)[-1])
#            _, (_, results_fw) = lstm_cell_fw(emb, dtype=tf.float32, sequence_length=nwords_char)
##            output_fw = tf.layers.dropout(output_fw, rate=config.dropout_rate, training=training)
#            emb = results_fw
        return emb


class TrigramEmbeddingEncoder(Layer):
    def __init__(self, trigram_size, emb_size, region_radius, trainable=True, name="tri_encoder",
                 initializer=None, aggregation='mean', **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        self._region_radius = region_radius
        self._paddings = tf.constant([[0, 0], [region_radius, region_radius], [0, 0]])
        self._trigram_emb_layer = []
        for i in range(2 * region_radius + 1):
            self._trigram_emb_layer.append(TrigramEmbeddingLayer(trigram_size, emb_size,
                                                                 trainable=trainable,
                                                                 aggregation=aggregation,
                                                                 name=name+'_trigram_emb_' + str(i)))
    def _forward(self, seq):
        h = []
        max_len = tf.shape(seq)[1]
        paded_seq = tf.cast(tf.pad(seq, self._paddings, "CONSTANT"), dtype=tf.int32)
        for i in range(2 * self._region_radius + 1):
            h.append(self._trigram_emb_layer[i](paded_seq[:,i:i+max_len,:]))
        h = tf.tanh(tf.add_n(h))
        return h
#        self._emb_size = emb_size
#        self._aggregation = aggregation
#        self._region_radius = region_radius
#        if self._aggregation == 'mean':
#            self._W = tf.get_variable(name + '_W', shape=[trigram_size*(region_radius*2+1), emb_size],
#                                      initializer=tf.initializers.glorot_uniform(),
#                                      trainable=trainable)
#        if self._aggregation == 'region':
#            self._W = tf.get_variable(name + '_W', shape=[trigram_size, emb_size],
#                                      initializer=tf.initializers.glorot_uniform(),
#                                      trainable=trainable)
#            # Context matrix
#            self._U = tf.get_variable(name + '_U',
#                         shape=[trigram_size*(region_radius*2+1), emb_size],
#                         dtype=tf.float32,
#                         trainable=True)
#
#    def _forward(self, seq):
#        max_len = seq.get_shape()[2]
#        region_size = 2 * self._region_radius + 1
#        paddings = tf.constant([[0, 0], [0, 0], [self._region_radius, self._region_radius]])
#        padded_seq = tf.pad(seq, paddings, "CONSTANT")
#        s = tf.shape(seq)
#        r = xrange(region_size)
#        align_seq = tf.map_fn(lambda i: padded_seq[:,:,i:i+tf.shape(seq)[2]], r)
#        align_layer = tf.map_fn(lambda i: tf.ones(shape=s,dtype=tf.int32)*i, r)
##        align_layer = tf.convert_to_tensor(align_layer)
##        align_seq = tf.convert_to_tensor(align_seq)
#        align_emb = align_layer + align_seq*region_size
#        align_emb = tf.transpose(align_emb, perm=[1, 2, 0, 3])
#        trigram_emb = tf.nn.embedding_lookup(self._W, seq)
#        if self._aggregation == 'mean':
#            align_emb = tf.nn.embedding_lookup(self._W, align_emb)
##            mask = tf.tile(tf.expand_dims(tf.cast(tf.equal(seq, 0), dtype=tf.float32), axis=-1), [1,1,1,self._emb_size])
##            trigram_emb = tf.reduce_max(trigram_emb - mask*1000, axis=2)
#            trigram_emb = tf.div_no_nan(tf.reduce_sum(tf.reduce_sum(align_emb, axis=2), axis=2), tf.count_nonzero(seq, axis=2, dtype=tf.float32, keepdims=True))
#            h = tf.tanh(trigram_emb)
#        elif self._aggregation == 'region':
#            # Context-Word Embedding
##            print 'align_seq', align_seq
##            print 'self._U', self._U
#            align_emb = tf.nn.embedding_lookup(self._U, align_emb)
#            trigram_emb = tf.expand_dims(trigram_emb, 2)
#            projected_emb = align_emb * trigram_emb
#            h = tf.reduce_max(tf.reduce_max(projected_emb, axis=2), axis=2)
#        return h


class CharEmbeddingEncoder(Layer):
    def __init__(self, char_size, emb_size, region_radius, groups, filters, kernel_size,
                 trainable=True, name="char_encoder", initializer=None, aggregation='mean', **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._W = tf.get_variable(name + '_W', shape=[char_size - 1, emb_size],
                                  initializer=tf.initializers.glorot_uniform(),
                                  trainable=trainable)
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
                                      kernel_initializer=tf.initializers.glorot_uniform(),
                                      name=self._name+'_'+str(i)))

    def _forward(self, seq):
        char_shape = seq.get_shape()
        nwords_char = tf.count_nonzero(seq, axis=-1, dtype=tf.float32, keepdims=True)
        W = tf.concat((tf.zeros(shape=[1, self._emb_size]), self._W), 0)
        # [batch_size, max_len, max_char, emb_size]
        char_emb = tf.nn.embedding_lookup(W, seq)
        char_emb = tf.reshape(char_emb, [-1, char_shape[2], self._emb_size])
        h = []
        for i in range(self._groups):
            # [batch_size, max_len, max_char, filters[i]]
            h.append(self._conv_layers[i](char_emb))
        h = tf.concat(h, axis=-1)
        h = tf.reshape(h, [-1, char_shape[1], char_shape[2], h.get_shape()[2]])

#        if self._aggregation == 'mean':
        h = tf.div_no_nan(tf.reduce_sum(h, axis=2), nwords_char)
#        elif self._aggregation == 'rnn':
#            # Forwarding LSTM
#            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(tf.shape(h)[-1])
#            _, (_, results_fw) = lstm_cell_fw(h, dtype=tf.float32, sequence_length=nwords_char)
##            output_fw = tf.layers.dropout(output_fw, rate=config.dropout_rate, training=training)
#            h = results_fw

        return h


def main():
    """main"""
    pass


if '__main__' == __name__:
    main()
