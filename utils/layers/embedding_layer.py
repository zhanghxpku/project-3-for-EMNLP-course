#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
from .layer import Layer


class EmbeddingLayer(Layer):
    """EmbeddingLayer"""

    def __init__(self, vocab_size, emb_size, trainable=True, name="embedding",
                 initializer=None, **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()

        self._W = self.get_variable(name + '_W', shape=[vocab_size, emb_size],
                                    initializer=initializer, trainable=trainable)

    def _forward(self, seq, zero_forward=False):
        if zero_forward:
            seq_mask = tf.cast(tf.stack([tf.sign(seq)] * self._emb_size, axis=-1), tf.float32)
            emb = tf.nn.embedding_lookup(self._W, seq)
            emb = emb * seq_mask
            return emb
        else:
            return tf.nn.embedding_lookup(self._W, seq)


class InitializedEmbeddingLayer(Layer):
    def __init__(self, vocab_size, emb_size, init_dict, trainable=False, name="embedding",
                 initializer=None, **kwargs):

        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size

        embedding = np.zeros([vocab_size, emb_size])
        with open(init_dict) as fin:
            for i, line in enumerate(fin):
                line_list = line.strip().split('\t')
                if len(line_list) == 1:
                    id, vec = i, [float(_) for _ in line_list[0].split()]
                else:
                    id, vec = int(line_list[0]), [float(_) for _ in line_list[1].split()]
                if len(vec) != emb_size or id >= vocab_size:
                    print
                    'Load pretrained emb: id:%s, len_vec:%s, line:%s', (id, len(vec), line)
                    assert False
                else:
                    embedding[id] = vec

        if trainable:
            self._W = self.get_variable(name + '_W', shape=[vocab_size, emb_size],
                                        initializer=tf.constant_initializer(embedding), trainable=trainable)
        else:
            self._W = tf.constant(embedding, dtype=tf.float32)

    def _forward(self, seq, zero_forward=False):
        if zero_forward:
            W = tf.concat((tf.zeros(shape=[2, self._emb_size]), self._W[2:, :]), 0)
            return tf.nn.embedding_lookup(W, seq)
        else:
            return tf.nn.embedding_lookup(self._W, seq)


class TrigramEmbeddingLayer(Layer):
    def __init__(self, trigram_size, emb_size, trainable=True, name="tri_embedding",
                 initializer=None, **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        self._W = tf.get_variable(name + '_W', shape=[trigram_size - 1, emb_size],
                                  initializer=tf.initializers.glorot_uniform(),
                                  trainable=trainable)
        
    def _forward(self, seq):
        W = tf.concat((tf.zeros(shape=[1, self._emb_size]), self._W), 0)
        trigram_emb = tf.nn.embedding_lookup(W, seq)
        emb = tf.div_no_nan(tf.reduce_sum(trigram_emb, axis=2), tf.count_nonzero(trigram_emb, axis=2, dtype=tf.float32))
        return emb


class TrigramEmbeddingEncoder(Layer):
    def __init__(self, trigram_size, emb_size, region_radius, trainable=True, name="tri_encoder",
                 initializer=None, **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        self._region_radius = region_radius
        self._paddings = tf.constant([[0, 0], [region_radius, region_radius], [0, 0]])
        self._trigram_emb_layer = []
        for i in range(2 * region_radius + 1):
            self._trigram_emb_layer.append(TrigramEmbeddingLayer(trigram_size, emb_size, name=name+'_trigram_emb_' + str(i)))
        
    def _forward(self, seq):
        h = []
        max_len = tf.shape(seq)[1]
        paded_seq = tf.cast(tf.pad(seq, self._paddings, "CONSTANT"), dtype=tf.int32)
        for i in range(2 * self._region_radius + 1):
            # each item: [batch_size, max_len, emb_size]
            h.append(self._trigram_emb_layer[i](paded_seq[:,i:i+max_len,:]))
        h = tf.tanh(tf.add_n(h))
        return h


def main():
    """main"""
    pass


if '__main__' == __name__:
    main()
