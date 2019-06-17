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
            mask = tf.expand_dims(tf.cast(tf.not_equal(seq, 0), dtype=tf.float32), axis=-1)
#            seq_mask = tf.cast(tf.stack([tf.sign(seq)] * self._emb_size, axis=-1), tf.float32)
            emb = tf.nn.embedding_lookup(self._W, seq)
            return emb * mask
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
#            W = tf.concat((tf.zeros(shape=[1, self._emb_size]), self._W[1:, :]), 0)
            mask = tf.expand_dims(tf.cast(tf.not_equal(seq, 0), dtype=tf.float32), axis=-1)
            return tf.nn.embedding_lookup(self._W, seq) * mask
        else:
            return tf.nn.embedding_lookup(self._W, seq)


def main():
    """main"""
    pass


if '__main__' == __name__:
    main()
