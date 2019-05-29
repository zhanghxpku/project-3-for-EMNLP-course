#/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np

import utils.layers as layers
import utils.tools.config
import model_base
#from tensorflow.contrib import autograph


class DependencyModel(model_base.ModelBase):
    """Dependency Model"""
    def __init__(self, config):
        super(DependencyModel, self).__init__(config)
        self.emb_layer = layers.InitializedEmbeddingLayer(config.vocab_size, config.max_char, config.word2vec_dict, trainable=False, name='emb')
        relation2grams = tf.constant(np.load(config.relation2grams))
        self.relation_emb_layer = tf.get_variable('relation_emb', initializer=relation2grams, trainable=False)
    
    def build_graph(self, inputs, mode):
        config = self.config
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        # Layers
        trigram_emb_encoder = layers.TrigramEmbeddingEncoder(config.trigram_size, config.emb_size, config.region_radius, trainable=False, name='trigrams')
        semantic_proj = tf.layers.Dense(config.semantic_size, name='semantic_proj',
                                        activation = tf.nn.tanh,
                                        use_bias = False)
        
        tags = inputs['relation']
        tag_emb = self.relation_emb_layer
        
        # [batch_size, max_len, max_char]
        word_emb = self.emb_layer(inputs['word'])
        
        # [batch_size, max_len, emb_size]
        h_word = trigram_emb_encoder(word_emb)
        h_word = tf.layers.dropout(h_word, rate=config.dropout_rate, training=training)
#        weights = tf.cast(tf.not_equal(inputs['word'], 0), dtype=tf.float32)
        # [batch_size, emb_size]
        h_word = tf.reduce_max(h_word, axis=1)
        # [batch_size, semantic_size]
        pattern = tf.tanh(semantic_proj(h_word))
        
        # [relation_size, relation_max_len, emb_size]
#        tag_emb = tf.reshape(tag_emb, [-1, config.relation_max_len, config.relation_max_char])
        h_tag = trigram_emb_encoder(tag_emb)
#        h_tag = tf.reshape(h_tag, [-1, config.relation_size, config.semantic_size])
        h_tag = tf.layers.dropout(h_tag, rate=config.dropout_rate, training=training)
        h_tag = tf.reduce_max(h_tag, axis=1)
        # [relation_size, semantic_size]
        relation = tf.tanh(semantic_proj(h_tag))
        
        # [batch_size]
        norm_p = tf.expand_dims(tf.norm(pattern, axis=-1), axis=-1)
        pattern = tf.div_no_nan(pattern, norm_p)
        # [relation_size]
        norm_r = tf.expand_dims(tf.norm(relation, axis=-1), axis=-1)
        relation = tf.div_no_nan(relation, norm_r)
#        print 'pattern', pattern
#        print 'relation', relation
        # [batch_size, relation_size]
        score = tf.matmul(pattern, relation, transpose_b=True) * config.gamma
        
        self.logits = score
        self.logits_op = score
        
#        if training:
#            prob = tf.nn.sampled_softmax_loss(
#                            weights=tf.ones([tf.shape(score)[1],tf.shape(score)[1]]),
#                            biases=tf.zeros(tf.shape(score)[1:]),
#                            labels=tf.expand_dims(tags,axis=-1),
#                            inputs=score,
#                            num_sampled=config.negative_samples,
#                            num_classes=config.relation_size,
#                            partition_strategy='div')
##            print 'prob', prob
#            self.loss_op = tf.reduce_sum(prob)
#        else:
        labels_one_hot = tf.one_hot(tags, config.relation_size)
        self.loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=labels_one_hot,
                        logits=score)

        self.infer_op = tf.argmax(score, -1)
        metric_layer = layers.EMMetricLayer()
        self.metric = metric_layer(self.infer_op, inputs['relation'])

