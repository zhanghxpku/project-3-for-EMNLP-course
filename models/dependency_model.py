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
    """CNN Dependency Model"""
    def __init__(self, config):
        super(DependencyModel, self).__init__(config)
        relation2word = tf.constant(np.load(config.relation2word))
        self.relation2word_layer = tf.get_variable('relation2word', initializer=relation2word,
                                                  trainable=False)
        relation2word_emb = tf.constant(np.load(config.relation2word_emb))
        self.relation2word_emb_layer = tf.get_variable('relation2word_emb', initializer=relation2word_emb,
                                                  trainable=False)

        if config.encoder == 'trigram':
            word2gram_emb = tf.constant(np.load(config.word2gram))
            self.word2gram_emb_layer = tf.get_variable('word2gram_emb', initializer=word2gram_emb,
                                                  trainable=False)
            relation2gram_emb = tf.constant(np.load(config.relation2gram))
            self.relation2gram_emb_layer = tf.get_variable('relation2gram_emb', initializer=relation2gram_emb,
                                                  trainable=False)
        else:
            word2char_emb = tf.constant(np.load(config.word2char))
            self.word2char_emb_layer = tf.get_variable('word2char_emb', initializer=word2char_emb,
                                                  trainable=False)
            relation2char_emb = tf.constant(np.load(config.relation2char))
            self.relation2char_emb_layer = tf.get_variable('relation2char_emb', initializer=relation2char_emb,
                                                  trainable=False)

    def build_graph(self, inputs, mode):
        config = self.config
        training = (mode == tf.estimator.ModeKeys.TRAIN)        

        # [batch_size, max_len]
        patterns = inputs['word']
        # [relation_size, relation_max_len]
        relations = self.relation2word_layer
        
        # [batch_size]
        tags = inputs['relation']
        
        # Encoding
        # Word-level Encoder
        patterns_emb = inputs['word_emb']
#        if not config.use_word_pretrain_emb:
#            word_emb_layer = layers.EmbeddingLayer(config.vocab_size, config.word_emb_size, name='word_emb')
#        else:
        word_emb_layer = layers.InitializedEmbeddingLayer(config.vocab_size_emb, config.word_emb_size, config.word2vec_emb,
                                                          trainable=config.word_emb_finetune,
                                                          name='word_emb')
        patterns_emb = word_emb_layer(patterns_emb)
        relations_emb = self.relation2word_emb_layer
        relations_emb = word_emb_layer(relations_emb)
        
        if config.use_rnn:
            nwords = tf.count_nonzero(patterns_emb, axis=-1, dtype=tf.int32)
            relation_nwords = tf.count_nonzero(relations, axis=-1, dtype=tf.int32)
            # Bi-LSTM
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            with tf.variable_scope('lstm1'):
                output_fw, (_, results_fw) = lstm_cell_fw(patterns_emb, dtype=tf.float32, sequence_length=nwords)
                output_fw = tf.layers.dropout(output_fw, rate=config.dropout_rate, training=training)
                output_fw_relation, (_, results_fw_relation) = lstm_cell_fw(relations_emb, dtype=tf.float32, sequence_length=relation_nwords)
                output_fw_relation = tf.layers.dropout(output_fw_relation, rate=config.dropout_rate, training=training)
            with tf.variable_scope('lstm2'):
                output_bw, (_, results_bw) = lstm_cell_bw(patterns_emb, dtype=tf.float32, sequence_length=nwords)
                output_bw = tf.layers.dropout(output_bw, rate=config.dropout_rate, training=training)
                output_bw_relation, (_, results_bw_relation) = lstm_cell_bw(relations_emb, dtype=tf.float32, sequence_length=relation_nwords)
                output_bw_relation = tf.layers.dropout(output_bw_relation, rate=config.dropout_rate, training=training)
    
            h_pattern_word = tf.concat([output_fw, output_bw], axis=-1)
            h_pattern_word = tf.transpose(h_pattern_word, perm=[1, 0, 2])
            h_relation_word = tf.concat([output_fw_relation, output_bw_relation], axis=-1)
            h_relation_word = tf.transpose(h_relation_word, perm=[1, 0, 2])
        else:
            h_pattern_word = patterns_emb
            h_relation_word = relations_emb

        if config.word_aggregation == 'none':
            pass
        elif config.word_aggregation == 'end':
            h_pattern_word = tf.concat([results_fw, results_bw], axis=-1)
            h_relation_word = tf.concat([results_fw_relation, results_bw_relation], axis=-1)
        elif config.word_aggregation == 'attention':
            assert False, 'nor implemented error'

        # Char-level Encoder
        if config.use_rnn_char:
            aggregation = 'rnn'
        else:
            aggregation = 'mean'
        if config.encoder == 'trigram':
            trigram_emb_encoder = layers.TrigramEmbeddingEncoder(config.trigram_size, config.char_emb_size, config.region_radius,
                                                                 aggregation=aggregation,
                                                                 name='trigrams')
            # [batch_size, max_len, max_char, char_emb_size]
            pattern_grams = tf.nn.embedding_lookup(self.word2gram_emb_layer, patterns)
            # [batch_size, max_len, char_emb_size]
            h_pattern_char = trigram_emb_encoder(pattern_grams)
            # [relation_size, relation_max_len, char_emb_size]
            h_relation_char = trigram_emb_encoder(self.relation2gram_emb_layer)
        elif config.encoder == 'CNN':
            char_emb_encoder = layers.CharEmbeddingEncoder(config.char_size, config.char_emb_size,
                                                           groups=config.groups,
                                                           filters=config.filters,
                                                           kernel_size=config.kernel_size,
                                                           aggregation=aggregation,
                                                           name='char')
            # [batch_size, max_len, max_char]
            pattern_chars = tf.nn.embedding_lookup(self.word2char_emb_layer, patterns)
            # [batch_size, max_len, char_emb_size]
            h_pattern_char = char_emb_encoder(pattern_chars)
            # [relation_size, relation_max_len, char_emb_size]
            h_relation_char = char_emb_encoder(self.relation2char_emb_layer)
        
        # [batch_size, max_len, char_emb_size+word_emb_size]
        h_pattern = tf.concat([h_pattern_char, h_pattern_word], axis=-1)
        # [relation_size, relation_max_len, char_emb_size+word_emb_size]
        h_relation = tf.concat([h_relation_char, h_relation_word], axis=-1)

        if config.use_highway:
            dense_t = tf.layers.Dense(h_pattern.get_shape()[-1],
                                      activation=tf.math.sigmoid,
                                      kernel_initializer=tf.initializers.glorot_uniform(),
                                      name='dense_t')
            dense_h = tf.layers.Dense(h_pattern.get_shape()[-1],
                                      activation=tf.nn.tanh,
                                      kernel_initializer=tf.initializers.glorot_uniform(),
                                      name='dense_h')
            t_pattern = dense_t(h_pattern)
            h_pattern = t_pattern * dense_h(h_pattern) + (1 - t_pattern) * h_pattern
            t_relation = dense_t(h_relation)
            h_relation = t_relation * dense_h(h_relation) + (1 - t_relation) * h_relation

        # Aggregation
        if config.aggregation == 'max':
            emb_size = tf.shape(h_pattern)[-1]
            pattern_mask = tf.tile(tf.expand_dims(tf.cast(tf.equal(patterns, 0), dtype=tf.float32), axis=-1), [1,1,emb_size])
            relation_mask = tf.tile(tf.expand_dims(tf.cast(tf.equal(relations, 0), dtype=tf.float32), axis=-1), [1,1,emb_size])
            h_pattern = tf.reduce_max(h_pattern - pattern_mask*1000, axis=1)
            h_relation = tf.reduce_max(h_relation - relation_mask*1000, axis=1)
        elif config.aggregation == 'mean':
            h_pattern = tf.div_no_nan(tf.reduce_sum(h_pattern), tf.count_nonzero(patterns, axis=-1, dtype=tf.float32, keepdims=True))
            h_relation = tf.div_no_nan(tf.reduce_sum(h_relation), tf.count_nonzero(relations, axis=-1, dtype=tf.float32, keepdims=True))
        elif config.aggregation == 'attention':
            assert False, 'nor implemented error'

        # Projection to semantic space
        dense_layers = []
        for i in range(config.layer_num):
            dense_layers.append(tf.layers.Dense(config.semantic_size, name='dense_'+str(i),
                                        activation = tf.nn.tanh,
                                        use_bias = True,
                                        kernel_initializer=tf.initializers.glorot_uniform()))
        semantic_proj = tf.layers.Dense(config.semantic_size, name='semantic_proj',
                                        activation = tf.nn.tanh,
                                        use_bias = True,
                                        kernel_initializer=tf.initializers.glorot_uniform())

        # [batch_size, semantic_size]
        for i in range(config.layer_num):
            h_pattern = dense_layers[i](h_pattern)
            h_pattern = tf.layers.dropout(h_pattern, rate=config.dropout_rate, training=training)
        pattern = semantic_proj(h_pattern)
        
        for i in range(config.layer_num):
            h_relation = dense_layers[i](h_relation)
            h_relation = tf.layers.dropout(h_relation, rate=config.dropout_rate, training=training)
        # [relation_size, semantic_size]
        relation = semantic_proj(h_relation)
        
        # [batch_size]
        norm_p = tf.expand_dims(tf.norm(pattern, axis=-1), axis=-1)
        pattern = tf.div_no_nan(pattern, norm_p)
        # [relation_size]
        norm_r = tf.expand_dims(tf.norm(relation, axis=-1), axis=-1)
        relation = tf.div_no_nan(relation, norm_r)
        # [batch_size, relation_size]
        score = tf.matmul(pattern, relation, transpose_b=True) * config.gamma
        
        self.logits = score
        self.logits_op = score

        # negative sampling
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
        self.loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=labels_one_hot,
                        logits=score))

        self.infer_op = tf.argmax(score, -1)
        metric_layer = layers.EMMetricLayer()
        self.metric = metric_layer(self.infer_op, tags)


class LSTMDependencyModel(model_base.ModelBase):
    """LSTM Dependency Model"""
    def __init__(self, config):
        super(LSTMDependencyModel, self).__init__(config)
        relation2words = tf.constant(np.load(config.relation2words))
        self.relation_emb_layer = tf.get_variable('relation_emb', initializer=relation2words,
                                                  trainable=False)
    
    def build_graph(self, inputs, mode):
        config = self.config
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        tags = inputs['relation']

        # Layers
        if not config.use_word_pretrain_emb:
            word_emb_layer = layers.EmbeddingLayer(config.vocab_size, config.emb_size, name='word_emb')
        else:
            word_emb_layer = layers.InitializedEmbeddingLayer(config.vocab_size, config.emb_size, config.word2vec_dict, trainable=config.word_emb_finetune, name='word_emb')
        semantic_proj = tf.layers.Dense(config.semantic_size, name='semantic_proj',
                                        activation = tf.nn.tanh,
                                        use_bias = False,
                                        kernel_initializer=tf.initializers.glorot_uniform())
        
        # [batch_size, max_len, emb_size]
        word_emb = word_emb_layer(inputs['word'])
#        print 'word_emb', word_emb
        nwords = tf.count_nonzero(inputs['word'], axis=-1, dtype=tf.int32)
#        print 'nwords', nwords

        tag_emb = self.relation_emb_layer
        # [relation_size, relation_max_len, emb_size]
        nwords_tag = tf.count_nonzero(tag_emb, axis=-1, dtype=tf.int32)
#        print 'nwords_tag', nwords_tag
        tag_emb = word_emb_layer(tag_emb)
#        print 'tag_emb', tag_emb

        word_emb = tf.transpose(word_emb, perm=[1, 0, 2])
        tag_emb = tf.transpose(tag_emb, perm=[1, 0, 2])
        # Bi-LSTM
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        with tf.variable_scope('lstm1'):
            output_fw, _ = lstm_cell_fw(word_emb, dtype=tf.float32, sequence_length=nwords)
            output_fw = tf.layers.dropout(output_fw, rate=config.dropout_rate, training=training)
            output_fw_tag, _ = lstm_cell_fw(tag_emb, dtype=tf.float32, sequence_length=nwords_tag)
            output_fw_tag = tf.layers.dropout(output_fw_tag, rate=config.dropout_rate, training=training)
        with tf.variable_scope('lstm2'):
            output_bw, _ = lstm_cell_bw(word_emb, dtype=tf.float32, sequence_length=nwords)
            output_bw = tf.layers.dropout(output_bw, rate=config.dropout_rate, training=training)
            output_bw_tag, _ = lstm_cell_bw(tag_emb, dtype=tf.float32, sequence_length=nwords_tag)
            output_bw_tag = tf.layers.dropout(output_bw_tag, rate=config.dropout_rate, training=training)
        h_word = tf.concat([output_fw, output_bw], axis=-1)
        h_word = tf.transpose(h_word, perm=[1, 0, 2])
        
        h_tag = tf.concat([output_fw_tag, output_bw_tag], axis=-1)
        h_tag = tf.transpose(h_tag, perm=[1, 0, 2])

        # [batch_size, max_len, emb_size]
        h_word = tf.reduce_max(h_word, axis=1)
        # [batch_size, semantic_size]
        pattern = tf.tanh(semantic_proj(h_word))
        # [relation_size, relation_max_len, emb_size]
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
        self.loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=labels_one_hot,
                        logits=score))

        self.infer_op = tf.argmax(score, -1)
        metric_layer = layers.EMMetricLayer()
        self.metric = metric_layer(self.infer_op, tags)
