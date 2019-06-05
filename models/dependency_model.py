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
        if not config.use_word_pretrain_emb:
            relation2word_emb = tf.constant(np.load(config.relation2word_train))
            self.relation2word_emb_layer = tf.get_variable('relation2word_emb', initializer=relation2word_emb,
                                                      trainable=False)
        else:
            relation2word_emb = tf.constant(np.load(config.relation2word_emb))
            self.relation2word_emb_layer = tf.get_variable('relation2word_emb', initializer=relation2word_emb,
                                                      trainable=False)
        comb2relation = tf.constant(np.load(config.comb2relation))
        self.comb2relation_layer = tf.get_variable('comb2relation', initializer=comb2relation,
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
        if not training:
            dropout_rate = 0.0
        else:
            dropout_rate = config.dropout_rate
        # [batch_size, max_len]
        patterns = inputs['word']
        # [relation_size, relation_max_len]
        relations = self.relation2word_layer
        
        # [batch_size]
        tags = inputs['relation']
        entity_order = inputs['entity_order']
        entity_mask1 = tf.cast(inputs['entity_mask1'], dtype=tf.float32)
        entity_mask2 = tf.cast(inputs['entity_mask2'], dtype=tf.float32)
        
        # Encoding
        # Word-level Encoder
        if not config.use_word_pretrain_emb:
            patterns_emb = inputs['word_train']
            word_emb_layer = layers.EmbeddingLayer(config.vocab_size_train, config.word_emb_size, name='word_emb')
        else:
            patterns_emb = inputs['word_emb']
            word_emb_layer = layers.InitializedEmbeddingLayer(config.vocab_size_emb, config.word_emb_size, config.word2vec_emb,
                                                          trainable=config.word_emb_finetune,
                                                          name='word_emb')
        patterns_emb = word_emb_layer(patterns_emb)
        relations_emb = self.relation2word_emb_layer
        relations_emb = word_emb_layer(relations_emb)
        
        patterns_emb = tf.layers.dropout(patterns_emb, rate=dropout_rate, training=training)
        relations_emb = tf.layers.dropout(relations_emb, rate=dropout_rate, training=training)

        if config.use_rnn:
            h_pattern_word = tf.transpose(patterns_emb, perm=[1, 0, 2])
            h_relation_word = tf.transpose(relations_emb, perm=[1, 0, 2])
            nwords = tf.count_nonzero(inputs['word_emb'], axis=-1, dtype=tf.int32)
            relation_nwords = tf.count_nonzero(self.relation2word_emb_layer, axis=-1, dtype=tf.int32)
            # Bi-LSTM
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            with tf.variable_scope('lstm1'):
                output_fw, (_, results_fw) = lstm_cell_fw(h_pattern_word, dtype=tf.float32, sequence_length=nwords)
                output_fw = tf.layers.dropout(output_fw, rate=dropout_rate, training=training)
                output_fw_relation, (_, results_fw_relation) = lstm_cell_fw(h_relation_word, dtype=tf.float32, sequence_length=relation_nwords)
                output_fw_relation = tf.layers.dropout(output_fw_relation, rate=dropout_rate, training=training)
            with tf.variable_scope('lstm2'):
                output_bw, (_, results_bw) = lstm_cell_bw(h_pattern_word, dtype=tf.float32, sequence_length=nwords)
                output_bw = tf.layers.dropout(output_bw, rate=dropout_rate, training=training)
                output_bw_relation, (_, results_bw_relation) = lstm_cell_bw(h_relation_word, dtype=tf.float32, sequence_length=relation_nwords)
                output_bw_relation = tf.layers.dropout(output_bw_relation, rate=dropout_rate, training=training)

            h_pattern_word = tf.concat([output_fw, output_bw], axis=-1)
            h_pattern_word = tf.transpose(h_pattern_word, perm=[1, 0, 2])
            h_relation_word = tf.concat([output_fw_relation, output_bw_relation], axis=-1)
            h_relation_word = tf.transpose(h_relation_word, perm=[1, 0, 2])
        else:
            h_pattern_word = patterns_emb
            h_relation_word = relations_emb
            
        h_pattern_word = tf.layers.dropout(h_pattern_word, rate=dropout_rate, training=training)
        h_relation_word = tf.layers.dropout(h_relation_word, rate=dropout_rate, training=training)
        
        h_entity1_word = tf.expand_dims(entity_mask1,axis=-1) * h_pattern_word
        h_entity1_word = tf.div_no_nan(tf.reduce_sum(h_entity1_word, axis=1), tf.expand_dims(tf.reduce_sum(entity_mask1, axis=1),axis=-1))
        h_entity2_word = tf.expand_dims(entity_mask2,axis=-1) * h_pattern_word
        h_entity2_word = tf.div_no_nan(tf.reduce_sum(h_entity2_word, axis=1), tf.expand_dims(tf.reduce_sum(entity_mask1, axis=1),axis=-1))

        if config.word_aggregation == 'max':
            emb_size = tf.shape(h_pattern_word)[-1]
            pattern_mask = tf.tile(tf.expand_dims(tf.cast(tf.equal(patterns, 0), dtype=tf.float32), axis=-1), [1,1,emb_size])
            relation_mask = tf.tile(tf.expand_dims(tf.cast(tf.equal(relations, 0), dtype=tf.float32), axis=-1), [1,1,emb_size])
            h_pattern_word = tf.reduce_max(h_pattern_word - pattern_mask*1000, axis=1)
            h_relation_word = tf.reduce_max(h_relation_word - relation_mask*1000, axis=1)
        elif config.word_aggregation == 'mean':
            emb_size = tf.shape(h_pattern_word)[-1]
            pattern_mask = tf.expand_dims(tf.cast(tf.not_equal(patterns, 0), dtype=tf.float32), axis=-1)
            relation_mask = tf.expand_dims(tf.cast(tf.not_equal(relations, 0), dtype=tf.float32), axis=-1)
            h_pattern_word = tf.div_no_nan(tf.reduce_sum(h_pattern_word*pattern_mask), tf.count_nonzero(patterns, axis=-1, dtype=tf.float32, keepdims=True))
            h_relation_word = tf.div_no_nan(tf.reduce_sum(h_relation_word*relation_mask), tf.count_nonzero(relations, axis=-1, dtype=tf.float32, keepdims=True))
        elif config.word_aggregation == 'end':
            h_pattern_word = results_fw + results_bw
            h_relation_word = results_fw_relation + results_bw_relation
        elif config.word_aggregation == 'attention':
            assert False, 'nor implemented error'

        # Char-level Encoder
        if config.encoder == 'trigram':
            trigram_emb_encoder = layers.TrigramEmbeddingEncoder(config.trigram_size, config.char_emb_size, config.region_radius,
                                                                 aggregation=config.encoder_aggregation,
                                                                 name='trigrams')
            # [batch_size, max_len, max_char, char_emb_size]
            pattern_grams = tf.nn.embedding_lookup(self.word2gram_emb_layer, patterns)
            # [batch_size, max_len, char_emb_size]
            h_pattern_char = trigram_emb_encoder(pattern_grams)
            # [relation_size, relation_max_len, char_emb_size]
            h_relation_char = trigram_emb_encoder(self.relation2gram_emb_layer)
        elif config.encoder == 'char':
            char_emb_encoder = layers.CharEmbeddingEncoder(config.char_size, config.char_emb_size,
                                                           region_radius=config.region_radius,
                                                           groups=config.groups,
                                                           filters=config.filters,
                                                           kernel_size=config.kernel_size,
                                                           name='char')
            # [batch_size, max_len, max_char]
            pattern_chars = tf.nn.embedding_lookup(self.word2char_emb_layer, patterns)
            # [batch_size, max_len, char_emb_size]
            h_pattern_char = char_emb_encoder(pattern_chars)
            # [relation_size, relation_max_len, char_emb_size]
            h_relation_char = char_emb_encoder(self.relation2char_emb_layer)
        
        if config.use_rnn_char:
            h_pattern_char = tf.transpose(h_pattern_char, perm=[1, 0, 2])
            h_relation_char = tf.transpose(h_relation_char, perm=[1, 0, 2])
            nwords = tf.count_nonzero(inputs['word'], axis=-1, dtype=tf.int32)
            relation_nwords = tf.count_nonzero(self.relation2word_layer, axis=-1, dtype=tf.int32)
            # Bi-LSTM
            lstm_cell_fw_char = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
            lstm_cell_bw_char = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
            lstm_cell_bw_char = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw_char)
            with tf.variable_scope('lstm3'):
                output_fw_char, (_, results_fw_char) = lstm_cell_fw_char(h_pattern_char, dtype=tf.float32, sequence_length=nwords)
                output_fw_char = tf.layers.dropout(output_fw_char, rate=dropout_rate, training=training)
                output_fw_relation_char, (_, results_fw_relation_char) = lstm_cell_fw_char(h_relation_char, dtype=tf.float32, sequence_length=relation_nwords)
                output_fw_relation_char = tf.layers.dropout(output_fw_relation_char, rate=dropout_rate, training=training)
            with tf.variable_scope('lstm4'):
                output_bw_char, (_, results_bw_char) = lstm_cell_bw_char(h_pattern_char, dtype=tf.float32, sequence_length=nwords)
                output_bw_char = tf.layers.dropout(output_bw_char, rate=dropout_rate, training=training)
                output_bw_relation_char, (_, results_bw_relation_char) = lstm_cell_bw_char(h_relation_char, dtype=tf.float32, sequence_length=relation_nwords)
                output_bw_relation_char = tf.layers.dropout(output_bw_relation_char, rate=dropout_rate, training=training)

            h_pattern_char = tf.concat([output_fw_char, output_bw_char], axis=-1)
            h_pattern_char = tf.transpose(h_pattern_char, perm=[1, 0, 2])
            h_relation_char = tf.concat([output_fw_relation_char, output_bw_relation_char], axis=-1)
            h_relation_char = tf.transpose(h_relation_char, perm=[1, 0, 2])
        
        h_pattern_char = tf.layers.dropout(h_pattern_char, rate=dropout_rate, training=training)
        h_relation_char = tf.layers.dropout(h_relation_char, rate=dropout_rate, training=training)
        
        h_entity1_char = tf.expand_dims(entity_mask1,axis=-1) * h_pattern_char
        h_entity1_char = tf.div_no_nan(tf.reduce_sum(h_entity1_char, axis=1), tf.expand_dims(tf.reduce_sum(entity_mask1, axis=1),axis=-1))
        h_entity2_char = tf.expand_dims(entity_mask2,axis=-1) * h_pattern_char
        h_entity2_char = tf.div_no_nan(tf.reduce_sum(h_entity2_char, axis=1), tf.expand_dims(tf.reduce_sum(entity_mask1, axis=1),axis=-1))
        
        # Aggregation
        if config.aggregation == 'max':
            emb_size = tf.shape(h_pattern_char)[-1]
            pattern_mask = tf.tile(tf.expand_dims(tf.cast(tf.equal(patterns, 0), dtype=tf.float32), axis=-1), [1,1,emb_size])
            relation_mask = tf.tile(tf.expand_dims(tf.cast(tf.equal(relations, 0), dtype=tf.float32), axis=-1), [1,1,emb_size])
            h_pattern_char = tf.reduce_max(h_pattern_char - pattern_mask*1000, axis=1)
            h_relation_char = tf.reduce_max(h_relation_char - relation_mask*1000, axis=1)
        elif config.aggregation == 'mean':
            emb_size = tf.shape(h_pattern_char)[-1]
            pattern_mask = tf.expand_dims(tf.cast(tf.not_equal(patterns, 0), dtype=tf.float32), axis=-1)
            relation_mask = tf.expand_dims(tf.cast(tf.not_equal(relations, 0), dtype=tf.float32), axis=-1)
            h_pattern_char = tf.div_no_nan(tf.reduce_sum(h_pattern_char*pattern_mask), tf.count_nonzero(patterns, axis=-1, dtype=tf.float32, keepdims=True))
            h_relation_char = tf.div_no_nan(tf.reduce_sum(h_relation_char*relation_mask), tf.count_nonzero(relations, axis=-1, dtype=tf.float32, keepdims=True))
        elif config.word_aggregation == 'end':
            h_pattern_char = results_fw_char + results_bw_char
            h_relation_char = results_fw_relation_char + results_bw_relation_char
        elif config.aggregation == 'attention':
            assert False, 'nor implemented error'
        
#        # [batch_size, max_len, char_emb_size+word_emb_size]
#        h_pattern = tf.concat([h_pattern_char, h_pattern_word], axis=-1)
#        # [relation_size, relation_max_len, char_emb_size+word_emb_size]
#        h_relation = tf.concat([h_relation_char, h_relation_word], axis=-1)
#        
#        h_entity1 = tf.expand_dims(tf.concat([h_entity1_char, h_entity1_word], axis=-1), axis=1)
#        h_entity2 = tf.expand_dims(tf.concat([h_entity2_char, h_entity2_word], axis=-1), axis=1)
#        
        h_entity1 = tf.expand_dims(h_entity1_char, axis=1)
        h_entity2 = tf.expand_dims(h_entity2_char, axis=1)
        # [batch_size, 2, emb_size]
        h_entity = tf.concat([h_entity1, h_entity2], axis=1)
        
        h_pattern = h_pattern_char
        h_relation = h_relation_char

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
            h_pattern = tf.layers.dropout(h_pattern, rate=dropout_rate, training=training)
        pattern = semantic_proj(h_pattern)
        pattern = tf.layers.dropout(pattern, rate=dropout_rate, training=training)
        
        for i in range(config.layer_num):
            h_relation = dense_layers[i](h_relation)
            h_relation = tf.layers.dropout(h_relation, rate=dropout_rate, training=training)
        
        # [single_size+comb_size, 3, semantic_size]
        h_relation_comb = tf.nn.embedding_lookup(h_relation, self.comb2relation_layer)
        w = tf.get_variable('w', shape=[3])
        h_relation = tf.einsum('j,ijk->ik', w, h_relation_comb)
#        relation = tf.reduce_max(relation_comb, axis=1)
        # [relation_size, semantic_size]
        relation = semantic_proj(h_relation)
        relation = tf.layers.dropout(relation, rate=dropout_rate, training=training)
        
        h_entity = semantic_proj(h_entity)
#        # [single_size+comb_size, 3, semantic_size]
#        relation_comb = tf.nn.embedding_lookup(relation, self.comb2relation_layer)
#        w = tf.get_variable('w', shape=[3])
#        relation = tf.einsum('j,ijk->ik', w, relation_comb)
##        relation = tf.reduce_max(relation_comb, axis=1)
        
        # [batch_size]
        norm_p = tf.expand_dims(tf.norm(pattern, axis=-1), axis=-1)
        pattern = tf.div_no_nan(pattern, norm_p)
        # [single_size+comb_size]
        norm_r = tf.expand_dims(tf.norm(relation, axis=-1), axis=-1)
        relation = tf.div_no_nan(relation, norm_r)
        # [batch_size, single_size+comb_size]
        score = tf.matmul(pattern, relation, transpose_b=True) * config.gamma
        
        single_mask = tf.concat([tf.zeros(shape=[config.single_size]),tf.ones(shape=[config.comb_size])], axis=0)
        cvt_mask = tf.concat([tf.ones(shape=[config.single_size]),tf.zeros(shape=[config.comb_size])], axis=0)
        mask = tf.nn.embedding_lookup(tf.stack([single_mask, cvt_mask]), inputs['typ'])
        score = score - mask * 10000

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
#            self.loss_op = tf.reduce_mean(prob)
#        else:
        labels_one_hot = tf.one_hot(tags, config.single_size + config.comb_size)
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=labels_one_hot,
                        logits=score))
        # [batch_size]
        self.infer_op = tf.argmax(score, -1)
        # [batch_size, 2, semantic_size]
        pred_relation = semantic_proj(tf.nn.embedding_lookup(h_relation_comb[:,:2,:], self.infer_op if not training else tags))
        
#        print 'h_entity', h_entity
#        print 'pred_relation', pred_relation
        score_order = tf.einsum('bij,bkj->bik', h_entity, pred_relation) * config.gamma
        coef = tf.constant([1,-1,-1,1], shape=[2,2], dtype=tf.float32)
        # [batch_size, 2]
        score_order = tf.einsum('bij,ji->bi', score_order, coef)

        order_one_hot = tf.one_hot(entity_order, 2)
        self.loss_op += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=order_one_hot,
                        logits=score_order) * tf.cast(inputs['typ'],tf.float32))*0.001
        
        self.infer_order_op = tf.argmax(score_order, -1)
        
        metric_layer = layers.EMMetricLayer()
        infer_total_op = self.infer_op * 2 + self.infer_order_op
        entity_total = tags * 2 + entity_order
#        self.metric = metric_layer(self.infer_op,tags,single_weights=tf.cast(1-inputs['typ'],tf.float32), cvt_weights=tf.cast(inputs['typ'],tf.float32))
        self.metric = metric_layer(self.infer_op,tags,self.infer_order_op,entity_order,infer_total_op,entity_total,single_weights=tf.cast(1-inputs['typ'],tf.float32), cvt_weights=tf.cast(inputs['typ'],tf.float32))
