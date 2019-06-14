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
        if not config.word.use_word_pretrain_emb:
            relation2word_emb = tf.constant(np.load(config.word.relation2word))
            self.relation2word_emb_layer = tf.get_variable('relation2word_emb', initializer=relation2word_emb,
                                                      trainable=False)
        else:
            relation2word_emb = tf.constant(np.load(config.word.relation2word_emb))
            self.relation2word_emb_layer = tf.get_variable('relation2word_emb', initializer=relation2word_emb,
                                                      trainable=False)
        comb2relation = tf.constant(np.load(config.comb2relation))
        self.comb2relation_layer = tf.get_variable('comb2relation', initializer=comb2relation,
                                                  trainable=False)

        if config.char.encoder_type == 'trigram':
            relation2gram_emb = tf.constant(np.load(config.char.relation2gram))
            self.relation2gram_emb_layer = tf.get_variable('relation2gram_emb', initializer=relation2gram_emb,
                                                  trainable=False)
        else:
            relation2char_emb = tf.constant(np.load(config.char.relation2char))
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
        patterns_word = inputs['word_emb']
        # [relation_size, relation_max_len]
        relations_word = self.relation2word_emb_layer
        # [batch_size, max_char]
        chars = inputs['char']
        grams = inputs['gram']
        # [relation_size, relation_max_char]
        if config.char.encoder_type == 'trigram':
            relations_char = self.relation2gram_emb_layer
        else:
            relations_char = self.relation2char_emb_layer

#        # [batch_size]
        tags = inputs['relation']
        entity_order = inputs['entity_order']

        if config.train_order:
    #        mask_1 = inputs['entity_mask1']
    #        mask_2 = inputs['entity_mask2']
    #        
            entity_mask1 = inputs['entity_mask1']
            entity_mask2 = inputs['entity_mask2']
            entity_mask3 = inputs['entity_mask3']
            entity_mask4 = inputs['entity_mask4']
        
# #############################################################################################
#        # Encoding
#        # Word-level Encoder
#        if not config.word.use_word_pretrain_emb:
#            patterns_emb = inputs['word']
#            word_emb_layer = layers.EmbeddingLayer(config.vocab_size_train, config.word.emb_size, name='word_emb')
#        else:
#            patterns_emb = inputs['word_emb']
#            word_emb_layer = layers.InitializedEmbeddingLayer(config.vocab_size_emb, config.word.emb_size, config.word.word2vec_emb,
#                                                          trainable=config.word.word_emb_finetune,
#                                                          name='word_emb')
#        relations_emb = self.relation2word_emb_layer
#
#        if config.word.encoder == 'none':
#            patterns_emb = word_emb_layer(patterns_emb, zero_forward=True)
#            relations_emb = word_emb_layer(relations_emb, zero_forward=True)
#        elif config.word.encoder == 'mean':
#            word_encoder = layers.MeanEncoder(config.vocab_size_emb, config.word.emb_size, config.word.region_radius,
#                                              name='word')
#            patterns_emb = word_encoder(patterns_emb)
#            relations_emb = word_encoder(relations_emb)
#        elif config.word.encoder == 'region':
#            word_encoder = layers.RegionEncoder(config.vocab_size_emb, config.word.emb_size, config.word.region_radius,
#                                                name='word')
#            patterns_emb = word_encoder(patterns_emb)
#            relations_emb = word_encoder(relations_emb)
#        elif config.word.encoder == 'CNN':
#            word_encoder = layers.CNNEncoder(config.vocab_size_emb, config.word.emb_size, config.word.groups, config.word.filters, config.word.kernel_size,
#                                             name='word', emb=word_emb_layer)
#            patterns_emb = word_encoder(patterns_emb)
#            relations_emb = word_encoder(relations_emb)
#        
#        patterns_emb = tf.layers.dropout(patterns_emb, rate=dropout_rate, training=training)
#        relations_emb = tf.layers.dropout(relations_emb, rate=dropout_rate, training=training)
#
#        if config.word.use_rnn:
#            h_pattern_word = tf.transpose(patterns_emb, perm=[1, 0, 2])
#            h_relation_word = tf.transpose(relations_emb, perm=[1, 0, 2])
#            nwords = tf.count_nonzero(patterns_word, axis=-1, dtype=tf.int32)
#            relation_nwords = tf.count_nonzero(relations_word, axis=-1, dtype=tf.int32)
#            # Bi-LSTM
#            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(config.word.hidden_size)
#            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(config.word.hidden_size)
#            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
#            with tf.variable_scope('lstm1'):
#                output_fw, (_, results_fw) = lstm_cell_fw(h_pattern_word, dtype=tf.float32, sequence_length=nwords)
#                output_fw = tf.layers.dropout(output_fw, rate=dropout_rate, training=training)
#                output_fw_relation, (_, results_fw_relation) = lstm_cell_fw(h_relation_word, dtype=tf.float32, sequence_length=relation_nwords)
#                output_fw_relation = tf.layers.dropout(output_fw_relation, rate=dropout_rate, training=training)
#            with tf.variable_scope('lstm2'):
#                output_bw, (_, results_bw) = lstm_cell_bw(h_pattern_word, dtype=tf.float32, sequence_length=nwords)
#                output_bw = tf.layers.dropout(output_bw, rate=dropout_rate, training=training)
#                output_bw_relation, (_, results_bw_relation) = lstm_cell_bw(h_relation_word, dtype=tf.float32, sequence_length=relation_nwords)
#                output_bw_relation = tf.layers.dropout(output_bw_relation, rate=dropout_rate, training=training)
#
#            h_pattern_word = tf.concat([output_fw, output_bw], axis=-1)
#            h_pattern_word = tf.transpose(h_pattern_word, perm=[1, 0, 2])
#            h_relation_word = tf.concat([output_fw_relation, output_bw_relation], axis=-1)
#            h_relation_word = tf.transpose(h_relation_word, perm=[1, 0, 2])
#        else:
#            h_pattern_word = patterns_emb
#            h_relation_word = relations_emb
#            
#        h_pattern_word = tf.layers.dropout(h_pattern_word, rate=dropout_rate, training=training)
#        h_relation_word = tf.layers.dropout(h_relation_word, rate=dropout_rate, training=training)
#        
#        if config.train_order:
#    #        h_entity1_word = tf.div_no_nan(tf.reduce_sum(h_pattern_word[mask_1[0]:mask_1[1]], axis=1), tf.case(mask_1[1]-mask_1[0],dtype=tf.float32))
#    #        h_entity2_word = tf.div_no_nan(tf.reduce_sum(h_pattern_word[mask_1[2]:mask_1[3]], axis=1), tf.case(mask_1[2]-mask_1[3],dtype=tf.float32))
#            h_entity1_word = tf.expand_dims(entity_mask1,axis=-1) * h_pattern_word
#            h_entity1_word = tf.div_no_nan(tf.reduce_sum(h_entity1_word, axis=1), tf.expand_dims(tf.reduce_sum(entity_mask1, axis=1),axis=-1))
#            h_entity2_word = tf.expand_dims(entity_mask2,axis=-1) * h_pattern_word
#            h_entity2_word = tf.div_no_nan(tf.reduce_sum(h_entity2_word, axis=1), tf.expand_dims(tf.reduce_sum(entity_mask1, axis=1),axis=-1))
#
#        if config.word.aggregation == 'max':
#            emb_size = tf.shape(h_pattern_word)[-1]
#            pattern_mask = tf.tile(tf.expand_dims(tf.cast(tf.equal(patterns_word, 0), dtype=tf.float32), axis=-1), [1,1,emb_size])
#            relation_mask = tf.tile(tf.expand_dims(tf.cast(tf.equal(relations_word, 0), dtype=tf.float32), axis=-1), [1,1,emb_size])
#            h_pattern_word = tf.reduce_max(h_pattern_word - pattern_mask*1000, axis=1)
#            h_relation_word = tf.reduce_max(h_relation_word - relation_mask*1000, axis=1)
#        elif config.word.aggregation == 'mean':
#            emb_size = tf.shape(h_pattern_word)[-1]
#            pattern_mask = tf.expand_dims(tf.cast(tf.not_equal(patterns_word, 0), dtype=tf.float32), axis=-1)
#            relation_mask = tf.expand_dims(tf.cast(tf.not_equal(relations_word, 0), dtype=tf.float32), axis=-1)
#            h_pattern_word = tf.div_no_nan(tf.reduce_sum(h_pattern_word*pattern_mask), tf.count_nonzero(patterns_word, axis=-1, dtype=tf.float32, keepdims=True))
#            h_relation_word = tf.div_no_nan(tf.reduce_sum(h_relation_word*relation_mask), tf.count_nonzero(relations_word, axis=-1, dtype=tf.float32, keepdims=True))
#        elif config.word.aggregation == 'end':
#            h_pattern_word = tf.concat([results_fw, results_bw], axis=-1)
#            h_relation_word = tf.concat([results_fw_relation, results_bw_relation], axis=-1)
#        elif config.word.aggregation == 'attention':
#            w_word = tf.get_variable('w_word', shape=[tf.shape(h_pattern_word)[-1]])
#            w = tf.math.sqrt(tf.cast(tf.shape(h_pattern_word)[-1],dtype=tf.float32))
#            M_pattern_word = tf.einsum('i,bli->bl', w_word, tf.nn.tanh(h_pattern_word))
#            alpha_pattern_word = tf.nn.softmax(M_pattern_word / w)
#            h_pattern_word = tf.einsum('bl,bli->bi', alpha_pattern_word, h_pattern_word)
#            
#            M_relation_word = tf.einsum('i,bli->bl', w_word, tf.nn.tanh(h_relation_word))
#            alpha_relation_word = tf.nn.softmax(M_relation_word / w)
#            h_relation_word = tf.einsum('bl,bli->bi', alpha_relation_word, h_relation_word)

# #########################################################################################
        # Char-level Encoder
        if config.char.encoder_type == 'trigram':
            vocab_size = config.trigram_size
            # [relation_size, max_char]
            patterns_emb_char = grams
            # [relation_size, relation_max_char]
            relations_emb_char = self.relation2gram_emb_layer
        else:
            vocab_size = config.char_size
            # [relation_size, max_char]
            patterns_emb_char = chars
            # [relation_size, relation_max_char]
            relations_emb_char = self.relation2char_emb_layer
        
        if config.word.encoder == 'none':
            char_emb_layer = layers.EmbeddingLayer(vocab_size, config.char.emb_size, name='char_emb')
            patterns_emb_char = char_emb_layer(patterns_emb_char, zero_forward=True)
            relations_emb_char = char_emb_layer(relations_emb_char, zero_forward=True)
        elif config.word.encoder == 'mean':
            char_encoder = layers.MeanEncoder(vocab_size, config.char.emb_size, config.char.region_radius, name='char_emb')
            patterns_emb_char = char_encoder(patterns_emb_char)
            relations_emb_char = char_encoder(relations_emb_char)
        elif config.word.encoder == 'region':
            char_encoder = layers.RegionEncoder(vocab_size, config.char.emb_size, config.char.region_radius, name='char_emb')
            patterns_emb_char = char_encoder(patterns_emb_char)
            relations_emb_char = char_encoder(relations_emb_char)
        elif config.word.encoder == 'CNN':
            char_encoder = layers.CNNEncoder(vocab_size, config.char.emb_size, config.char.groups, config.char.filters, config.char.kernel_size, name='char_emb')
            patterns_emb_char = char_encoder(patterns_emb_char)
            relations_emb_char = char_encoder(relations_emb_char)
#        if config.encoder == 'trigram':
#            trigram_emb_encoder = layers.TrigramEmbeddingEncoder(config.trigram_size, config.char_emb_size, config.region_radius,
#                                                                 aggregation=config.encoder_aggregation,
#                                                                 name='trigrams')
#            # [batch_size, max_len, max_char, char_emb_size]
#            pattern_grams = tf.nn.embedding_lookup(self.word2gram_emb_layer, patterns)
#            # [batch_size, max_len, char_emb_size]
#            h_pattern_char = trigram_emb_encoder(pattern_grams)
#            # [relation_size, relation_max_len, char_emb_size]
#            h_relation_char = trigram_emb_encoder(self.relation2gram_emb_layer)
#        elif config.encoder == 'char':
#            char_emb_encoder = layers.CharEmbeddingEncoder(config.char_size, config.char_emb_size,
#                                                           region_radius=config.region_radius,
#                                                           groups=config.groups,
#                                                           filters=config.filters,
#                                                           kernel_size=config.kernel_size,
#                                                           name='char')
#            # [batch_size, max_len, max_char]
#            pattern_chars = tf.nn.embedding_lookup(self.word2char_emb_layer, patterns)
#            # [batch_size, max_len, char_emb_size]
#            h_pattern_char = char_emb_encoder(pattern_chars)
#            # [relation_size, relation_max_len, char_emb_size]
#            h_relation_char = char_emb_encoder(self.relation2char_emb_layer)
        
        h_pattern_char = patterns_emb_char
        h_relation_char = relations_emb_char
        
        if config.char.use_rnn:
            h_pattern_char = tf.transpose(h_pattern_char, perm=[1, 0, 2])
            h_relation_char = tf.transpose(h_relation_char, perm=[1, 0, 2])
            nwords = tf.count_nonzero(chars, axis=-1, dtype=tf.int32)
            relation_nwords = tf.count_nonzero(relations_char, axis=-1, dtype=tf.int32)
            # Bi-LSTM
            lstm_cell_fw_char = tf.contrib.rnn.LSTMBlockFusedCell(config.char.hidden_size)
            lstm_cell_bw_char = tf.contrib.rnn.LSTMBlockFusedCell(config.char.hidden_size)
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
        
        if config.train_order:
    #        h_entity1_char = tf.div_no_nan(tf.reduce_sum(h_pattern_char[mask_2[0]:mask_2[1]], axis=1), tf.case(mask_2[1]-mask_2[0],dtype=tf.float32))
    #        h_entity2_char = tf.div_no_nan(tf.reduce_sum(h_pattern_char[mask_2[2]:mask_2[3]], axis=1), tf.case(mask_2[2]-mask_2[3],dtype=tf.float32))
            h_entity1_char = tf.expand_dims(entity_mask3,axis=-1) * h_pattern_char
            h_entity1_char = tf.div_no_nan(tf.reduce_sum(h_entity1_char, axis=1), tf.expand_dims(tf.reduce_sum(entity_mask1, axis=1),axis=-1))
            h_entity2_char = tf.expand_dims(entity_mask4,axis=-1) * h_pattern_char
            h_entity2_char = tf.div_no_nan(tf.reduce_sum(h_entity2_char, axis=1), tf.expand_dims(tf.reduce_sum(entity_mask1, axis=1),axis=-1))
        
        # Aggregation
        if config.char.aggregation == 'max':
            emb_size = tf.shape(h_pattern_char)[-1]
            pattern_mask = tf.tile(tf.expand_dims(tf.cast(tf.equal(chars, 0), dtype=tf.float32), axis=-1), [1,1,emb_size])
            relation_mask = tf.tile(tf.expand_dims(tf.cast(tf.equal(relations_char, 0), dtype=tf.float32), axis=-1), [1,1,emb_size])
            h_pattern_char = tf.reduce_max(h_pattern_char - pattern_mask*1000, axis=1)
            h_relation_char = tf.reduce_max(h_relation_char - relation_mask*1000, axis=1)
        elif config.char.aggregation == 'mean':
            emb_size = tf.shape(h_pattern_char)[-1]
            pattern_mask = tf.expand_dims(tf.cast(tf.not_equal(chars, 0), dtype=tf.float32), axis=-1)
            relation_mask = tf.expand_dims(tf.cast(tf.not_equal(relations_char, 0), dtype=tf.float32), axis=-1)
            h_pattern_char = tf.div_no_nan(tf.reduce_sum(h_pattern_char*pattern_mask), tf.count_nonzero(chars, axis=-1, dtype=tf.float32, keepdims=True))
            h_relation_char = tf.div_no_nan(tf.reduce_sum(h_relation_char*relation_mask), tf.count_nonzero(relations_char, axis=-1, dtype=tf.float32, keepdims=True))
        elif config.char.aggregation == 'end':
            h_pattern_char = tf.concat([results_fw_char, results_bw_char], axis=-1)
            h_relation_char = tf.concat([results_fw_relation_char, results_bw_relation_char], axis=-1)
        elif config.char.aggregation == 'attention':
            w_char = tf.get_variable('w_char', shape=[h_pattern_char.get_shape()[-1]])
            d = tf.math.sqrt(tf.cast(tf.shape(h_pattern_char)[-1],dtype=tf.float32))
            M_pattern_char = tf.einsum('i,bli->bl', w_char, tf.nn.tanh(h_pattern_char))
            alpha_pattern_char = tf.nn.softmax(M_pattern_char / d)
            h_pattern_char = tf.einsum('bl,bli->bi', alpha_pattern_char, h_pattern_char)
            
            M_relation_char = tf.einsum('i,bli->bl', w_char, tf.nn.tanh(h_relation_char))
            alpha_relation_char = tf.nn.softmax(M_relation_char / d)
            h_relation_char = tf.einsum('bl,bli->bi', alpha_relation_char, h_relation_char)

# ###########################################################################################       
#        # [batch_size, max_len, char_emb_size+word_emb_size]
#        h_pattern = tf.concat([h_pattern_char, h_pattern_word], axis=-1)
#        # [relation_size, relation_max_len, char_emb_size+word_emb_size]
#        h_relation = tf.concat([h_relation_char, h_relation_word], axis=-1)

        h_pattern = h_pattern_char
        h_relation = h_relation_char

        if config.train_order:
    #        h_entity1 = tf.expand_dims(tf.concat([h_entity1_char, h_entity1_word], axis=-1), axis=1)
    #        h_entity2 = tf.expand_dims(tf.concat([h_entity2_char, h_entity2_word], axis=-1), axis=1)
    #        
            h_entity1 = tf.expand_dims(h_entity1_char, axis=1)
            h_entity2 = tf.expand_dims(h_entity2_char, axis=1)
            # [batch_size, 2, emb_size]
            h_entity = tf.concat([h_entity1, h_entity2], axis=1)

        if config.use_highway:
            dense_t = tf.layers.Dense(h_pattern.get_shape()[-1],
                                      activation=tf.math.sigmoid,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                      name='dense_t')
            dense_h = tf.layers.Dense(h_pattern.get_shape()[-1],
                                      activation=tf.nn.tanh,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
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
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)))
        semantic_proj = tf.layers.Dense(config.semantic_size, name='semantic_proj',
                                        activation = tf.nn.tanh,
                                        use_bias = True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

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
        h_relation = tf.einsum('j,ijk->ik', tf.nn.softmax(w), h_relation_comb)
#        relation = tf.reduce_max(relation_comb, axis=1)
        # [relation_size, semantic_size]
        relation = semantic_proj(h_relation)
        relation = tf.layers.dropout(relation, rate=dropout_rate, training=training)
        
        if config.train_order:
            h_entity = semantic_proj(h_entity)
        
        # #########################################################################
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
        metric_layer = layers.EMMetricLayer()
        
        if config.train_order:
            # [batch_size, 2, semantic_size]
            infer_tags = self.infer_op if not training else tags
            pred_relation = semantic_proj(tf.nn.embedding_lookup(h_relation_comb[:,:2,:], infer_tags))
            
    ##        print 'h_entity', h_entity
    ##        print 'pred_relation', pred_relation
            score_order = tf.einsum('bij,bkj->bik', h_entity, pred_relation) * config.gamma
            coef = tf.constant([1,-1,-1,1], shape=[2,2], dtype=tf.float32)
            # [batch_size, 2]
            score_order = tf.einsum('bij,ji->bi', score_order, coef)
    
    #        order_one_hot = tf.one_hot(entity_order, 2)
    #        self.loss_op += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    #                        labels=order_one_hot,
    #                        logits=score_order) * tf.cast(inputs['typ'],tf.float32))*0.001
            
            self.infer_order_op = tf.argmax(score_order, -1)

            infer_total_op = self.infer_op * 2 + self.infer_order_op
            entity_total = tags * 2 + entity_order
            self.metric = metric_layer(self.infer_op,tags,self.infer_order_op,entity_order,infer_total_op,entity_total,single_weights=tf.cast(1-inputs['typ'],tf.float32), cvt_weights=tf.cast(inputs['typ'],tf.float32))
        else:
            self.metric = metric_layer(self.infer_op,tags,single_weights=tf.cast(1-inputs['typ'],tf.float32), cvt_weights=tf.cast(inputs['typ'],tf.float32))
