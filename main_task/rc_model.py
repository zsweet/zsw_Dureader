# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import time
import logging
from tensorflow.python.layers import core as layers_core
import json
import numpy as np
import tensorflow as tf
from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder


class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab,vocab1, args):

        # logging
        self.logger = logging.getLogger("brc")
        #self.yes_no = args.yes_no
        #self.entity = args.entity

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1
        self.d_a = 350
        self.r = 500
        self.batch_size = args.batch_size
        self.train_as = args.train_as
        self.train1 = True

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        # the vocab
        self.vocab = vocab
        self.vocab1 = vocab1

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        #sess_config.gpu_options.per_process_gpu_memory_fraction = 0.55

        self.sess = tf.Session(config=sess_config)

        self.W_s1 = tf.get_variable('W_s1', shape=[self.d_a, 2 * self.hidden_size])
        # shape(W_s2) = r * d_a
        self.W_s2 = tf.get_variable('W_s2', shape=[self.r, self.d_a])

        self._build_graph()

        # save info
        if self.train_as:
            t_param = [var for var in tf.trainable_variables() if not 'answer_sys' in var.name]
            self.pr_saver = tf.train.Saver(t_param)
        else:
            self.pr_saver = tf.train.Saver()

        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        if self.train_as:
            self._setup_placeholders()
            self._embed()
            self._encode()
            self._answer_sys()
            self._compute_loss()
            self._create_train_op()
        else:
            self._setup_placeholders()
            self._embed()
            self._encode()
            self._match()
            self._fuse()
            self._decode()
            self._compute_loss()
            self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """

        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.a = tf.placeholder(tf.int32, [None, None])
        if self.train_as:
            self.start_array = tf.placeholder(tf.float32, [None, None])
            self.end_array = tf.placeholder(tf.float32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.a_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        ### Add label element
        # self.p_label = tf.placeholder(tf.float32, [None, None,None])
        # self.q_label = tf.placeholder(tf.float32, [None, None,None])
        self.label = tf.placeholder(tf.float32, [None, None])


    def dot_attention(self, inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
        def softmax_mask(val, mask):
            return -1e30 * (1 - tf.cast(mask, tf.float32)) + val
        with tf.variable_scope(scope):
            JX = tf.shape(inputs)[1]

            with tf.variable_scope("attention"):
                inputs_ = tf.nn.relu(
                    tf.layers.dense(inputs, hidden, use_bias=False, name="inputs"))
                memory_ = tf.nn.relu(
                    tf.layers.dense(memory, hidden, use_bias=False, name="memory"))
                outputs = tf.matmul(inputs_, tf.transpose(
                    memory_, [0, 2, 1])) / (hidden ** 0.5)
                mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
                # logits = attention 'a'
                logits = tf.nn.softmax(softmax_mask(outputs, mask))
                outputs = tf.matmul(logits, memory)
                res = tf.concat([inputs, outputs], axis=2)

            with tf.variable_scope("gate"):
                dim = res.get_shape().as_list()[-1]
                gate = tf.nn.sigmoid(tf.layers.dense(outputs, dim, use_bias=False))
                return res * gate


    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )

            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

           ### Add label element
            # self.label = tf.expand_dims(self.label, 1)
            # p_label = tf.tile(self.label, [1, tf.shape(self.p_emb)[1], 1])
            # q_label = tf.tile(self.label, [1, tf.shape(self.q_emb)[1], 1])

            p_label = tf.tile(self.label, [tf.shape(self.p_emb)[1], 1])
            p_label = tf.reshape(p_label, [tf.shape(self.p_emb)[0], -1, 5])
            q_label = tf.tile(self.label, [tf.shape(self.q_emb)[1], 1])
            q_label = tf.reshape(q_label, [tf.shape(self.q_emb)[0], -1, 5])
            #self.logger.info('p_emb shape :{}, q_emb :{} p_label:{} ,q_label: {}'.format(tf.shape(self.p_emb),tf.shape(self.q_emb),tf.shape(self.p_label),tf.shape(self.q_label)))
            self.p_emb = tf.concat([self.p_emb, p_label], 2)
            self.q_emb = tf.concat([self.q_emb, q_label], 2)

         #   self.logger.info('p_emb shape :{}, q_emb :{}'.format(tf.shape(self.p_emb),tf.shape(self.q_emb)))



    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size)
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, self.q_fin = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

        self.shape1 = tf.shape(self.sep_p_encodes)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """

        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.p_length, self.q_length)
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

        # Self attentive
        self.shape2 = tf.shape(self.match_p_encodes)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        self.c_mask = tf.cast(self.p, tf.bool)
        self.self_att = self.dot_attention(
            self.match_p_encodes, self.match_p_encodes, mask=self.c_mask, hidden=self.hidden_size, keep_prob=1.0)
        with tf.variable_scope('fusion'):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.self_att, self.p_length,
                                         self.hidden_size, layer_num=1)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

        #self.shape3 = tf.shape(self.fuse_p_encodes)
        #self.shape4 = tf.shape(self.self_att)
    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            self.shape5 = tf.shape(concat_passage_encodes)
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size]
            )[0:, 0, 0:, 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes,
                                                          no_dup_question_encodes)


    def _yesno(self):
        with tf.variable_scope('yesno'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            self.yesno1 = tf.layers.dense(tf.reduce_mean(concat_passage_encodes, 1), self.hidden_size, name = 'fc_1' )
            self.yesno2 = tf.layers.dense(self.yesno1, self.hidden_size/10, name = 'fc_2' )
            self.yn_out = tf.layers.dense(self.yesno2, 3, name='fc_3')


    def _entity_decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size]
            )[0:, 0, 0:, 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes,
                                                          no_dup_question_encodes)

    def _answer_sys(self):
        with tf.variable_scope('answer_sys'):

            if self.train_as:
                with tf.device('/cpu:0'), tf.variable_scope('word_embedding1'):
                    self.word_embeddings1 = tf.get_variable(
                        'word_embeddings1',
                        shape=(self.vocab1.size(), self.vocab1.embed_dim),
                        initializer=tf.constant_initializer(self.vocab1.embeddings),
                        trainable=True
                    )
                self.a_emb = tf.nn.embedding_lookup(self.word_embeddings, self.a)

            batch_size = tf.shape(self.start_label)[0]
            self.output_layer = layers_core.Dense(
                self.vocab1.size(), use_bias=False, name="output_projection")
            with tf.variable_scope('question_encoding_sys'):

                _, self.hidden_question = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
                #_, self.hidden_question = srnn('bi-lstm', self.q_emb, self.hidden_size)


            self.c_p_emb = tf.concat([self.p_emb, tf.expand_dims(self.start_array, -1), tf.expand_dims(self.end_array, -1)], axis = 2)
            with tf.variable_scope('passage_encoding_sys'):

                _, self.hidden_passage = rnn('bi-lstm', self.c_p_emb, self.p_length, self.hidden_size)
                #_, self.hidden_passage = srnn('bi-lstm', self.c_p_emb, self.hidden_size)
            self.init_input = tf.layers.dense(tf.concat([self.hidden_passage,self.hidden_question], axis = 1), self.hidden_size, name = 'init_fc')
            self.cell = tf.contrib.rnn.GRUCell(
                self.hidden_size)

            #self.a_emb = self.a_emb[::5, :, :]
            #a_length = self.a_length[::5]

            if self.train1:
                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.a_emb, self.a_length)
                my_decoder = tf.contrib.seq2seq.BasicDecoder(
                    self.cell,
                    helper,
                    self.init_input,
                    output_layer=self.output_layer
                )

                outputs, self.final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder)

                self.sample_id = outputs.sample_id

                self.logits = outputs.rnn_output
            else:
                start_tokens = tf.fill([batch_size], self.vocab1.get_id('<sos>'))
                end_token = self.vocab1.get_id('<eos>')

                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.word_embeddings1, start_tokens, end_token)


                #self.a_emb = self.a_emb[batch_size, 200, 2 * self.hidden_size]
                #self.a_length = tf.reshape(self.a_length, [5])
                # Decoder
                my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=self.cell,
                    embedding=self.word_embeddings1,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=self.init_input,
                    beam_width=5,
                    output_layer=self.output_layer,
                    length_penalty_weight=0.5)


                # Dynamic decoding
                outputs, self.final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    maximum_iterations=200)

                self.sample_id = outputs.sample_id

                self.logits = outputs.rnn_output

    def _compute_loss(self, yes_no = False, entity = False):
        """
        The loss function
        """
        batch_size = tf.shape(self.start_label)[0]
        self.all_params = tf.trainable_variables()
        #a_length = self.a_length[::5]

        if self.train_as:

            def get_max_time(tensor):
                time_axis = 1
                return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

            target_output = self.a

            max_time = get_max_time(target_output)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_output, logits=self.logits)
            target_weights = tf.sequence_mask(
                self.a_length, max_time, dtype=self.logits.dtype)
            self.oloss = target_output
            self.rloss = self.logits
            self.loss = tf.reduce_sum(crossent * target_weights)/ tf.to_float(batch_size)
        else:
            def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
                """
                negative log likelyhood loss
                """
                with tf.name_scope(scope, "log_loss"):
                    labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                    losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
                return losses

            self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
            self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)

            self.point_loss = tf.add(self.start_loss, self.end_loss)



            self.loss = tf.reduce_mean(self.point_loss)

            if self.weight_decay > 0:
                with tf.variable_scope('l2_loss'):
                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
                self.loss += self.weight_decay * l2_loss




    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))

        def get_variables_with_name(name, train_only=True, printable=False):
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if (name in var.name)]
            return d_vars
        if self.train_as:
            trainable_params = get_variables_with_name('answer_sys')
            gradients = tf.gradients(self.loss, trainable_params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, trainable_params))
        else:
            self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        self.train1 = True
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.p : batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: dropout_keep_prob,
                         self.a : batch['answer_token_ids'],
                         self.a_length: batch['answer_length'],
                         # self.p_label:batch['p_label'],
                         # self.q_label:batch['q_label']
                         self.label: batch['label']
                         }
            # self.logger.info(tf.shape(batch['label']))
            # self.logger.info(self.sess.run(self.p_emb),feed_dict)
            # self.logger.info(self.sess.run(self.q_emb),feed_dict)
            #self.logger.info('s1 {} '.format(batch['passage_length']))
            _, loss= self.sess.run([self.train_op, self.loss], feed_dict)
            #self.logger.info('loss {} \n oloss {} \n sample_id {} \n, logit {}'.format(loss, oloss,sampleid,logit))
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            #self.logger.info('s1 {} s2 {} s3 {} s4 {} s5 {}'.format(s1, s2, s3, s4, s5))
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))

                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_bleu_4 = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir, save_prefix+ '_' + str(epoch))
                        max_bleu_4 = bleu_rouge['Bleu-4']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        self.train1 = False
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: 1.0,
                         self.a : batch['answer_token_ids'],
                         self.a_length: batch['answer_length'],
                         # self.p_label:batch['p_label'],
                         # self.q_label:batch['q_label']
                         self.label: batch['label']
                         }
            if self.train_as:
                sample_ids, loss = self.sess.run([self.sample_id,self.loss], feed_dict)
            else:
                start_probs, end_probs, loss = self.sess.run([self.start_probs,
                                                          self.end_probs, self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            #self.logger.info('example {}'.format(sample_ids[0]))
            if self.train_as:

                for sample, sample_id in zip(batch['raw_data'], sample_ids):

                    best_answer = ''.join(self.vocab1.recover_from_ids(sample_id))
                    if save_full_info:
                        sample['pred_answers'] = [best_answer]
                        pred_answers.append(sample)
                    else:
                        pred_answers.append({'question_id': sample['question_id'],
                                             'question_type': sample['question_type'],
                                             'answers': [best_answer],
                                             'o_answers': sample['answers'],
                                             'entity_answers': [[]],
                                             'yesno_answers': []})
                    if 'answers' in sample:
                        ref_answers.append({'question_id': sample['question_id'],
                                            'question_type': sample['question_type'],
                                            'answers': sample['answers'],
                                            'entity_answers': [[]],
                                            'yesno_answers': []})
                #self.logger.info('example {}'.format(best_answer))

            else:

                for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                    best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)
                    if save_full_info:
                        sample['pred_answers'] = [best_answer]
                        pred_answers.append(sample)
                    else:
                        pred_answers.append({'question_id': sample['question_id'],
                                             'question_type': sample['question_type'],
                                             'answers': [best_answer],
                                             'entity_answers': [[]],
                                             'yesno_answers': []})
                    if 'answers' in sample:
                        ref_answers.append({'question_id': sample['question_id'],
                                             'question_type': sample['question_type'],
                                             'answers': sample['answers'],
                                             'entity_answers': [[]],
                                             'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer,  ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.pr_saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))


