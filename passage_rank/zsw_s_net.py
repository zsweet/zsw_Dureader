import tensorflow as tf
from basic_rnn import rnn
import tensorflow.contrib as tc
import logging
import time
import numpy as np
from func import *
from dataset import BRCDataset
import os
import json

class S_netModel():
    def __init__(self,vocab, args):
        # logging
        self.logger = logging.getLogger("brc")
        self.args = args
        # the vocab
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1
        self.d_a = 350
        self.r = 500
        self.max_para = args.max_p_num
        self.batch_size = args.batch_size
        self.train_as = args.train_as
        self.vocab = vocab

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
       """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()  # The embedding layer, question and passage share embeddings
        self._encode()  # Employs two Bi-LSTMs to encode passage and question separately
        self._weightParagraphAttention()  # The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        self._weightQestionAttention()  # Employs Bi-LSTM again to fuse the context information after match layer
        self._get_result()  # Employs Pointer Network to get the the probs of each position

        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        self.all_params = tf.trainable_variables()
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))




    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.l = tf.placeholder(tf.int32, [None])

        self.dropout_keep_prob = tf.placeholder(tf.float32)

        self.class_label = tf.placeholder(tf.float32,[None,None])

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
           # self.p_mask = tf.cast(self.p_emb, tf.bool)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)
          #  self.p_mask = tf.cast(self.p_emb, tf.bool)
           
            p_label = tf.tile(self.class_label, [tf.shape(self.p_emb)[1], 1])
            p_label = tf.reshape(p_label, [tf.shape(self.p_emb)[0], -1, 5])
            q_label = tf.tile(self.class_label, [tf.shape(self.q_emb)[1], 1])
            q_label = tf.reshape(q_label, [tf.shape(self.q_emb)[0], -1, 5])
            
            self.p_emb = tf.concat([self.p_emb, p_label], 2)
            self.q_emb = tf.concat([self.q_emb, q_label], 2)

          

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size,dropout_keep_prob=self.dropout_keep_prob)
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size,dropout_keep_prob=self.dropout_keep_prob)  ##???????????
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)  ###T*300
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)  ###J*300
        #print ('paragram',self.sep_p_encodes,'qestion',self.sep_q_encodes)



    def _weightQestionAttention(self):
        with tf.variable_scope('weight_question_attention'):
            self.q_mask = tf.cast(self.q, tf.bool)
            self.weight_q_encodes = summ(self.sep_q_encodes[:, :, -2 * self.hidden_size:], self.hidden_size, mask=self.q_mask,keep_prob=self.use_dropout, is_train=False)

    def _weightParagraphAttention(self):
        with tf.variable_scope('weight_p_attention'):
            atten = attention(self.hidden_size)
            self.p_attn = atten(self.sep_p_encodes, self.sep_q_encodes)

            self.atten_output, _ = rnn('bi-lstm', self.p_attn, self.p_length, self.hidden_size)


    def _get_result(self):
        self.c_mask = tf.cast(self.p, tf.bool)
        #attenOutputs = tf.concat([self.weight_q_encodes,self.weight_p_encodes], -1)

        G=None
        batch_size = None

        reuse_flag = False
        for i in range(self.max_para):
            vP = self.atten_output[i::self.max_para]
            batch_size = tf.shape(vP)[0]
            pr_att = pr_attention(batch=batch_size, hidden=tf.shape(self.weight_q_encodes[i::self.max_para])[1], keep_prob=1)
            r_P = pr_att(self.weight_q_encodes[i::self.max_para], vP, self.hidden_size, self.c_mask[i::self.max_para],reuse_flag = reuse_flag)
            reuse_flag = True
            concatenate = tf.concat([self.weight_q_encodes[i::self.max_para], r_P], axis=1)
            g = tf.nn.tanh(dense(concatenate, hidden=self.hidden_size, use_bias=False, scope="g" + str(i)))
            g_ = dense(g, 1, use_bias=False, scope="g_" + str(i))
            if i == 0:
                G = g_
            else:
                G = tf.concat([G, g_], axis = 1)

        self.lable1 = tf.reshape(self.l, [batch_size, -1])
        self.rank = tf.nn.softmax(G)
        self.loss = tf.losses.softmax_cross_entropy(
            self.lable1, G)

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
        for bitx, batch in enumerate(train_batches, 1):

            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.l: np.array(batch['label']),
                         self.dropout_keep_prob : 1.0,
                         self.class_label: batch['class_label']
                         }

            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
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
                    eval_loss, bleu_rouge = self.evaluate(eval_batches,result_dir=self.args.result_dir, result_prefix='dev.predicted',epoch_mark='_epoch'+str(epoch))
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))
                    self.save(save_dir, save_prefix+'_epoch'+str(epoch))  ####修改文件名称 保存结果
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_epoch' + str(epoch))


    def evaluate(self, eval_batches, result_dir=None, result_prefix=None,epoch_mark=""):
        pred_answers, ref_answers = [], []
        self.train1 = False
        total_loss, total_num = 0, 0
        count = 0
        total = 0
        score  = ave_loss = -1

        batch_count = 0
        for b_itx, batch in enumerate(eval_batches):
                batch_count += 1
                feed_dict = {self.p: batch['passage_token_ids'],
                             self.q: batch['question_token_ids'],
                             self.p_length: batch['passage_length'],
                             self.q_length: batch['question_length'],
                             self.l: batch['label'],
                             self.dropout_keep_prob: 1.0,
                             self.class_label: batch['class_label']
                             }
                rank, lable, loss = self.sess.run([self.rank, self.lable1, self.loss], feed_dict)

                total_loss += loss * len(batch['raw_data'])
                total_num += len(batch['raw_data'])
                padded_p_len = len(batch['passage_token_ids'][0])
                # self.logger.info('example {}'.format(sample_ids[0]))

                best_true = []
                best_pred = []
                for a, b in zip(rank, lable):

                    a = a.tolist()
                    b = b.tolist()
                    a1 = a.index(max(a))
                    b1 = b.index(max(b))

                    best_true.append(b1)
                    best_pred.append(a1)

                    if a1 == b1:
                        count += 1
                total += len(rank)
                tc = 0
                TC = 0
                for sample, q in zip(batch['passage_token_ids'], batch['question_token_ids']):
                    tc += 1
                    passage1 = ''.join(self.vocab.recover_from_ids(sample))
                    passage2 = passage1.replace('<blank>', '')
                    q1 = ''.join(self.vocab.recover_from_ids(q))
                    q2 = q1.replace('<blank>', '')
                    pred_answers.append({'passage': passage2})
                    if tc % self.max_para == 0:
                        tc = 0

                        pred_answers.append({'Question': q2,
                                             'True': [best_true[TC]],
                                             'Pred': [best_pred[TC]]})
                        TC += 1
                if result_dir is not None and result_prefix is not None:
                    fileName = result_prefix + epoch_mark
                    result_file = os.path.join(result_dir, fileName)
                    with open(result_file, 'w') as fout:
                        for pred_answer in pred_answers:
                            fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

                score = 1.0 * count / total
                ave_loss = 1.0 * total_loss / total_num
                if total % 1000 == 0:
                    self.logger.info('Right {} of Total {},accuracy rate {}'.format(count, total,count*1.0/total))
        self.logger.info('Right {} of Total {},accuracy rate {}'.format(count, total, count * 1.0 / total))
        # compute the bleu and rouge scores if reference answers is provided
        return ave_loss, score


    def predict(self, eval_batches, result_dir=None, result_prefix=None):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        count = total = dc = 0
        fout = open(os.path.join(result_dir, result_prefix), 'w')
        for b_itx, batch in enumerate(eval_batches):
                feed_dict = {self.p: batch['passage_token_ids'],
                             self.q: batch['question_token_ids'],
                             self.p_length: batch['passage_length'],
                             self.q_length: batch['question_length'],
                             self.l: batch['label'],
                             self.dropout_keep_prob: 1.0,
                             self.class_label: batch['class_label']
                             }
                rank = self.sess.run([self.rank], feed_dict)
                CT= 0
                best = []

                for a in rank[0]:
                    a = a.tolist()
                    a1 = a.index(max(a))
                    best.append(a1)

                pred_answers = []
                for qid, docID, index_bag in zip(batch['question_id'], batch['doc_idx'], batch['index']):  ############

                    if CT % self.max_para ==0:
                        rdx = CT/self.max_para
                        pred_answers.append({'question_id' : qid,
                                             'best' : index_bag[best[int(rdx)]],  ####################
                                             'doc_num' : docID})
                        dc += 1
                        if dc % 10000 == 0:
                            self.logger.info('Question saved: {}'.format(dc))
                    CT += 1
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')



    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix, sys_dir = None, sys_prefix = None):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """

        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))

