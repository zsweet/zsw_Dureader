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
This module prepares and runs the whole system.
"""

import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import BRCDataset

from vocab import Vocab
from zsw_s_net import S_netModel


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=30,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=5,
                                help='train epochs')
    train_settings.add_argument('--train_as', type=bool, default=True,
                                help='train epochs')


    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=6,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=400,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=150,
                                help='max length of answer')
    model_settings.add_argument('--max_train_sample_num', type=int, default=150000,
                                help='the max sample number from train dataset')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/preprocessed/trainset/search.train.json','../data/preprocessed/trainset/zhidao.train.json'],
                               help='list of files that contain the preprocessed train data')
    #path_settings.add_argument('--train_files', nargs='+',default=['../data/demo/trainset/search.train.json'], help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/preprocessed/devset/search.dev.json','../data/preprocessed/devset/zhidao.dev.json'],
                               help='list of files that contain the preprocessed dev data')

    #path_settings.add_argument('--test_files', nargs='+',
    #                           default=['../data/test1set/preprocessed/search.test1.json','../data/test1set/preprocessed/zhidao.test1.json','../data/preprocessed/devset/search.dev.json','../data/preprocessed/devset/zhidao.dev.json' ],
    #                           help='list of files that contain the preprocessed test data')

    path_settings.add_argument('--test_files', nargs='+',
                             default=['../data/broad_test/search.test.json','../data/broad_test/zhidao.test.json' ],
                             help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/pr',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/pr',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/pr',
                               help='the dir to output the results')
    path_settings.add_argument('--result_prefix', default='dev_result',
                               help='the dir to output file name the results')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    path_settings.add_argument('--word_embedding_path',default='../data/jwe_word2vec_size300.txt',
                               help='path of the word vector embedding.')

    return parser.parse_args()



def train(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')


    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,args.max_train_sample_num, args.train_files)

    vocab = Vocab(lower=True)


    for word in brc_data.word_iter('train'):
        vocab.add(word)



    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)

    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))


    logger.info('Assigning embeddings...')
    vocab.load_pretrained_embeddings(args.word_embedding_path )


    #vocab.randomly_init_embeddings(300)
    #vocab1.randomly_init_embeddings(300)
    logger.info('Saving vocab...')
    brc_data.convert_to_ids(vocab)
    logger.info('Initialize the model...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    rc_model = S_netModel(vocab, args)
    logger.info('Training the model...')
    #rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo +'sys')
    #if args.train_as:
    #    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo + 'syst')
    rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,save_prefix=args.algo,dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')

    logger.info('evaluate the trained model!')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,result_dir=args.result_dir, result_prefix='test.predicted')
    logger.info('Done with model evaluating !')

def evaluate(args):
    """
       predicts answers for test files
       """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
  #  assert len(args.test_files) > 0, 'No test files are provided.'


   # brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, args.max_train_sample_num,args.test_files, use_type="test")
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, args.max_train_sample_num,args.dev_files, use_type="dev")

    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)

    rc_model = S_netModel(vocab, args)
    logger.info('Restoring the model...')
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('evaluate answers for dev set...')
    test_batches = brc_data.gen_mini_batches('dev', args.batch_size,pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    #rc_model.predict(test_batches,result_dir=args.result_dir, result_prefix=args.result_prefix)
    rc_model.evaluate(test_batches, result_dir=args.result_dir, result_prefix=args.result_prefix)

def predict(args):
    """
       predicts answers for test files
       """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
  #  assert len(args.test_files) > 0, 'No test files are provided.'


    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, args.max_train_sample_num,args.test_files, use_type="test")
   # brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, args.max_train_sample_num,args.dev_files, use_type="dev")

    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)

    rc_model = S_netModel(vocab, args)
    logger.info('Restoring the model...')
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.predict(test_batches,result_dir=args.result_dir, result_prefix=args.result_prefix)
    #rc_model.evaluate(test_batches, result_dir=args.result_dir, result_prefix=args.result_prefix)

def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.train:
        train(args)
    if args.predict:
        predict(args)
    if args.evaluate:
        evaluate(args)
if __name__ == '__main__':
    run()
