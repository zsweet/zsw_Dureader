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
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np
from collections import Counter
import random


class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_para_num, max_p_len, max_q_len,train_files=[]):
        self.logger = logging.getLogger("brc")

        self.max_para_num = max_para_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        if 1:
            self.train_set, self.dev_set, self.test_set = [], [], []
            dataset = []
            print(train_files)
            if train_files:
                for train_file in train_files:

                    dataset += self._load_dataset(train_file, train=True)
            random.shuffle(dataset)
            self.train_set, self.dev_set, self.test_set = self.split_dataset_proportions(dataset)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

    def compute_rank(self, answer, fake_answer, number = 5):
        para_infos = []
        for answer_tokens in answer:
            fake_tokens = fake_answer
            common_with_question = Counter(answer_tokens) & Counter(fake_tokens)
            correct_preds = sum(common_with_question.values())
            if correct_preds == 0:
                recall_wrt_question = 0
            else:
                recall_wrt_question = float(correct_preds) / len(fake_tokens)
            if answer_tokens[0] == '.' and len(answer_tokens) == 2:
                recall_wrt_question = -1
            para_infos.append((answer_tokens, recall_wrt_question, len(answer_tokens)))
        para_infos.sort(key=lambda x: (-x[1], x[2]))

        rank_bag = []

        # Select top number para
        for i in range(number):
            fake_passage_tokens = []
            for para_info in para_infos[i:i+1]:
                fake_passage_tokens += para_info[0]
            rank_bag.append(fake_passage_tokens)
        return rank_bag

    def split_dataset_proportions(self, dataset):
        #train_prop, val_prop, test_prop =[0.95, 0.025, 0.025]
        train_prop, val_prop, test_prop = [0.8, 0.1, 0.1]

        # Split into train, val, and test sets given proportions.
        train_n = int(len(dataset) * train_prop)
        val_n = int((len(dataset) * val_prop))

        train_set = dataset[:train_n]
        val_set = dataset[train_n: train_n + val_n]
        test_set = dataset[train_n + val_n:]

        return train_set, val_set, test_set

    def _load_dataset(self, data_path,max_para = 5, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path) as fin:
            data_set = []
            for lidx, line in enumerate(fin):

                sample = json.loads(line.strip())

                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                sample['question_tokens'] = sample['segmented_question']

                for d_idx, doc in enumerate(sample['documents']):
                    sample['passages'] = []
                    rank = [1, 0, 0, 0, 0]
                    refer_p = doc['segmented_paragraphs']

                    # PAD the passage
                    if len(refer_p) < max_para:
                        for i in range(max_para - len(refer_p)):
                            refer_p.append(['.', '.'])
                    if train:
                        most_related_para = doc['most_related_para']
                        gold_p = doc['segmented_paragraphs'][most_related_para]
                        rank_p = self.compute_rank(refer_p, gold_p)

                        pack_p = list(zip(rank_p, rank))
                        random.shuffle(pack_p)
                        rank_p[:],rank[:] = zip(*pack_p)

                        sample['passages'].append(
                            {'passage_rank': rank_p}
                        )
                        sample['rank'] = rank

                    if len(rank) == 0:
                        continue
                    else:
                        data_set.append(sample)
        return data_set


    def _get_max_para_in_passage(self,batch_data):
        res = 0
        for sample in batch_data['raw_data']:
            for passa in sample['passages']:
                res = max(res,len(passa))
        return res

    def _one_mini_batch(self, data,  indices, pad_id, max_para = 5):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_length':[],
                      'passage_token_ids': [],
                      'label': []
                      }

        for sidx, sample in enumerate(batch_data['raw_data']):
           for pidx in range(max_para):
               batch_data['question_token_ids'].append(sample['question_token_ids'])
               batch_data['question_length'].append(len(sample['question_token_ids']))
               passage_token_ids = sample['passages'][0]['passage_token_ids'][pidx]
               batch_data['passage_token_ids'].append(passage_token_ids)
               batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
               rank_id = sample['rank'][pidx]
               batch_data['label'].append(rank_id)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)

        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages'][0]['passage_rank']:
                    for token in passage:
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None: continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['question_tokens'])
                passage_list = sample['passages'][0]['passage_rank']
                token_list = []
                for passage in range(len(passage_list)):
                    token_passage = vocab.convert_to_ids(passage_list[passage])
                    token_list.append(token_passage)
                sample['passages'][0]['passage_token_ids'] = token_list


    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)

#
#
#
#
#
#
#
#
#
#
#
#
# # -*- coding:utf8 -*-
# # ==============================================================================
# # Copyright 2017 Baidu.com, Inc. All Rights Reserved
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #    http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# """
# This module implements data process strategies.
# """
#
# import os
# import json
# import logging
# import numpy as np
# from collections import Counter
#
#
# class BRCDataset(object):
#     """
#     This module implements the APIs for loading and using baidu reading comprehension dataset
#     """
#     def __init__(self, max_p_num,max_para_num, max_p_len, max_q_len,train_files=[], dev_files=[], test_files=[]):
#         self.logger = logging.getLogger("brc")
#         self.max_p_num = max_p_num
#         self.max_para_num = max_para_num
#         self.max_p_len = max_p_len
#         self.max_q_len = max_q_len
#
#         self.train_set, self.dev_set, self.test_set = [], [], []
#         if train_files:
#             for train_file in train_files:
#                 self.train_set += self._load_dataset(train_file, train=True)
#             self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))
#
#         if dev_files:
#             for dev_file in dev_files:
#                 self.dev_set += self._load_dataset(dev_file,train=True)
#             self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))
#
#         if test_files:
#             for test_file in test_files:
#                 self.test_set += self._load_dataset(test_file,train=True)
#             self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))
#
#     def _load_dataset(self, data_path, train=False):
#         """
#         Loads the dataset
#         Args:
#             data_path: the data file to load
#         """
#         with open(data_path,'r', encoding='UTF-8') as fin:
#             data_set = []
#             for lidx, line in enumerate(fin):
#                 sample = json.loads(line.strip())
#
#                 if train:
#                     if len(sample['answer_spans']) == 0:
#                         continue
#                     if sample['answer_spans'][0][1] >= self.max_p_len:
#                         continue
#
#                 if 'answer_docs' in sample:
#                     sample['answer_passages'] = sample['answer_docs']
#
#                 sample['question_tokens'] = sample['segmented_question']
#
#                 sample['passages'] = []
#                 for d_idx, doc in enumerate(sample['documents']):
#                     if train:  ###此处有修改
#                         para = []
#                         for paraToken in  doc['segmented_paragraphs']:
#                             para.append({'para_tokens':paraToken})
#
#                         sample['passages'].append(
#                             {
#                                 'para_list': para,
#                                 'most_related_para':doc['most_related_para'],
#                                 'is_selected': doc['is_selected']
#                              }
#                         )
#                     else:
#                         para_infos = []
#                         for para_tokens in doc['segmented_paragraphs']:
#                             question_tokens = sample['segmented_question']
#                             common_with_question = Counter(para_tokens) & Counter(question_tokens)
#                             correct_preds = sum(common_with_question.values())
#                             if correct_preds == 0:
#                                 recall_wrt_question = 0
#                             else:
#                                 recall_wrt_question = float(correct_preds) / len(question_tokens)
#                             para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
#                         para_infos.sort(key=lambda x: (-x[1], x[2]))
#                         fake_passage_tokens = []
#                         for para_info in para_infos[:1]:
#                             fake_passage_tokens += para_info[0]
#                         sample['passages'].append({'passage_tokens': fake_passage_tokens})
#                 data_set.append(sample)
#         return data_set
#
#
#     def _get_max_para_in_passage(self,batch_data):
#         res = 0
#         for sample in batch_data['raw_data']:
#             for passa in sample['passages']:
#                 res = max(res,len(passa))
#         return res
#
#     def _one_mini_batch(self, data, indices, pad_id):
#         """
#         Get one mini batch
#         Args:
#             data: all data
#             indices: the indices of the samples to be selected
#             pad_id:
#         Returns:
#             one batch of data
#         """
#         batch_data = {'raw_data': [data[i] for i in indices],
#                       'sample_list':[],
#                       'max_passage_num' :-1,
#                       'max_paragraph_num':-1
#                       }
#
#
#         max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
#         max_passage_num = min(self.max_p_num, max_passage_num)
#         max_paragraph_num = self._get_max_para_in_passage(batch_data)
#         max_paragraph_num = min(self.max_para_num,max_paragraph_num)
#         batch_data['max_passage_num']= max_passage_num
#         batch_data['max_paragraph_num'] = max_paragraph_num
#
#         for sidx, sample in enumerate(batch_data['raw_data']):
#             tmp_sample = []
#             for pass_idx in range(max_passage_num):
#                 document_data = {'question_token_ids': [], 'question_length': [], 'passage_token_ids': [],
#                                  'passage_length': [], 'most_related_para': []}
#                 if pass_idx < len(sample['passages']):
#                     for para_idx in range(max_paragraph_num):
#                         if para_idx < len(sample['passages'][pass_idx]):
#                             document_data['question_token_ids'].append(sample['question_token_ids'])
#                             document_data['question_length'].append(len(sample['question_token_ids']))
#                             paragraph_token_ids = sample['passages'][pass_idx][para_idx]['passage_token_ids']
#                             document_data['passage_token_ids'].append(paragraph_token_ids)
#                             document_data['passage_length'].append(min(len(paragraph_token_ids), self.max_p_len))
#                             is_most_rela_para = bool(sample[pass_idx]['most_related_para'] == para_idx)
#                             document_data['most_related_para'].append(is_most_rela_para)
#                         else:
#                             document_data['question_token_ids'].append([])
#                             document_data['question_length'].append(0)
#                             document_data['passage_token_ids'].append([])
#                             document_data['passage_length'].append(0)
#                             document_data['most_related_para'].append(bool(0))
#                 else :
#                     for idx in range(max_passage_num):
#                         document_data['question_token_ids'].append([])
#                         document_data['question_length'].append(0)
#                         document_data['passage_token_ids'].append([])
#                         document_data['passage_length'].append(0)
#                         document_data['most_related_para'].append(bool(0))
#                 tmp_sample.append(document_data)
#
#             batch_data['sample_list'].append(tmp_sample)
#         batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)
#         return batch_data
#
#     def _dynamic_padding(self, batch_data, pad_id):
#         """
#         Dynamically pads the batch_data with pad_id
#         """
#         pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
#         pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
#         for sidx, sample in enumerate(batch_data['sample_list']):
#             for passage in  sample:
#                 for para_idx,para in enumerate(passage['passage_token_ids']):
#                     tmp_para = (para+[pad_id] * (pad_p_len - len(para)))[:pad_p_len]
#                     passage['passage_token_ids'][para_idx]  = tmp_para
#                 for que_idx,que in enumerate(passage['question_token_ids']):
#                     tmp_que = (que + [pad_id] * (pad_q_len - len(para)))[:pad_q_len]
#                     passage['question_token_ids'][que_idx] = tmp_que
#         return batch_data, pad_p_len, pad_q_len
#
#     def word_iter(self, set_name=None):
#         """
#         Iterates over all the words in the dataset
#         Args:
#             set_name: if it is set, then the specific set will be used
#         Returns:
#             a generator
#         """
#         if set_name is None:
#             data_set = self.train_set + self.dev_set + self.test_set
#         elif set_name == 'train':
#             data_set = self.train_set
#         elif set_name == 'dev':
#             data_set = self.dev_set
#         elif set_name == 'test':
#             data_set = self.test_set
#         else:
#             raise NotImplementedError('No data set named as {}'.format(set_name))
#         if data_set is not None:
#             for sample in data_set:
#                 for token in sample['question_tokens']:
#                     yield token
#                 for passage in sample['passages']:
#                     for token in passage['passage_tokens']:
#                         yield token
#
#     def convert_to_ids(self, vocab):
#         """
#         Convert the question and passage in the original dataset to ids
#         Args:
#             vocab: the vocabulary on this dataset
#         """
#         for data_set in [self.train_set, self.dev_set, self.test_set]:
#             if data_set is None: continue
#             for sample in data_set:
#                 for passage in sample:
#                     for para in passage:
#                         para['question_token_ids'] = vocab.convert_to_ids(para['question_tokens'])
#
#     def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
#         """
#         Generate data batches for a specific dataset (train/dev/test)
#         Args:
#             set_name: train/dev/test to indicate the set
#             batch_size: number of samples in one batch
#             pad_id: pad id
#             shuffle: if set to be true, the data is shuffled.
#         Returns:
#             a generator for all batches
#         """
#         if set_name == 'train':
#             data = self.train_set
#         elif set_name == 'dev':
#             data = self.dev_set
#         elif set_name == 'test':
#             data = self.test_set
#         else:
#             raise NotImplementedError('No data set named as {}'.format(set_name))
#         data_size = len(data)
#         indices = np.arange(data_size)
#         if shuffle:
#             np.random.shuffle(indices)
#         for batch_start in np.arange(0, data_size, batch_size):
#             batch_indices = indices[batch_start: batch_start + batch_size]
#             yield self._one_mini_batch(data, batch_indices, pad_id)
