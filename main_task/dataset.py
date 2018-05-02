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


class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len, max_a_len,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True, yes_no=True, entity = False, td=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file, td=True, dev=True)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False, yes_no = False, entity = False, td = False,dev = False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path,encoding='UTF-8') as fin:
            data_set = []

            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue
                    match = sample['match_scores'][0]
                    match = float(match)
                    if match < 0.7:
                        continue
                ### Here I add the additional 5 label
                label = [0.0, 0.0, 0.0, 0.0, 0.0]
                Type = sample['question_type']
                FO = sample['fact_or_opinion']
                if Type == 'ENTITY':
                    label[0] = 1.0
                elif Type == 'YES_NO':
                    label[1] = 1.0
                else:
                    label[2] = 1.0

                if FO == 'FACT':
                    label[3] = 1.0
                else:
                    label[4] = 1.0
                sample['label'] = label
                ### Add over
                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                sample['question_tokens'] = sample['segmented_question']

                if td:
                    try:
                        sample['answer_tokens'] = sample['segmented_answers'][0]
                        sample['answer_tokens'].insert(0,'<sos>')
                        sample['answer_tokens'].append('<eos>')
                    except:
                        continue
                else:
                    sample['answer_tokens'] = ['.']

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    if train:
                        most_related_para = doc['most_related_para']
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             'is_selected': doc['is_selected']}
                        )
                    ### DEV related
                    elif dev:
                        most_related_para = doc['most_related_para']
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para]}
                        )
                    else:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            question_tokens = sample['segmented_question']
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            correct_preds = sum(common_with_question.values())
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0]
                        sample['passages'].append({'passage_tokens': fake_passage_tokens})
                data_set.append(sample)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id, yes_no = False, entity = False):
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
                      'question_token_ids_a': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_token_ids_a': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': [],
                       'yes_no' : [],
                      'entity' : [],
                      'answer_token_ids' : [],
                      'answer_length' : [],
                      ### Add label element
                      # 'p_label': [],
                      # 'q_label':[]
                      'label':[]
                      }
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_token_ids_a'].append(sample['question_token_ids_a'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)

                    passage_token_ids_a = sample['passages'][pidx]['passage_token_ids_a']
                    batch_data['passage_token_ids_a'].append(passage_token_ids_a)

                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                    batch_data['answer_token_ids'].append(sample['answer_token_ids'])
                    batch_data['answer_length'].append(min(len(sample['answer_token_ids']), self.max_a_len))
                    ### Add label element
                    # batch_data['q_label'].append(batch_data['question_length']*[sample['label']])
                    # batch_data['p_label'].append(batch_data['passage_length'] * [sample['label']])
                    batch_data['label'].append(sample['label'])
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
                    batch_data['answer_token_ids'].append([])
                    batch_data['answer_length'].append(0)
                    batch_data['passage_token_ids_a'].append([])
                    batch_data['question_token_ids_a'].append([])
                    ### Add label element
                    # batch_data['p_label'].append([])
                    # batch_data['q_label'].append([])
                    batch_data['label'].append([0.0,0.0,0.0,0.0,0.0])



        batch_data, padded_p_len, padded_q_len, padded_a_len = self._dynamic_padding(batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        pad_a_len = min(self.max_a_len, max(batch_data['answer_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        ####label padding
        # batch_data['p_label'] = [(label + (pad_p_len - len(label) * [[0, 0, 0, 0, 0]]))[:pad_p_len] for label in
        #                          batch_data['p_label']]
        # batch_data['q_label'] = [(label + (pad_q_len - len(label) * [[0, 0, 0, 0, 0]]))[:pad_q_len] for label in
        #                          batch_data['q_label']]


        batch_data['passage_token_ids_a'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids_a']]
        batch_data['answer_token_ids'] = [(ids + [pad_id] * (pad_a_len - len(ids)))[: pad_a_len]
                                           for ids in batch_data['answer_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]

        batch_data['question_token_ids_a'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids_a']]

        return batch_data, pad_p_len, pad_q_len, pad_a_len

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
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token

    def convert_to_ids(self, vocab, vocab1, vocab2, entity = False):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['question_tokens'])
                sample['question_token_ids_a'] = vocab2.convert_to_ids(sample['question_tokens'])
                sample['answer_token_ids'] = vocab1.convert_to_ids(sample['answer_tokens'])
                for passage in sample['passages']:
                    passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])
                    passage['passage_token_ids_a'] = vocab2.convert_to_ids(passage['passage_tokens'])

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