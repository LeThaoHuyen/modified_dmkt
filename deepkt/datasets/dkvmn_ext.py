from textwrap import fill
import numpy as np
from torch.utils.data import Dataset
from deepkt.datasets.transforms import SlidingWindow, Padding
from sklearn.preprocessing import label_binarize
import torch

class DKVMN_ExtDataset(Dataset):
    """
    prepare the data for data loader, including truncating long sequence and padding
    """

    def __init__(self, config, q_records, a_records, l_records, sa_records, num_items, max_seq_len, min_seq_len=2,
                 q_subseq_len=8, l_subseq_len=10, stride=None, train=True, metric="auc", q_rules_records=None, l_rules_records=None,
                 pt_q_records=None, pt_a_records=None, mode=None):
        """
        :param min_seq_len: used to filter out seq. less than min_seq_len
        :param max_seq_len: used to truncate seq. greater than max_seq_len
        :param max_subseq_len: used to truncate the lecture seq.
        """
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.q_subseq_len = q_subseq_len
        self.l_subseq_len = l_subseq_len
        self.num_items = num_items
        if stride is not None and train:
            self.stride = stride
        else:
            # because we pad 0 at the first element
            self.stride = max_seq_len - 1
        self.metric = metric
        self.padding_value = -1

        self.q_data, self.a_data, self.sa_data, self.l_data, self.q_rules_data, self.l_rules_data = self._transform(
            q_records, a_records, sa_records, l_records, q_rules_records, l_rules_records)
        print("train samples.: {}".format(self.l_data.shape))
        self.length = len(self.q_data)
        self.mode = mode
        self.pt_q_data = pt_q_records
        self.pt_a_data = pt_a_records

        self.num_concepts = config.num_concepts
        
    def __len__(self):
        """
        :return: the number of training samples rather than training users
        """
        return self.length

    def __getitem__(self, idx):
        """
        for DKT, we apply one-hot encoding and return the idx'th data sample here, we dont
        need to return target, since the input of DKT also contain the target information
        reference: https://github.com/jennyzhang0215/DKVMN/blob/master/code/python3/model.py

        for SAKT, we dont apply one-hot encoding, instead we return the past interactions,
        current exercises, and current answers
        # reference: https://github.com/TianHongZXY/pytorch-SAKT/blob/master/dataset.py

        idx: sample index
        """
       
        questions = self.q_data[idx] # questions = [Q1, Q2, ...]
        answers = self.a_data[idx]   # answers = [A1, A2, ...]
        lectures = self.l_data[idx]
        student_answers = self.sa_data[idx]
        question_rules = self.q_rules_data[idx]
        lecture_rules = self.l_rules_data[idx]

        
        # lecture_mask = []

        # padding_mask = [0]*self.num_concepts
        # padding_mask.append(1)

        # l_mask = [1]*self.num_concepts
        # l_mask.append(0)

        # for lecture_list in lectures:
        #     sub_lecture_mask = []
        #     for l in lecture_list:
        #         l = list(l)
        #         if self.isPaddingVector(l, self.padding_value):
        #             sub_lecture_mask.append(padding_mask) # to do: change to self.num_concepts --> need to pass config
        #         else:
        #             sub_lecture_mask.append(l_mask) # to do
        #     lecture_mask.append(sub_lecture_mask)

        # assert len(questions) == len(answers) == len(lectures)
        
        interactions = []
        target_answers = []
        target_mask = []
        # question_mask = []
        for question_list, answer_list, student_answers_list in zip(questions, answers, student_answers):
            interaction_list = []
            # sub_mask = []
            for i, q in enumerate(question_list):
                # q = [x1, x2, x3, ..., x8]
                # qa = [x1,x2, .., x8, a, 0, 1, 0] # last 3 elements for student's answer
                q = list(q)   
                if self.isPaddingVector(q, self.padding_value):
                    target_mask.append(False)
                    # sub_mask.append(padding_mask)
                else:
                    target_mask.append(True)
                    # sub_mask.append(l_mask)

                q.append(answer_list[i])
                interaction_list.append(q)

            # question_mask.append(sub_mask)
            interaction_list = np.array(interaction_list, dtype=float)
            interactions.append(interaction_list)
            target_answers.extend(answer_list)

        if self.mode == None:
            return np.array(interactions), lectures, questions, np.array(target_answers), np.array(target_mask), question_rules, lecture_rules
        else:
            pt_questions = self.pt_q_data[idx]
            target_answers = self.pt_a_data[idx]
            target_mask = [True]*10
            return np.array(interactions), lectures, questions, np.array(target_answers), np.array(target_mask), question_rules, lecture_rules, np.array(pt_questions)


    def isPaddingVector(self, q, padding_value):
        count = 0
        for x in q:
            if x == padding_value:
                count += 1
        return count == len(q)

    def _transform(self, q_records, a_records, sa_records, l_records, q_rules_records, l_rules_records):
        q_data = []
        a_data = []
        sa_data = []
        l_data = []
        q_rules_data = []
        l_rules_data = []

        setup_dim = len(q_records[0][0][0])
        padding_setup = [self.padding_value]*setup_dim
        padding_mask = [0]*12
        padding_mask.append(1)

        for q_list, a_list, sa_list, l_list, q_rules_list, l_rules_list in zip(q_records, a_records, sa_records, l_records, q_rules_records, l_rules_records):
            assert len(q_list) == len(a_list) == len(sa_list)

            q_padding = Padding(self.max_seq_len, side='right', fillvalue=[padding_setup])
            a_padding = Padding(self.max_seq_len, side='right', fillvalue=[self.padding_value])  
            rule_padding = Padding(self.max_seq_len, side='right', fillvalue=[padding_mask])
            
            q_list = q_padding(q_list[-self.max_seq_len:])
            a_list = a_padding(a_list[-self.max_seq_len:])
            sa_list = a_padding(sa_list[-self.max_seq_len:])
            l_list = q_padding(l_list[-self.max_seq_len:])
            q_rules_list = rule_padding(q_rules_list[-self.max_seq_len:])
            l_rules_list = rule_padding(l_rules_list[-self.max_seq_len:])

            assert len(q_list) == len(a_list) == len(sa_list) == len(l_list) == self.max_seq_len

            sub_q_padding = Padding(self.q_subseq_len, side='right', fillvalue=padding_setup)
            sub_a_padding = Padding(self.q_subseq_len, side='right', fillvalue=self.padding_value)
            sub_l_padding = Padding(self.l_subseq_len, side='right', fillvalue=padding_setup)
            sub_qrule_padding = Padding(self.q_subseq_len, side='right', fillvalue=padding_mask)
            sub_lrule_padding = Padding(self.l_subseq_len, side='right', fillvalue=padding_mask)

            q_list = [sub_q_padding(sub_list[-self.q_subseq_len:]) for sub_list in q_list]
            a_list = [sub_a_padding(sub_list[-self.q_subseq_len:]) for sub_list in a_list]
            sa_list = [sub_a_padding(sub_list[-self.q_subseq_len:]) for sub_list in sa_list]
            l_list = [sub_l_padding(sub_list[-self.l_subseq_len:]) for sub_list in l_list]

            q_rules_list = [sub_qrule_padding(sub_list[-self.q_subseq_len:]) for sub_list in q_rules_list]
            l_rules_list = [sub_lrule_padding(sub_list[-self.l_subseq_len:]) for sub_list in l_rules_list]

            assert len(q_list[0]) == len(a_list[0]) == len(sa_list[0]) == self.q_subseq_len
            assert len(l_list[0]) == self.l_subseq_len

            q_data.append(q_list)
            a_data.append(a_list)
            l_data.append(l_list)
            sa_data.append(sa_list)

            q_rules_data.append(q_rules_list)
            l_rules_data.append(l_rules_list)

        return np.array(q_data), np.array(a_data), np.array(sa_data), np.array(l_data), np.array(q_rules_data), np.array(l_rules_data)
