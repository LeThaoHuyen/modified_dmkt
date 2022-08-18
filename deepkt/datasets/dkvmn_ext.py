from textwrap import fill
import numpy as np
from torch.utils.data import Dataset
from deepkt.datasets.transforms import SlidingWindow, Padding
from sklearn.preprocessing import label_binarize

class DKVMN_ExtDataset(Dataset):
    """
    prepare the data for data loader, including truncating long sequence and padding
    """

    def __init__(self, q_records, a_records, l_records, sa_records, num_items, max_seq_len, min_seq_len=2,
                 q_subseq_len=8, l_subseq_len=10, stride=None, train=True, metric="auc", pt_q_records=None, pt_sa_records=None):
        """
        :param min_seq_len: used to filter out seq. less than min_seq_len
        :param max_seq_len: used to truncate seq. greater than max_seq_len
        :param max_subseq_len: used to truncate the lecture seq.
        """
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.q_subseq_len = q_subseq_len
        self.num_items = num_items
        if stride is not None and train:
            self.stride = stride
        else:
            # because we pad 0 at the first element
            self.stride = max_seq_len - 1
        self.metric = metric
        self.padding_value = -1

        self.q_data, self.a_data, self.l_data, self.sa_data = self._transform(
            q_records, a_records, l_records, sa_records, q_subseq_len, l_subseq_len)
        print("train samples.: {}".format(self.l_data.shape))
        self.length = len(self.q_data)
        self.pt_q_data = pt_q_records
        self.pt_sa_data = pt_sa_records
        

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

        assert len(questions) == len(answers) == len(lectures)
        interactions = []

        
        target_answers = []
        target_mask = []
        for question_list, answer_list, student_answers_list in zip(questions, answers, student_answers):
            interaction_list = []
            for i, q in enumerate(question_list):
                # q = [x1, x2, x3, ..., x8]
                q = list(q)
                if self.isPaddingVector(q, self.padding_value):
                    target_mask.append(False)
                else:
                    target_mask.append(True)

                q.append(answer_list[i])
                # q.append(student_answers_list[i])
                interaction_list.append(q)
            
            interaction_list = np.array(interaction_list, dtype=float)
            interactions.append(interaction_list)
            target_answers.extend(answer_list)
            # target_answers.extend(student_answers_list)

        pt_questions = self.pt_q_data[idx]
        target_answers.extend(self.pt_sa_data[idx])
        target_mask.extend([True]*10)
        # target_answers = (self.pt_sa_data[idx])
        # target_mask = ([True]*10)

        return np.array(interactions), lectures, questions, np.array(target_answers), np.array(target_mask), np.array(pt_questions)


    def isPaddingVector(self, q, padding_value):
        count = 0
        for x in q:
            if x == padding_value:
                count += 1
        return count == len(q)

    # def _transform(self, q_records, a_records, l_records=None, max_subseq_len=None):
    #     """
    #     transform the data into feasible input of model,
    #     truncate the seq. if it is too long and
    #     pad the seq. with 0s if it is too short
    #     """
    #     if l_records is not None and max_subseq_len is not None:
    #         assert len(q_records) == len(a_records) == len(l_records)
    #         l_data = []
    #         lec_sliding = SlidingWindow(self.max_seq_len, self.stride,
    #                                     fillvalue=[0] * max_subseq_len)
    #         padding = Padding(max_subseq_len, side='left', fillvalue=0)
    #     else:
    #         assert len(q_records) == len(a_records)

    #     q_data = []
    #     a_data = []
    #     # if seq length is less than max_seq_len, the sliding will pad it with fillvalue
    #     # the reason of inserting the first attempt with 0 and setting stride=self.max_seq_len-1
    #     # is to make sure every test point will be evaluated if a model cannot test the first
    #     # attempt
    #     sliding = SlidingWindow(self.max_seq_len, self.stride, fillvalue=0)
    #     for index in range(len(q_records)):
    #         q_list = q_records[index]
    #         a_list = a_records[index]
    #         q_list.insert(0, 0)
    #         a_list.insert(0, 0)
    #         assert len(q_list) == len(a_list)
    #         sample = {"q": q_list, "a": a_list}
    #         output = sliding(sample)
    #         q_data.extend(output["q"])
    #         a_data.extend(output["a"])
    #         if l_records is not None:
    #             l_list = l_records[index]
    #             l_list = [padding({"l": l[-max_subseq_len:]})["l"] for l in l_list]
    #             l_list.insert(0, [0] * max_subseq_len)
    #             assert len(q_list) == len(a_list) #== len(l_list)
    #             sample = {"l": l_list}
    #             lec_output = lec_sliding(sample)
    #             l_data.extend(lec_output["l"])
    #             assert len(q_data) == len(a_data) #== len(l_data)

    #     if l_records is not None:
    #         return np.array(q_data), np.array(a_data), np.array(l_data)
    #     else:
    #         return np.array(q_data), np.array(a_data)


    # def _transform(self, q_records, a_records, l_records=None, max_subseq_len=None):
    #     q_data = []
    #     a_data = []
    #     l_data = []

    #     for q_list, a_list, l_list in zip(q_records, a_records, l_records):
    #         assert len(q_list) == len(a_list)

    #         if len(q_list) >= self.max_seq_len:
    #             q_list = q_list[: self.max_seq_len]
    #             a_list = a_list[: self.max_seq_len]
    #         else:
    #             q_list.extend([0]*(self.max_seq_len - len(q_list)))
    #             a_list.extend([0]*(self.max_seq_len - len(a_list)))

    #         assert len(q_list) == len(a_list)

    #         if len(l_list) >= self.max_seq_len:
    #             l_list = l_list[: self.max_seq_len]
    #         else:
    #             l_list.extend([[0] for _ in range (self.max_seq_len - len(l_list))])

    #         assert len(q_list) == len(a_list) == len(l_list) 

    #         q_data.append(q_list)
    #         a_data.append(a_list)
    #         l_data.append(l_list)


    #     return np.array(q_data), np.array(a_data), np.array(l_data)


    def _transform(self, q_records, a_records, l_records, sa_records, q_subseq_len, l_subseq_len):
        q_data = []
        a_data = []
        l_data = []
        sa_data = []

        setup_dim = len(q_records[0][0][0])

        for q_list, a_list, l_list, sa_list in zip(q_records, a_records, l_records, sa_records):
            assert len(q_list) == len(a_list) == len(sa_list)
            
            if len(q_list) >= self.max_seq_len:
                q_list = q_list[-self.max_seq_len:]
                a_list = a_list[-self.max_seq_len:]
                sa_list = sa_list[-self.max_seq_len:]
            else:
                q_list.extend([[[self.padding_value]*setup_dim] for _ in range (self.max_seq_len - len(q_list))])
                a_list.extend([[self.padding_value] for _ in range (self.max_seq_len - len(a_list))])
                sa_list.extend([[self.padding_value] for _ in range(self.max_seq_len - len(sa_list))]) # coi lai thu nen de la 0 ko
            
            assert len(q_list) == len(a_list) == len(sa_list)

            if len(l_list) >= self.max_seq_len:
                l_list = l_list[-self.max_seq_len:]
            else:
                l_list.extend([[[0]*setup_dim] for _ in range (self.max_seq_len - len(l_list))])

            #padding = Padding(max_subseq_len, side='left', fillvalue=0)
            q_padding = Padding(q_subseq_len, side='right', fillvalue=[self.padding_value]*setup_dim)
            a_padding = Padding(q_subseq_len, side='right', fillvalue=self.padding_value)
            l_padding = Padding(l_subseq_len, side='right', fillvalue=[self.padding_value]*setup_dim)
            sa_padding = Padding(q_subseq_len, side='right', fillvalue=self.padding_value)

            q_list = [q_padding({"q": q[-q_subseq_len:]})["q"] for q in q_list]
            a_list = [a_padding({"a": a[-q_subseq_len:]})["a"] for a in a_list]
            l_list = [l_padding({"l": l[-l_subseq_len:]})["l"] for l in l_list]
            sa_list = [sa_padding({"sa": sa[-q_subseq_len:]})["sa"] for sa in sa_list]

            assert len(q_list) == len(a_list) == len(l_list) == len(sa_list)

            q_data.append(q_list)
            a_data.append(a_list)
            l_data.append(l_list)
            sa_data.append(sa_list)


        return np.array(q_data), np.array(a_data), np.array(l_data), np.array(sa_data)