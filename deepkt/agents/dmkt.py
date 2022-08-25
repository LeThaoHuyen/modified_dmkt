import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
import sys
import matplotlib.pyplot as plt
from deepkt.agents.base import BaseAgent
from deepkt.graphs.models.dmkt import DMKT

# should not remove import statements below, it;s being used seemingly.
from deepkt.dataloaders import *

cudnn.benchmark = True
from deepkt.utils.misc import print_cuda_statistics
import warnings
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

warnings.filterwarnings("ignore")


# Notes on training DKVMN:
#   1. the batch size should not be large
#   2. the learning rate should not be large
#   3. should clip the gradient to avoid gradient vanishing or exploding
#   4. setup a good parameter initializer for key and value memory
#   5. use learning rate scheduler or reduce learning rate adaptively


class DMKTAgent(BaseAgent):
    def __init__(self, config):
        """initialize the agent with provided config dict which inherent from the base agent
        class"""
        super().__init__(config)

        # initialize the data_loader, which include preprocessing the data
        data_loader = globals()[config.data_loader]  # remember to import the dataloader
        self.data_loader = data_loader(config=config)
        # self.data_loader have attributes: train_data, train_loader, test_data, test_loader
        # note that self.data_loader.train_data is same as self.data_loader.train_loader.dataset
        self.mode = config.mode
        self.metric = config.metric
        self.num_folds = 10
        self.batch_size = config.batch_size

        config.num_items = self.data_loader.num_items
        config.num_nongradable_items = self.data_loader.num_nongradable_items
        self.model = DMKT(config)
        self.criterion = nn.BCELoss(reduction='sum')
        # self.criterion = nn.CrossEntropyLoss(reduction='sum')

        if config.optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.config.learning_rate,
                                       momentum=self.config.momentum)
        else:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.config.learning_rate,
                                        eps=self.config.epsilon)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )
        # self.scheduler = optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     max_lr=config.max_learning_rate,
        #     steps_per_epoch=len(self.data_loader.train_loader),
        #     epochs=config.max_epoch
        # )

        # self.scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer,
        #     milestones=[2000, 4000, 6000, 8000, 10000],
        #     gamma=0.667
        # )

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        # this loading should be after checking cuda
        self.load_checkpoint(self.config.checkpoint_file)

    

    def train(self):
        """
        Main training loop
        :return:
        """
        if self.mode == 'train_cv':
            k_fold = KFold(n_splits = self.num_folds, shuffle=True)
            dataset = self.data_loader.train_data
            results = []

            for fold, (train_ids, test_ids) in enumerate (k_fold.split(dataset)):
                print("\n")
                print(f"Train on fold {fold}:")
                train_subsampler = SubsetRandomSampler(train_ids)
                test_subsampler = SubsetRandomSampler(test_ids)

                train_loader = DataLoader(dataset, batch_size = self.batch_size, sampler=train_subsampler)
                test_loader = DataLoader(dataset, batch_size= self.batch_size,sampler=test_subsampler)

                self.model = DMKT(self.config)
                self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.config.learning_rate,
                                        eps=self.config.epsilon)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    patience=0,
                    min_lr=1e-5,
                    factor=0.5,
                    verbose=True
                )
                self.current_epoch = 0
                for epoch in range(1, self.config.max_epoch +1):
                    self.train_one_epoch(train_loader)
                    self.current_epoch += 1
                    self.validate(test_loader)

                print(f"Best ROC-AUC for fold {fold}:", self.best_val_perf)
                results.append(self.best_val_perf)
                self.best_val_perf = 0

            average = sum(results) / len(results)
            print("-------------------------")
            print("{}-fold cross validation average result: {:.6f}".format(self.num_folds, average))

        else:
            for epoch in range(1, self.config.max_epoch + 1):
                self.train_one_epoch(self.data_loader.train_loader)
                self.validate(self.data_loader.test_loader)
                self.current_epoch += 1
                if self.early_stopping():
                    break
            
            print("Best ROC-AUC:", self.best_val_perf)
            epochs = range(1, self.config.max_epoch + 1)

            plt.plot(epochs, self.train_loss_list, 'g', label='Training loss')
            plt.plot(epochs, self.test_loss_list, 'b', label='validation loss')
            plt.title('Training and Validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    def mask_select(self,data, mask):
        res = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                if mask[i][j] == True:
                    res.append(data[i][j])
        res = torch.stack(res, 0)
        return res

    def train_one_epoch(self, train_loader):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        self.logger.info("\n")
        self.logger.info("Train Epoch: {}".format(self.current_epoch))
        self.logger.info("Learning rate: {}".format(self.optimizer.param_groups[0]['lr']))
        self.train_loss = 0
        train_elements = 0
        
        for batch_idx, data in enumerate(tqdm(train_loader)):
            interactions, lec_interactions_list, questions, target_answers, target_mask, lecture_mask, question_mask = data
            # target_answers = [1, 0, 1, 2, ...]
            # output = [[0.02, 0.98, 0], [0.45, 0.55, 0], ...]

            interactions = interactions.to(self.device)
            lec_interactions_list = lec_interactions_list.to(self.device)
            questions = questions.to(self.device)
            target_answers = target_answers.to(self.device)
            target_mask = target_mask.to(self.device)
            question_mask = question_mask.to(self.device)

            self.optimizer.zero_grad()  # clear previous gradient
            # need to double check the target mask
            output = self.model(questions, interactions, lec_interactions_list, lecture_mask, question_mask)

            # label = self.mask_select(target_answers, target_mask)
            # output = self.mask_select(output, target_mask)
            # print("target answer {}".format(target_answers))
            label = torch.masked_select(target_answers, target_mask)
            # print("output: {}".format(output))
            output = torch.masked_select(output, target_mask)

            loss = self.criterion(output.float(), label.float())
            # loss = self.criterion(output, label)
            # should use reduction="mean" not "sum", otherwise, performance drops significantly
            self.train_loss += loss.item()
            train_elements += target_mask.int().sum()
            loss.backward()  # compute the gradient

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()  # update the weight
            # self.scheduler.step()  # for CycleLR Scheduler or MultiStepLR
            self.current_iteration += 1
        # used for ReduceLROnPlateau
        self.train_loss = self.train_loss / train_elements
        self.train_loss_list.append(self.train_loss)
        self.scheduler.step(self.train_loss)
        self.logger.info("Train Loss: {:.6f}".format(self.train_loss))

    def validate(self, test_loader):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        if self.mode == "train":
            self.logger.info("Validation Result at Epoch: {}".format(self.current_epoch))
        elif self.mode == 'test-post-test':
            self.logger.info("Validation Result on post tests at Epoch: {}".format(self.current_epoch))
        else:
            self.logger.info("Test Result at Epoch: {}".format(self.current_epoch))
        test_loss = 0
        test_elements = 0
        pred_labels = []
        true_labels = []

        # print(self.mode)
        with torch.no_grad():
            for data in test_loader: # 1 batch
                if self.mode == 'test-post-test':
                    interactions, lec_interactions_list, questions, target_answers, target_mask, lecture_mask, question_mask, pt_questions = data
                    pt_questions = pt_questions.to(self.device)
                else:
                    interactions, lec_interactions_list, questions, target_answers, target_mask, lecture_mask, question_mask = data

                interactions = interactions.to(self.device)
                lec_interactions_list = lec_interactions_list.to(self.device)
                questions = questions.to(self.device)
                target_answers = target_answers.to(self.device)
                target_mask = target_mask.to(self.device)
                question_mask = question_mask.to(self.device)

                if self.mode == 'test-post-test':
                    output = self.model(questions, interactions, lec_interactions_list, lecture_mask, question_mask, pt_questions)
                    
                else:
                    output = self.model(questions, interactions, lec_interactions_list, lecture_mask, question_mask)
                # output = torch.masked_select(output[:, 1:], target_mask[:, 1:])
                # label = torch.masked_select(target_answers[:, 1:], target_mask[:, 1:])
                # test_elements += target_mask[:, 1:].int().sum()
                output = torch.masked_select(output, target_mask)
                label = torch.masked_select(target_answers, target_mask)

                # label = self.mask_select(target_answers, target_mask)
                # output = self.mask_select(output, target_mask)

                # if self.mode == 'test-post-test':
                #     # output = [[0.2, 0.1, 0.7], [0.4, 0.6, 0], ...]
                #     # --> output = [0, 1,...]
                #     _, max_indice = torch.max(output, 1)
                #     fill_values = torch.zeros(output.size(0)).long()
                #     output = torch.where(max_indice == 1, max_indice, fill_values) 
                #     criterition = nn.BCELoss(reduction='sum')
                #     test_loss += criterition(output.float(), label.float()).item()
                # else:
                #     test_loss += self.criterion(output, label).item()
                    
                # test_loss += self.criterion(output, label).item()
                test_loss += self.criterion(output.float(), label.float())
                test_elements += target_mask.int().sum()
                pred_labels.extend(output.tolist())
                true_labels.extend(label.tolist())

                # print(pred_labels)
                # print(output)
                # print(list(zip(true_labels, pred_labels)))
                # print(true_labels)
                # print(pred_labels)
        test_loss = test_loss/test_elements
        self.logger.info("Test Loss: {:.6f}".format(test_loss))
        self.test_loss_list.append(test_loss)
        self.track_best(true_labels, pred_labels)

