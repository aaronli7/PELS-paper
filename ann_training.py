#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:52:51 2020

@author: aaronli
"""

import loader
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import pickle
import numpy as np

# Hyper Parameters
EPOCH = 100
BATCH_SIZE = 32
TIME_STEP = 10  # length of LSTM time sequence, or window size
INPUT_SIZE = 18 # num of feature for deep learning
LR = 0.01   # learning rate
KFOLD = 1
isGPU = torch.cuda.is_available()

# detection deep learning dataset path
data_path_DL = './w10_dataset/fault_detection/'

all_data_DL = loader.pmuDetectionDL(data_path_DL)


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(180, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output 

data_output = {'ann':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[], 'test_loss':[], 'train_loss':[]}}


#%% for 
for num_of_training in range(KFOLD):
    ann = ANN()
    if isGPU:
        ann.cuda()
        ann = nn.DataParallel(ann, device_ids=[0, 1, 2])
    ann_optimizer = torch.optim.Adam(ann.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    print(ann)
    
    training_data, test_data = torch.utils.data.random_split(all_data_DL, [int(all_data_DL.len * 0.85), all_data_DL.len - int(all_data_DL.len * 0.85)])
    training_Loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_Loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
    ann_test_loss_draw = []
    ann_loss_draw = []
    
    for epoch in range(EPOCH):
        print('epoch {}'.format(epoch + 1))
         # training-----------------------------------------
        ann_train_loss = 0.
        ann_train_acc = 0.
        
        ann.train()

        for step, (batch_x, batch_y) in enumerate(training_Loader):
            batch_x = batch_x.view(-1, 10, 18)
            batch_x = batch_x.float()
            
            if isGPU:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                
            output_ann = ann(batch_x)

            loss_ann = loss_func(output_ann, batch_y)
            
            ann_train_loss += loss_ann.item()
            
            if isGPU:
                ann_pred = torch.max(output_ann, 1)[1].cuda()
            else:
                ann_pred = torch.max(output_ann, 1)[1]
        
            ann_train_correct = (ann_pred == batch_y).sum()
            ann_train_acc += ann_train_correct.item()
            ann_optimizer.zero_grad()
            loss_ann.backward()
            ann_optimizer.step()
            
        print('ANN:\n Train Loss: {:.6f}, Accuracy: {:.6f}\n'.format(ann_train_loss / 
              (len(training_data)), ann_train_acc / (len(training_data))))

        ann_loss_draw.append(ann_train_loss/(len(training_data)))
        
        # evaluation----------------------------------------------------------
        ann.eval()
        ann_eval_loss = 0.
        ann_eval_acc = 0.
        ann_final_prediction = np.array([])
        ann_final_test = np.array([])
        ann_f1_score = []
        ann_recall = []
        ann_precision = []
        
        for step, (batch_x, batch_y) in enumerate(test_Loader):
            batch_x = batch_x.view(-1, 10, 18)
            batch_x = batch_x.float()
            
            if isGPU:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
    
            output_ann = ann(batch_x)
            loss_ann = loss_func(output_ann, batch_y)
            ann_eval_loss += loss_ann.item()
            ann_pred = torch.max(output_ann, 1)[1]
            ann_train_correct = (ann_pred == batch_y).sum()
            
            if isGPU:
                ann_pred = torch.max(output_ann, 1)[1].cuda()
            else:
                ann_pred = torch.max(output_ann, 1)[1]

            ann_eval_acc += ann_train_correct.item()
            
            # loss = loss_func(output, batch_y)
            # eval_loss += loss.item()
            # pred = torch.max(output, 1)[1]
            # num_correct = (pred == batch_y).sum()
            # eval_acc += num_correct.item()
            
            # F1 metrics
            
            ann_final_prediction = np.concatenate((ann_final_prediction, ann_pred.cpu().numpy()), axis=0)
            ann_final_test = np.concatenate((ann_final_test, batch_y), axis=0)
            
        ann_f1_score.append(sklearn.metrics.f1_score(ann_final_test, ann_final_prediction, average='weighted').item())
        ann_recall.append(sklearn.metrics.recall_score(ann_final_test, ann_final_prediction, average='weighted').item())
        ann_precision.append(sklearn.metrics.precision_score(ann_final_test, ann_final_prediction, average='weighted').item())
        print('ANN:\n Test Loss: {:.6f}, Acc: {:.6f}'.format(ann_eval_loss / 
            (len(test_data)), ann_eval_acc / (len(test_data))))
        
        ann_test_loss_draw.append(ann_eval_loss/(len(test_data)))
        print('ANN:\n F1: {}, recall: {}, precision: {}'.format(ann_f1_score[-1], ann_recall[-1], ann_precision[-1]))
        
    ann_test_y = label_binarize(ann_final_test, classes=[0, 1])
    ann_pred_y = label_binarize(ann_final_prediction, classes=[0, 1])
    ann_fpr, ann_tpr, _ = roc_curve(ann_test_y.ravel(), ann_pred_y.ravel())
    ann_roc_auc = auc(ann_fpr, ann_tpr)
    
    data_output['ann']['F1'].append(ann_f1_score[-1])
    data_output['ann']['precision'].append(ann_precision[-1])
    data_output['ann']['recall'].append(ann_recall[-1])
    data_output['ann']['accuracy'].append(ann_eval_acc / (len(test_data)))
    data_output['ann']['auc'].append(ann_roc_auc.item())
    data_output['ann']['fpr'].append(list(ann_fpr))
    data_output['ann']['tpr'].append(list(ann_tpr))
    data_output['ann']['test_loss'].append(ann_test_loss_draw)
    data_output['ann']['train_loss'].append(ann_loss_draw)    