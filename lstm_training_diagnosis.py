#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 23:27:21 2020

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
from sklearn.metrics import roc_curve, auc, confusion_matrix
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
data_path_DL = './w10_dataset_new/fault_diagnosis/'

all_data_DL = loader.pmuformDiagnosisDL(data_path_DL)

#%%----------------------create the LSTM Net ------------------------------------
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
                input_size=INPUT_SIZE,
                hidden_size=32,
                num_layers=2,
                batch_first=True,
                )
        #fully connected
        self.out = nn.Linear(32, 7)
    
    def forward(self, x):
        lstm_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(lstm_out[:, -1, :])
        return out
    


data_output = {'lstm':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[], 'test_loss':[], 'train_loss':[]}}


for num_of_training in range(KFOLD):
    lstm = LSTM()
    
    if isGPU:
        lstm = nn.DataParallel(lstm, device_ids=[0, 1, 2])
        lstm.cuda()

    lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    
     # print the structure of the network
    print(lstm)
    
     # data partition: 15% testing, 85% training
    training_data, test_data = torch.utils.data.random_split(all_data_DL, [int(all_data_DL.len * 0.85), all_data_DL.len - int(all_data_DL.len * 0.85)])
    training_Loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_Loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
    
    # training and testing
    lstm_test_loss_draw = []
    
    lstm_loss_draw = []
    
    for epoch in range(EPOCH):
        print('epoch {}'.format(epoch + 1))
        
        # training-----------------------------------------
        lstm_train_loss = 0.
        
        lstm_train_acc = 0.
        
        lstm.train()
        
        for step, (batch_x, batch_y) in enumerate(training_Loader):
            batch_x = batch_x.view(-1, TIME_STEP, INPUT_SIZE)
            
            if isGPU:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            output_lstm = lstm(batch_x)
            
            loss_lstm = loss_func(output_lstm, batch_y)
            
            lstm_train_loss += loss_lstm.item()
                     
            if isGPU:
                lstm_pred = torch.max(output_lstm, 1)[1].cuda()
            else:
                lstm_pred = torch.max(output_lstm, 1)[1]
            
            lstm_train_correct = (lstm_pred == batch_y).sum()
            
            lstm_train_acc += lstm_train_correct.item()
            
            lstm_optimizer.zero_grad()
            
            loss_lstm.backward()
            
            lstm_optimizer.step()
            
        print('LSTM:\n Train Loss: {:.6f}, Accuracy: {:.6f}\n'.format(lstm_train_loss / 
              (len(training_data)), lstm_train_acc / (len(training_data))))
        
        lstm_loss_draw.append(lstm_train_loss/(len(training_data)))
        
        # evaluation--------------------------------------------------
        lstm.eval()
        
        lstm_eval_loss = 0.

        lstm_eval_acc = 0.
        
        lstm_final_prediction = np.array([])
        lstm_final_test = np.array([])
        lstm_f1_score = []
        lstm_recall = []
        lstm_precision = []
        lstm_accuracy = []
                
        
        for step, (batch_x, batch_y) in enumerate(test_Loader):
            batch_x = batch_x.view(-1, TIME_STEP, INPUT_SIZE)
            batch_x_cnn = torch.unsqueeze(batch_x, dim=1).type(torch.float)
            
            if isGPU:
                batch_x = batch_x.cuda()
                batch_x_cnn = batch_x_cnn.cuda()
                batch_y = batch_y.cuda()
    
            output_lstm = lstm(batch_x)

            
            loss_lstm = loss_func(output_lstm, batch_y)
            
            lstm_eval_loss += loss_lstm.item()
            
            lstm_pred = torch.max(output_lstm, 1)[1]
            
            lstm_train_correct = (lstm_pred == batch_y).sum()
            
            if isGPU:
                lstm_pred = torch.max(output_lstm, 1)[1].cuda()
            else:
                lstm_pred = torch.max(output_lstm, 1)[1]
            
            lstm_eval_acc += lstm_train_correct.item()
            
            # loss = loss_func(output, batch_y)
            # eval_loss += loss.item()
            # pred = torch.max(output, 1)[1]
            # num_correct = (pred == batch_y).sum()
            # eval_acc += num_correct.item()
            
            # F1 metrics
            lstm_final_prediction = np.concatenate((lstm_final_prediction, lstm_pred.cpu().numpy()), axis=0)
            lstm_final_test = np.concatenate((lstm_final_test, batch_y), axis=0)
            

        
        lstm_f1_score.append(sklearn.metrics.f1_score(lstm_final_test, lstm_final_prediction, average='weighted').item())

        lstm_recall.append(sklearn.metrics.recall_score(lstm_final_test, lstm_final_prediction, average='weighted').item())

        lstm_precision.append(sklearn.metrics.precision_score(lstm_final_test, lstm_final_prediction, average='weighted').item())
        
        lstm_precision.append(sklearn.metrics.balanced_accuracy_score(lstm_final_test, lstm_final_prediction))
        
        print('LSTM:\n Test Loss: {:.6f}, Acc: {:.6f}'.format(lstm_eval_loss / 
              (len(test_data)), lstm_eval_acc / (len(test_data))))
        
        lstm_test_loss_draw.append(lstm_eval_loss/(len(test_data)))
        
        print('LSTM:\n F1: {}, recall: {}, precision: {}'.format(lstm_f1_score[-1], lstm_recall[-1], lstm_precision[-1]))
        
        
    # confusing matrix
    current_cm = confusion_matrix(list(lstm_final_test), list(lstm_final_prediction))
    print(current_cm)
        
    # ROC curve and AUC
    lstm_test_y = label_binarize(lstm_final_test, classes=[0, 1, 2, 3, 4, 5, 6])
    lstm_pred_y = label_binarize(lstm_final_prediction, classes=[0, 1, 2, 3, 4, 5, 6])
    lstm_fpr, lstm_tpr, _ = roc_curve(lstm_test_y.ravel(), lstm_pred_y.ravel())
    lstm_roc_auc = auc(lstm_fpr, lstm_tpr)
    

    
    data_output['lstm']['F1'].append(lstm_f1_score[-1])
    data_output['lstm']['precision'].append(lstm_precision[-1])
    data_output['lstm']['recall'].append(lstm_recall[-1])
    data_output['lstm']['accuracy'].append(lstm_eval_acc / (len(test_data)))
    data_output['lstm']['auc'].append(lstm_roc_auc.item())
    data_output['lstm']['fpr'].append(list(lstm_fpr))
    data_output['lstm']['tpr'].append(list(lstm_tpr))
    data_output['lstm']['test_loss'].append(lstm_test_loss_draw)
    data_output['lstm']['train_loss'].append(lstm_loss_draw)  
    