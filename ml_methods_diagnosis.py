#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:50:14 2020

@author: aaronli
"""

import loader

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
import pickle
import numpy as np

# Hyper Parameters

KFOLD = 1

# detection deep learning dataset path
data_path_DL = './w10_dataset_new/fault_diagnosis/'

all_data_DL = loader.pmuformDiagnosisDL(data_path_DL)


data_output = {'knn':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[]}, 
               'dtree':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[]}, 
               'svm':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[]}}

for num_of_training in range(KFOLD):
    
    # machine learning methods
    ml_X = all_data_DL.x_data.numpy()
    ml_X = ml_X.reshape(ml_X.shape[0], -1)
    ml_Y = all_data_DL.y_data.numpy()
    
    ml_X_train, ml_X_test, ml_Y_train, ml_Y_test = train_test_split(ml_X, ml_Y, test_size=0.15)
    
    # train dtree
    dtree_model = DecisionTreeClassifier(max_depth = None).fit(ml_X_train, ml_Y_train)
    dtree_pred = dtree_model.predict(ml_X_test)
    
    # train SVM 
    svm_model = SVC(kernel = 'rbf', C = 1).fit(ml_X_train, ml_Y_train)
    svm_pred = svm_model.predict(ml_X_test)
    
    # train KNN 
    knn_model = KNeighborsClassifier(n_neighbors = 5).fit(ml_X_train, ml_Y_train)
    knn_pred = knn_model.predict(ml_X_test)
    
    # metrics for machine learning methods
    dtree_f1 = sklearn.metrics.f1_score(ml_Y_test, dtree_pred, labels = [1,2,3,4,5,6], average='micro')
    dtree_precision = sklearn.metrics.precision_score(ml_Y_test, dtree_pred, labels = [1,2,3,4,5,6], average='micro')
    dtree_recall = sklearn.metrics.recall_score(ml_Y_test, dtree_pred, labels = [1,2,3,4,5,6], average='micro')
    dtree_acc = sklearn.metrics.accuracy_score(ml_Y_test, dtree_pred)
    
    svm_f1 = sklearn.metrics.f1_score(ml_Y_test, svm_pred, labels = [1,2,3,4,5,6], average='micro')
    svm_precision = sklearn.metrics.precision_score(ml_Y_test, svm_pred, labels = [1,2,3,4,5,6], average='micro')
    svm_recall = sklearn.metrics.recall_score(ml_Y_test, svm_pred, labels = [1,2,3,4,5,6], average='micro')
    svm_acc = sklearn.metrics.accuracy_score(ml_Y_test, svm_pred)
    
    knn_f1 = sklearn.metrics.f1_score(ml_Y_test, knn_pred, labels = [1,2,3,4,5,6], average='macro')
    knn_precision = sklearn.metrics.precision_score(ml_Y_test, knn_pred, labels = [1,2,3,4,5,6], average='macro')
    knn_recall = sklearn.metrics.recall_score(ml_Y_test, knn_pred, labels = [1,2,3,4,5,6], average='macro')
    knn_acc = sklearn.metrics.accuracy_score(ml_Y_test, svm_pred)

    # for ROC curve
    svm_test_y = label_binarize(ml_Y_test, classes=[0, 1, 2, 3, 4, 5, 6])
    svm_pred_y = label_binarize(svm_pred, classes=[0, 1, 2, 3, 4, 5, 6])
    svm_fpr, svm_tpr, _ = roc_curve(svm_test_y.ravel(), svm_pred_y.ravel())
    svm_roc_auc = sklearn.metrics.auc(svm_fpr, svm_tpr)
    
    knn_test_y = label_binarize(ml_Y_test, classes=[0, 1, 2, 3, 4, 5, 6])
    knn_pred_y = label_binarize(knn_pred, classes=[0, 1, 2, 3, 4, 5, 6])
    knn_fpr, knn_tpr, _ = roc_curve(knn_test_y.ravel(), knn_pred_y.ravel())
    knn_roc_auc = sklearn.metrics.auc(knn_fpr, knn_tpr)
    
    dtree_test_y = label_binarize(ml_Y_test, classes=[0, 1, 2, 3, 4, 5, 6])
    dtree_pred_y = label_binarize(dtree_pred, classes=[0, 1, 2, 3, 4, 5, 6])
    dtree_fpr, dtree_tpr, _ = roc_curve(dtree_test_y.ravel(), dtree_pred_y.ravel())
    dtree_roc_auc = sklearn.metrics.auc(dtree_fpr, dtree_tpr)
    
    
    # confusing matrix
    current_cm_dtree = confusion_matrix((ml_Y_test), (dtree_pred))
    current_cm_knn = confusion_matrix((ml_Y_test), (knn_pred))
    current_cm_svm = confusion_matrix((ml_Y_test), (svm_pred))
    print(current_cm_dtree)
    print(current_cm_knn)
    print(current_cm_svm)

    
    
    # machine learning method output
    data_output['knn']['F1'].append(knn_f1)
    data_output['knn']['precision'].append(knn_precision)
    data_output['knn']['recall'].append(knn_recall)
    data_output['knn']['accuracy'].append(knn_acc)
    data_output['knn']['auc'].append(knn_roc_auc.item())
    data_output['knn']['fpr'].append(list(knn_fpr))
    data_output['knn']['tpr'].append(list(knn_tpr))
    
    data_output['dtree']['F1'].append(dtree_f1)
    data_output['dtree']['precision'].append(dtree_precision)
    data_output['dtree']['recall'].append(dtree_recall)
    data_output['dtree']['accuracy'].append(dtree_acc)
    data_output['dtree']['auc'].append(dtree_roc_auc.item())
    data_output['dtree']['fpr'].append(list(dtree_fpr))
    data_output['dtree']['tpr'].append(list(dtree_tpr))
    
    data_output['svm']['F1'].append(svm_f1)
    data_output['svm']['precision'].append(svm_precision)
    data_output['svm']['recall'].append(svm_recall)
    data_output['svm']['accuracy'].append(svm_acc)
    data_output['svm']['auc'].append(svm_roc_auc.item())
    data_output['svm']['fpr'].append(list(svm_fpr))
    data_output['svm']['tpr'].append(list(svm_tpr))
    
    
    #%% ---------------------dataoutput------------------------------------------------------------------

for i in range(KFOLD):
    print('--------Fold {}----------------'.format(i))
    print('Dtree: F1:{}, prec:{}, rec:{}, acc:{}, auc:{}'.format(data_output['dtree']['F1'][i],
          data_output['dtree']['precision'][i], data_output['dtree']['recall'][i],
          data_output['dtree']['accuracy'][i], data_output['dtree']['auc'][i]))
    print('SVM: F1:{}, prec:{}, rec:{}, acc:{}, auc:{}'.format(data_output['svm']['F1'][i],
          data_output['svm']['precision'][i], data_output['svm']['recall'][i],
          data_output['svm']['accuracy'][i], data_output['svm']['auc'][i]))
    print('KNN: F1:{}, prec:{}, rec:{}, acc:{}, auc:{}'.format(data_output['knn']['F1'][i],
          data_output['knn']['precision'][i], data_output['knn']['recall'][i],
          data_output['knn']['accuracy'][i], data_output['knn']['auc'][i]))