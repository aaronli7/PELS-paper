#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:33:20 2020

@author: aaronli
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib as mpl
mpl.use('Agg')

# Detection
class pmuDetectionDL(Dataset):
    
    def __init__(self, path):
        
        # Directory of normal and abnormal data
        normal_path = path + 'Normal/'
        abnormal_path = path + 'Abnormal/'
        
        normal_files = [f for f in os.listdir(normal_path) if f.endswith('.csv')]
        abnormal_files = [f for f in os.listdir(abnormal_path) if f.endswith('.csv')]
        
        np_normal = []
        np_abnormal = []
        
        for f in normal_files:
            temp_file = np.loadtxt(normal_path + f, delimiter=',', dtype=np.float32)
            np_normal.append(temp_file)
        
        for f in abnormal_files:
            temp_file = np.loadtxt(abnormal_path + f, delimiter=',', dtype=np.float32)
            np_abnormal.append(temp_file)
            
        # x: input data, y: target
        self.y_data = np.concatenate([np.zeros(len(np_normal),dtype=np.long), \
                                      np.zeros(len(np_abnormal),dtype=np.long)+1])
        self.x_data = np.concatenate([np_normal, np_abnormal])
        
        self.len = self.x_data.shape[0]
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
    
# Diagnosis 
class pmuformDiagnosisDL(Dataset):
    
    def __init__(self, path):
        
        # Directory of normal and abnormal data
        normal_path = path + 'Normal/'
        fault_1_path = path + 'Fault_1/'
        fault_2_path = path + 'Fault_2/'
        fault_3_path = path + 'Fault_3/'
        fault_4_path = path + 'Fault_4/'
        fault_5_path = path + 'Fault_5/'
        fault_6_path = path + 'Fault_6/'
        
        
        normal_files = [f for f in os.listdir(normal_path) if f.endswith('.csv')]
        fault_1_files = [f for f in os.listdir(fault_1_path) if f.endswith('.csv')]
        fault_2_files = [f for f in os.listdir(fault_2_path) if f.endswith('.csv')]
        fault_3_files = [f for f in os.listdir(fault_3_path) if f.endswith('.csv')]
        fault_4_files = [f for f in os.listdir(fault_4_path) if f.endswith('.csv')]
        fault_5_files = [f for f in os.listdir(fault_5_path) if f.endswith('.csv')]
        fault_6_files = [f for f in os.listdir(fault_6_path) if f.endswith('.csv')]
        
        np_normal = []
        np_fault_1 = []
        np_fault_2 = []
        np_fault_3 = []
        np_fault_4 = []
        np_fault_5 = []
        np_fault_6 = []
        
        
        for f in normal_files:
            temp_file = np.loadtxt(normal_path + f, delimiter=',', dtype=np.float32)
            np_normal.append(temp_file)
        
        for f in fault_1_files:
            temp_file = np.loadtxt(fault_1_path + f, delimiter=',', dtype=np.float32)
            np_fault_1.append(temp_file)
            
        for f in fault_2_files:
            temp_file = np.loadtxt(fault_2_path + f, delimiter=',', dtype=np.float32)
            np_fault_2.append(temp_file)
            
        for f in fault_3_files:
            temp_file = np.loadtxt(fault_3_path + f, delimiter=',', dtype=np.float32)
            np_fault_3.append(temp_file)
            
        for f in fault_4_files:
            temp_file = np.loadtxt(fault_4_path + f, delimiter=',', dtype=np.float32)
            np_fault_4.append(temp_file)
            
        for f in fault_5_files:
            temp_file = np.loadtxt(fault_5_path + f, delimiter=',', dtype=np.float32)
            np_fault_5.append(temp_file)
            
        for f in fault_6_files:
            temp_file = np.loadtxt(fault_6_path + f, delimiter=',', dtype=np.float32)
            np_fault_6.append(temp_file)
            
        # x: input data, y: target
        self.y_data = np.concatenate([np.zeros(len(np_normal),dtype=np.long), \
                                      np.zeros(len(np_fault_1),dtype=np.long)+1, \
                                      np.zeros(len(np_fault_2),dtype=np.long)+2, \
                                      np.zeros(len(np_fault_3),dtype=np.long)+3, \
                                      np.zeros(len(np_fault_4),dtype=np.long)+4, \
                                      np.zeros(len(np_fault_5),dtype=np.long)+5, \
                                      np.zeros(len(np_fault_6),dtype=np.long)+6, ])
        self.x_data = np.concatenate([np_normal, np_fault_1, np_fault_2, np_fault_3, np_fault_4, 
                                      np_fault_5, np_fault_6])
        self.len = self.x_data.shape[0]
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len