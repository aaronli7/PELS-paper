This repository contains the code for the 2020 PELS-paper.
This code implemented the attack detection and diagnosis based on micro-PMU data using Pytorch deep learning framework and sklearn.

Usage:
- Run 'to_csv.m' in Matlab under the raw data directory to change the .mat files into .csv files for training.
- Run 'data_partition.m' in Matlab to create different folders for training.
- Run the other files ending in '_training' or '_training_diagnosis' for detection and diagnosis. The results will show different metrics including F1, accuracy, precision, recall. 
