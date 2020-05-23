#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:20:49 2019
Model learning using Mouse / Human data
run_num = sys.argv[1]
cell_lineage=sys.argv[2]

@author: chendi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#import sys

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')

import cnn_model
import plot_utils

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 100
learning_rate = 0.002

# Model version setting
species = 'Mouse'
run_num = 'epoch' + str(num_epochs) + '_best_weights'#_sigmoid_scalar' #'_PoissionNLL'#+ 'PoissionNLLoss_' #'_best_weights_' + 
model_name = 'model' + '_' + species + '_' + run_num #+ 'PoissionNLLoss_'

# Create output directory
folder = '../'
output_directory = folder + 'results/models/pytorch_version/' + model_name + '/'
directory = os.path.dirname(output_directory)
if not os.path.exists(directory):
    print('Creating directory %s' % output_directory)
    os.makedirs(directory)
else:
     print('Directory %s exists' % output_directory)
     
#-------------------------------------------#
#                 Load Data                 #
#-------------------------------------------#
# Load one-hot encoding sequence feature data
x = np.load('../data/' + species + '_Data/one_hot_seqs_ACGT.npy')
x = x.astype(np.float32)
peak_names = np.load('../data/' + species + '_Data/peak_names.npy')

#cell_lineage = sys.argv[2] #B
cell_lineage = ''
if cell_lineage == '':
    y = np.load('../data/' + species + '_Data/cell_type_array.npy')
    # Read cell_type_names = class names
    with open('../data/' + species + '_Data/cell_type_names.txt','r') as class_names_file:
        if species == 'Mouse':
            class_names = []
            for line in class_names_file:
                line = line.strip() # remove /n at the end
                task_num, class_name = line.split()
                class_names.append(class_name)
            class_names = list(filter(None, class_names))  # This removes empty entries 
        elif species == 'Human':
            class_names = class_names_file.read().split('\t')
            class_names[-1] = class_names[-1].strip() # remove the last newline
            class_names = list(filter(None, class_names))  # This removes empty entries
    cell_names = class_names
else:
    y = np.load('../data/' + species + '_Data/cell_type_array.{}.npy'.format(cell_lineage))
    cell_names=list(pd.read_csv('../data//immGenATAC/peaksAllCelltypesATAC/ATAC_Data_Intensity_FilteredPeaksLogQuantile.{}lineage.csv'.format(cell_lineage),index_col=0).columns)
y = y.astype(np.float32)

# split the data into training and test sets
train_data, eval_data, train_labels, eval_labels, train_names, eval_names = train_test_split(x, y, peak_names, test_size=0.1, random_state=42)

# split eval into half valid half test
half_eval_len = int(len(eval_names)/2)
valid_data = eval_data[:half_eval_len]
valid_labels = eval_labels[:half_eval_len]
valid_names = eval_names[:half_eval_len]

test_data = eval_data[half_eval_len:]
test_labels = eval_labels[half_eval_len:]
test_names = eval_names[half_eval_len:]

# Data loader
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(valid_data), torch.from_numpy(valid_labels))
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#-------------------------------------------#
#              Model training               #
#-------------------------------------------#
## Train a model from scratch
# create model 
num_classes = len(class_names)# 81
model = cnn_model.ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = cnn_model.pearson_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

# train model
model, training_loss, valid_loss, best_epoch = cnn_model.train_model(train_loader, valid_loader, model, device, criterion, optimizer, num_epochs)

# Add the best epoch to the model name
#model_name = model_name + '_at_' + str(best_epoch)
# Save training and valid loss
np.save(output_directory + 'training_loss.npy', training_loss)
np.save(output_directory + 'valid_loss.npy', valid_loss)

# save the model checkpoint
torch.save(model.state_dict(), folder + 'results/models/pytorch_version/' + model_name + '.ckpt')

# save the whole model
torch.save(model, folder + 'results/models/pytorch_version/' + model_name + '.pth')

# Predict on test set
predictions = cnn_model.test_model(test_loader, model, device, num_classes)

#-------------------------------------------#
#               Create Plots                #
#-------------------------------------------#

# plot the correlations histogram
# returns correlation measurement for every prediction-label pair
print("Creating plots...")

plot_utils.plot_training_valid_loss(training_loss, valid_loss, 'train', 'valid', output_directory)

correlations = plot_utils.plot_cors(test_labels, predictions, output_directory)

plot_utils.plot_corr_variance(test_labels, correlations, output_directory)

quantile_indx = plot_utils.plot_cors_piechart(correlations, test_labels, output_directory)

plot_utils.plot_random_predictions(test_labels, predictions, correlations, quantile_indx, test_names, output_directory, num_classes, cell_names)


#-------------------------------------------#
#                 Save Files                #
#-------------------------------------------#

#save predictions
np.save(output_directory + 'predictions.npy', predictions)

#save correlations
np.save(output_directory + 'correlations.npy', correlations)

#save test data set
np.save(output_directory + 'test_data.npy', test_data)
np.save(output_directory + 'test_labels.npy', test_labels)
np.save(output_directory + 'test_OCR_names.npy', test_names)