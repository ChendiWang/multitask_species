#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 4th 
Model learning using Mouse / Human data

@author: chendi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.utils.data

import Basset

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create model
weight_path = '../results/models/example/basset_pretrained_kipoi_weights.pth'
model = Basset.Basset(weight_path=weight_path)
model = model.to(device)

# Load sequence data 
species = 'Human'
x = np.load('../data/' + species + '_Data/one_hot_seqs_ACGT.npy')
x = x.astype(np.float32)
# Pad zeros to ends to satisfy 600 bp
window_size = 600  
x_padding = np.zeros((x.shape[0], x.shape[1], window_size - x.shape[2]))
x_padding = x_padding.astype(np.float32)
x = np.concatenate((x, x_padding), axis=-1)
y = np.load('../data/' + species + '_Data/cell_type_array.npy')
y = y.astype(np.float32)
peak_names = np.load('../data/' + species + '_Data/peak_names.npy')

batch_size = 4
_, test_data, _, test_labels, _, test_names = train_test_split(x, y, peak_names, test_size=0.01, random_state=42)
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Make predictions using Basset model
num_classes = 164
predictions =  torch.zeros(0, num_classes)
model.eval() # Set model to evaluate mode
with torch.no_grad():        
    for seqs, labels in test_loader:
        seqs = seqs.to(device)
        pred = model(seqs)
        pred = torch.sigmoid(pred)
        predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)

predictions = predictions.numpy()