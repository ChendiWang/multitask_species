#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:57:03 2019

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
import matplotlib.pyplot as plt

import tensorflow as tf

import utils

species = 'Mouse'
source_folder = '../data/mouseMutation/knockouts/'
gene_names_df = pd.read_csv(source_folder + 'mart_export.txt', index_col=None, header=0, encoding='latin-1')
gene_names_df.columns = [c.replace(' ', '_') for c in gene_names_df.columns]

#gene_list = ['Agbl1', 'Alpk1', 'Bex1', 'Mx1'] # For T4 activities
#gene_list = ['Epha7', 'Gm16432', 'Sgce', '4930422G04Rik'] # For GN activities
#gene_list = ['Ifi203', 'Lifr', 'Saa4', 'Tm4sf20', 'A430107O13Rik'] # For GN activities
gene_list = ['Ifi203', 'Saa4']

# TODO: Add a loop to iterate through the genes
gene_info = gene_names_df.loc[gene_names_df.Gene_name == gene_list[0]]  # [1 index]
gene_chromosome = 'chr' + str(gene_info['Chromosome/scaffold_name'].values.astype(int)[0])
gene_TSS = gene_info['Gene_start_(bp)'].values - 1
window_range_lower = gene_TSS - 10000
window_range_upper = gene_TSS + 10000

# Load coordinate file from ImmGen
# ImmGenATAC1219.peak.bed
immgen_coordinates_df = pd.read_csv('../data/' + species + '_Data/ImmGenATAC1219.peak.filteredSM0.05.bed', index_col=None, header=None, delimiter='\t', encoding='latin-1')
# Use peak name file to first restrict to the valid ones
peak_names = np.load('../data/' + species + '_Data/peak_names.npy').astype(str)
peak_names = pd.DataFrame(peak_names)
selected_ocrs = peak_names.merge(immgen_coordinates_df, how='inner', left_on=0, right_on=0, left_index=True)
#selected_ocrs = immgen_coordinates_df

selected_ocrs_chromo = selected_ocrs.loc[selected_ocrs[1] == gene_chromosome]
selected_ocrs_bound = selected_ocrs_chromo.loc[(selected_ocrs_chromo[2] > window_range_lower[0]).values * (selected_ocrs_chromo[3] < window_range_upper[0]).values] # use lower and upper bound

# Extract sequences from ref vs. variant strains
one_hot_ref = np.load('../data/' + species + '_Data/one_hot_seqs_ACGT.npy')
one_hot_ref = one_hot_ref .astype(np.float32)
one_hot_ref = np.swapaxes(one_hot_ref,1,2)

selected_one_hot_ref = one_hot_ref[selected_ocrs_bound.index.values, :, :]
selected_sequences = utils.seq_from_onehot(selected_one_hot_ref)

# GN cases
variant_sequences = []
variant_sequences.append('N')
selected_peak_names = selected_ocrs_bound[0].values.astype(str)

selected_one_hot_variant = np.zeros(selected_one_hot_ref.shape)
for i in range(len(variant_sequences)):
    selected_one_hot_variant[i, :, :] = utils.one_hot_encode_along_channel_axis(variant_sequences[i], onehot_axis=1)
    
# Apply model for ISM
folder = '../'
tag_selection = 'knockout_paper_ISM'
subfolder = 'weighted_loss/'
run_num = 'Basset_adam_lr0001_dropout_03_conv_relu_max_BN_batch_100_loss_combine001_epoch10_best_separatelayer'
model_name = 'model' + '_' + species + '_' + run_num

json_file = open(folder + 'results/models/keras_version/' + subfolder + model_name + '/whole_model_best.json', 'r')
keras_model_json = json_file.read()
json_file.close()
keras_model = tf.keras.models.model_from_json(keras_model_json)
keras_model.load_weights(folder + 'results/models/keras_version/' + subfolder + model_name + '/whole_model_best_weights.h5')
print('Loaded model from disk')
keras_model.summary()

predictions_before_mut = keras_model.predict(selected_one_hot_ref)
predictions_after_mut = keras_model.predict(selected_one_hot_variant)

output_directory = folder + 'results/motifs/keras_version/' + model_name + '/' + tag_selection + '/ISM/'
directory = os.path.dirname(output_directory)  
if not os.path.exists(directory):
    print('Creating directory %s' % output_directory)
    os.makedirs(directory)
else:
    print('Directory %s exists' % output_directory)
    
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
        class_names = list(filter(None, class_names))  

for i in range(len(variant_sequences)):   
    np.save(output_directory + 'ism_' + selected_peak_names[i] + '_cell_type_profile_before_mut.npy', predictions_before_mut[i, :])
    np.save(output_directory + 'ism_' + selected_peak_names[i] + '_cell_type_profile_after_mut.npy', predictions_after_mut[i, :])
    plt.clf()
    plt.subplots(figsize=(16, 5))
    line_before = plt.bar(np.arange(len(class_names)), predictions_before_mut[i, :].reshape(len(class_names),), color='blue', alpha=0.3)
    line_after = plt.bar(np.arange(len(class_names)), predictions_after_mut[i, :].reshape(len(class_names),), color='red', alpha=0.3)
    plt.title('ISM before and after')
    plt.xlabel('cell type')
    plt.ylabel('cell type profile')        
    plt.xticks(np.arange(len(class_names)), (class_names), fontsize=8, rotation=90)#
    plt.legend(['before', 'after'])
    plt.tight_layout()
    plt.savefig(output_directory + 'ism_' + selected_peak_names[i] + '_cell_type_profile_change.svg') 
    plt.close()