#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:54:48 2020

@author: chendi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

try:
    import cPickle as pickle
except:
    import pickle
    
import utils

species = 'Human'
ref_genome = 'hg19'
tag_selection = '2015_DeepSea_ISM'
flag_first_run = True
flag_overlapped_celltype = False

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
        class_names = list(filter(None, class_names))

folder = '../'
subfolder = 'reproducibility/'#'frozen/'#'multitask/'
flag_task = 'Human/'#'multitask/'#
data_augmentation = 'rc_shiftleft_shiftright/'
run_num = 'Basset_adam_lr0001_dropout_03_conv_relu_max_BN_convfc_batch_100_loss_combine00052_bysample_epoch10_best_separatelayer_rc_shiftleft_shiftright'
model_name = 'model' + '_' + species + '_' + run_num
# model_name = 'model_multitask_MaH_ratiobased_Basset_adam_lr0001_dropout_03_conv_relu_max_BN_convfc_batch100_loss_combine001_00052_bysample_epoch10_best_separatelayer_hard_branched_rc_shiftleft_shiftright'

# Select the SNP dataset
dataset = 'GRASP'
# dataset = 'GWAS'

# SNP set
if dataset == 'GRASP':
    df = pd.read_csv(folder + 'data/2015_08_DeepSEA/41592_2015_BFnmeth3547_MOESM649_ESM.csv', sep=',', skiprows=2, header=0, index_col=0) 
    label_txt = 'eQTL'
if dataset == 'GWAS':
    df = pd.read_csv(folder + 'data/2015_08_DeepSEA/41592_2015_BFnmeth3547_MOESM650_ESM.csv', sep=',', skiprows=2, header=0, index_col=0) 
    label_txt = 'GWAS Catalog'

for i in range(20):
    output_directory = folder + 'results/motifs/keras_version/'  + subfolder + flag_task + data_augmentation + model_name + \
        '_run' + str(i) + '/' + tag_selection + '/' + dataset + '/ISM_stats' + '/'

    directory = os.path.dirname(output_directory)  
    if not os.path.exists(directory):
        print('Creating directory %s' % output_directory)
        os.makedirs(directory)
    else:
        print('Directory %s exists' % output_directory)

    # Compare if the SAD score changes more in the positive set than the negative set
    if flag_first_run:
        # Load the model for prediction
        from custom_layer import GELU
        json_file = open(folder + 'results/models/keras_version/' + subfolder + flag_task + data_augmentation + model_name + \
             '_run' + str(i) + '/whole_model_best.json', 'r')
        keras_model_json = json_file.read()
        json_file.close()
        keras_model = tf.keras.models.model_from_json(keras_model_json, {'GELU': GELU})
        keras_model.load_weights(folder + 'results/models/keras_version/' + subfolder + flag_task + data_augmentation + model_name + \
            '_run' + str(i) + '/whole_model_best_weights.h5')
        print('Loaded model from disk')
        print(i)
        # Set window length
        num_base = 1
        interval = 251
        chr_dict = pickle.load(open(os.path.join('/media/chendi/DATA2/projects/XDNN/' + ref_genome + '_results/', 'chr_all_dict.pickle'), 'rb'))

        sad_pos = [] # SNP Accessibility Difference (SAD)
        sad_pos_norm = []
        sad_neg = [] 
        sad_neg_norm = [] 
        sad = []
        sad_norm = []
        if flag_task == 'multitask/': #subfolder
            if flag_overlapped_celltype:
                num_classes_1 = 28
                num_classes_2 = 8
            else:
                num_classes_1 = 81
                num_classes_2 = 18
            labels_mask = np.hstack([np.zeros(num_classes_1), np.ones(num_classes_2)])
            labels_mask = labels_mask[np.newaxis,:]
            species_tags = 2*np.ones((1)) # 2 for human # 1 for mouse

        for i, rows in df.iterrows():
            # print(i)
            chromo, midpoint, ref, alt, label = rows.chr, rows.pos, rows.ref, rows.alt, rows.label
            
            long_seq_record = chr_dict[chromo]
            long_seq = str(long_seq_record.seq)
                
            start = midpoint - np.floor(interval/2).astype(np.int64) 
            if interval % 2 == 0:
                stop = midpoint + np.floor(interval/2).astype(np.int64) - 1
            else:
                stop = midpoint + np.floor(interval/2).astype(np.int64)
            short_seq_first_half = long_seq[start-1:midpoint-1]
            short_seq_second_half = long_seq[midpoint:stop]
            
            short_seq = short_seq_first_half + ref + short_seq_second_half
            one_hot_seq = utils.one_hot_encode_along_channel_axis(short_seq, onehot_axis=1)    
            if flag_task == 'multitask/': #subfolder
                [dummy, predictions_before_mut] = keras_model.predict([one_hot_seq[np.newaxis,:], labels_mask[:, :num_classes_1], labels_mask[:, -num_classes_2:], species_tags])
                predictions_before_mut = np.squeeze(predictions_before_mut, axis=0)
            else:
                predictions_before_mut = np.squeeze(keras_model.predict(one_hot_seq[np.newaxis,:]), axis=0)       
            
            # Mutate the ref allele to alt and prediction the activity again
            short_seq = short_seq_first_half + alt + short_seq_second_half
            one_hot_seq = utils.one_hot_encode_along_channel_axis(short_seq, onehot_axis=1)
            if flag_task == 'multitask/': #subfolder
                [dummy, predictions_after_mut] = keras_model.predict([one_hot_seq[np.newaxis,:], labels_mask[:, :num_classes_1], labels_mask[:, -num_classes_2:], species_tags])
                predictions_after_mut = np.squeeze(predictions_after_mut, axis=0)
            else:
                predictions_after_mut = np.squeeze(keras_model.predict(one_hot_seq[np.newaxis,:]), axis=0)    
            
            # Using summation of the activity magnitute across cell types (mean/median/max) as the indicator of sad        
            sad.append(predictions_after_mut.flatten() - predictions_before_mut.flatten()) # a cell type long vector
            sad_norm.append((predictions_after_mut.flatten() - predictions_before_mut.flatten())/np.max(predictions_before_mut.flatten())) # a cell type long vector  
            
            if label == label_txt:
                sad_pos.append(sad[-1])
                sad_pos_norm.append(sad_norm[-1])
            else:
                sad_neg.append(sad[-1])
                sad_neg_norm.append(sad_norm[-1])
            
        sad_array = np.asarray(sad)
        del sad
        sad_norm_array = np.asarray(sad_norm)
        del sad_norm
        sad_pos_array = np.asarray(sad_pos)
        del sad_pos
        sad_pos_norm_array = np.asarray(sad_pos_norm)
        del sad_pos_norm
        sad_neg_array = np.asarray(sad_neg)
        del sad_neg
        sad_neg_norm_array = np.asarray(sad_neg_norm)
        del sad_neg_norm

        # Append sad scores by cell types to original df
        sad_df = pd.DataFrame(data=sad_array,
                            columns=class_names)
        sad_norm_df = pd.DataFrame(data=sad_norm_array,
                            columns=class_names)
        df = pd.merge(df, sad_df, how = 'inner', left_index=True, right_index=True)
        df = pd.merge(df, sad_norm_df, how = 'inner', left_index=True, right_index=True, suffixes=('_ori', '_norm'))

        # Save results to file
        df.to_csv(output_directory + dataset + '_sad.csv', header=True, index=False, sep = ',')

        np.save(output_directory + 'sad', sad_array)
        np.save(output_directory + 'sad_norm', sad_norm_array)
        np.save(output_directory + 'sad_pos', sad_pos_array)
        np.save(output_directory + 'sad_pos_norm', sad_pos_norm_array)
        np.save(output_directory + 'sad_neg', sad_neg_array)
        np.save(output_directory + 'sad_neg_norm', sad_neg_norm_array)

    else:
        # Read results from file
        df = pd.read_csv(output_directory + dataset + '_sad.csv', sep=',', header=0)
        sad_array = np.load(output_directory + 'sad.npy')
        sad_norm_array = np.load(output_directory + 'sad_norm.npy')
        sad_pos_array = np.load(output_directory + 'sad_pos.npy')
        sad_pos_norm_array = np.load(output_directory + 'sad_pos_norm.npy')
        sad_neg_array = np.load(output_directory + 'sad_neg.npy')
        sad_neg_norm_array = np.load(output_directory + 'sad_neg_norm.npy') 

    NUM_COLORS = len(class_names)
    cm = sns.color_palette('husl', n_colors=NUM_COLORS) 
    my_colors = 'brgkymc'

    # Measure the change of sad for post vs neg SNP
    operation_txt = 'max'

    # Take the absolute value for the changes before using max
    absolute_txt = '_absolute_value' # ''
    if absolute_txt:
        sad_array = np.absolute(sad_array)
        sad_norm_array = np.absolute(sad_norm_array)
        sad_pos_array = np.absolute(sad_pos_array)
        sad_pos_norm_array = np.absolute(sad_pos_norm_array)
        sad_neg_array = np.absolute(sad_neg_array)
        sad_neg_norm_array = np.absolute(sad_neg_norm_array)

    # Basset paper: "SAD profile means were significantly greater for the set of high-PICS SNPs (Mann-Whitney U test, P-value <1.3 × 10 −7 )"
    norm_txt = ''#'_norm'
    if norm_txt: 
        [stats, pvalue] = scipy.stats.mannwhitneyu(np.max(sad_pos_norm_array, -1), np.max(sad_neg_norm_array, -1)) # np.max/mean/median
        mean_stats_pos = np.mean(np.max(sad_pos_norm_array, -1)) # np.mean
        mean_stats_neg = np.mean(np.max(sad_neg_norm_array, -1))
    else:
        [stats, pvalue] = scipy.stats.mannwhitneyu(np.max(sad_pos_array, -1), np.max(sad_neg_array, -1)) # np.max/mean/median
        mean_stats_pos = np.mean(np.max(sad_pos_array, -1)) # np.mean
        mean_stats_neg = np.mean(np.max(sad_neg_array, -1))

    plt.clf()
    plt.bar([0, 1], [mean_stats_pos, mean_stats_neg], color=my_colors)
    plt.xticks(np.arange(2), ('Positive', 'Negative'))
    plt.title('Original sad' + norm_txt + 'score change average across cell types')
    plt.tight_layout()
    plt.savefig(output_directory + 'sad' + norm_txt + '_score_' + operation_txt + absolute_txt + '_average_change_barplot.svg') 
    plt.close()

    diff_ratio = mean_stats_pos / mean_stats_neg 
        
    # If one of them is negative, need to normalized to above 0, then compare the absolute value
    if mean_stats_neg < 0 or mean_stats_pos < 0:
        diff_ratio = (mean_stats_pos - 2*np.minimum(mean_stats_pos, mean_stats_neg)) / (mean_stats_neg - 2*np.minimum(mean_stats_pos, mean_stats_neg)) 
        plt.clf()
        plt.bar([0, 1], [mean_stats_neg - 2*np.minimum(mean_stats_pos, mean_stats_neg), mean_stats_pos - 2*np.minimum(mean_stats_pos, mean_stats_neg)], color=my_colors)
        plt.xticks(np.arange(2), ('Positive', 'Negative'))
        plt.title('Adjusted sad' + norm_txt + 'score change average across cell types')
        plt.tight_layout()
        plt.savefig(output_directory + 'sad' + norm_txt + '_score_' + operation_txt + absolute_txt + '_average_change_adjusted_barplot.svg') 
        plt.close()

    np.savez(output_directory + 'sad' + norm_txt + '_score_' + operation_txt + absolute_txt + '_stats.npz', 
            pvalue=pvalue, mean_stats_pos=mean_stats_pos, mean_stats_neg=mean_stats_neg, diff_ratio=diff_ratio)

    max_values = np.max(sad_array, -1)
    plt.clf()
    plt.hist(max_values, bins=10)  
    plt.title('distribution of Max sad across cell types')
    plt.savefig(output_directory + 'dist_sad_max' + absolute_txt + '_celltype.svg') 
    plt.close() 

    mean_values = np.mean(sad_array, -1)
    plt.clf()
    plt.hist(mean_values, bins=10)  
    plt.title('distribution of Mean sad across cell types')
    plt.savefig(output_directory + 'dist_sad_mean' + absolute_txt + '_celltype.svg') 
    plt.close()