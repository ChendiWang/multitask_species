#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:35:49 2020

@author: chendi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf

import utils
import plot_utils
import utils_influence
from custom_layer import GELU

# Hyper parameters
batch_size = 100

folder = '../'
species = 'Mouse'
if species == 'Mouse':
    combining_weight_txt = '001'
if species == 'Human':
    combining_weight_txt = '00052'

subfolder = 'frozen/'
run_num = 'Basset_adam_lr0001_dropout_03_conv_relu_max_BN_convfc_batch_100_loss_combine' + combining_weight_txt + '_bysample_epoch10_best_separatelayer_rc_shiftleft_shiftright'

model_name = 'model' + '_' + species + '_' + run_num

# Load the keras model
json_file = open(folder + 'results/models/keras_version/' + subfolder + model_name + '/whole_model_best.json', 'r')
model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(model_json, {'GELU': GELU}) #, {'GELU': GELU}
model.load_weights(folder + 'results/models/keras_version/' + subfolder + model_name + '/whole_model_best_weights.h5')
print('Loaded model from disk')
model.summary()

test_txt = ''#'_test' #''
output_directory = folder + 'results/motifs/keras_version/' + subfolder + model_name + '/well_predicted' + test_txt + '/'
directory = os.path.dirname(output_directory)  
if not os.path.exists(directory):
    print('Creating directory %s' % output_directory)
    os.makedirs(directory)
else:
     print('Directory %s exists' % output_directory)

# Load all data
x = np.load('../data/' + species + '_Data/one_hot_seqs_ACGT.npy')
x = x.astype(np.float32)
x = np.swapaxes(x,1,2)
y = np.load('../data/' + species + '_Data/cell_type_array.npy')
y = y.astype(np.float32)
peak_names = np.load('../data/' + species + '_Data/peak_names.npy')

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

num_classes = len(class_names)

# run predictions with full model on all data
predicted_labels = model.predict(np.stack(x))
correlations = utils.metric_pearson(y, predicted_labels)

# Select the well predicted OCRs for interpretation
if species == 'Mouse':
    prediction_threshold = 0.75
if species == 'Human':
    prediction_threshold = 0.85
index_well_predicted = np.stack(correlations) > prediction_threshold

x_subset = x[index_well_predicted, :, :]
y_subset = y[index_well_predicted, :]
peak_names_subset = peak_names[index_well_predicted]

# Use smaller cases for the whole set
if test_txt:
    test_num = 200
    x_subset = x_subset[0:test_num, :, :]
    y_subset = y_subset[0:test_num, :]
    peak_names_subset = peak_names_subset[0:test_num]

# Reproducibility due to different batch effect 
# # so rerun model prediction to match with the filter influence experiment
predicted_labels_subset = model.predict(np.stack(x_subset))

print("Length of well predicted cases")
print(len(x_subset))
print(len(y_subset))
np.save(output_directory + "selected_labels.npy", y_subset)
np.save(output_directory + "selected_features.npy", x_subset)
np.save(output_directory + "selected_peak_names.npy", peak_names_subset)
np.save(output_directory + "selected_predictions.npy", predicted_labels_subset)

# Use automatic layer name extraction, after activation
# layer_names=[]
# for i, layer in enumerate(model.layers):
#     print(i)
#     print(layer.name)
#     layer_names.append(layer.name) 
# layer_name = layer_names[1]
layer_name = 'activation_5'#'gelu_5'# for after activation in single species models
intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
activations = intermediate_layer_model.predict(x_subset)
np.save(output_directory + 'selected_first_layer_activations.npy', activations)

threshold = 0.5
flag_coding_order = 'ACGT'
flag_weighted = False
num_filter = 300
filter_to_ind_dict, pwm, act_ind, nseqs, activated_OCRs, n_activated_OCRs, OCR_matrix = plot_utils.get_memes(activations, x_subset, y_subset, output_directory, num_classes, flag_coding_order, threshold, flag_weighted, num_filter)

np.savetxt(output_directory + 'nseqs_per_filters.txt', nseqs)
np.save(output_directory + 'activated_OCRs.npy', activated_OCRs)
np.save(output_directory + 'n_activated_OCRs.npy',  n_activated_OCRs)
np.save(output_directory + 'OCR_matrix.npy', OCR_matrix)

# Filter influence computation
filter_number_layer_1 = 300
window_size = x_subset.shape[1]
target_layer_name = 'max_pooling1d_3'
# predictions_by_filter = np.zeros((len(x_subset), filter_number_layer_1, num_classes))
predictions_by_filter = np.expand_dims(predicted_labels_subset, 1)
predictions_by_filter = np.repeat(predictions_by_filter, filter_number_layer_1, axis=1)
for filter_number in range(filter_number_layer_1):
    print('The current filter is: ' + str(filter_number))
    filter_mask = np.ones((len(x_subset), window_size, filter_number_layer_1))
    # Comment out to test if change == 0, weights been copied from the original model
    filter_mask[:, :, filter_number] = 0
    filter_model = utils_influence.create_filter_model(model, window_size, target_layer_name)
    # filter_model.summary()
    predictions_by_filter[:, filter_number, :] = filter_model.predict([x_subset, filter_mask])

correlations = plot_utils.plot_cors(y_subset, predicted_labels_subset, output_directory)
filt_corr, corr_change, corr_change_mean, corr_change_mean_act = plot_utils.plot_filt_corr_change(predictions_by_filter, y_subset, correlations, output_directory)
infl, infl_signed_mean, infl_signed_mean_act, infl_absolute_mean, infl_absolute_mean_act = plot_utils.plot_filt_infl(predicted_labels_subset, predictions_by_filter, output_directory, class_names)
np.save(output_directory + "selected_correlations.npy", correlations)
np.save(output_directory + "selected_correlations_by_removing_filter.npy", filt_corr)
np.save(output_directory + 'filter_influence_per_ocr.npy', corr_change)
np.save(output_directory + 'filter_influence_mean.npy', corr_change_mean)
np.save(output_directory + 'filter_influence_mean_activated.npy', corr_change_mean_act)
np.save(output_directory + 'influence_celltype_original_per_ocr.npy', infl)
np.save(output_directory + 'filter_cellwise_influence_signed_mean.npy', infl_signed_mean)
np.save(output_directory + 'filter_cellwise_influence_signed_mean_activated.npy', infl_signed_mean_act)
np.save(output_directory + 'filter_cellwise_influence_absolute_mean.npy', infl_absolute_mean)
np.save(output_directory + 'filter_cellwise_influence_absolute_mean_activated.npy', infl_absolute_mean_act)