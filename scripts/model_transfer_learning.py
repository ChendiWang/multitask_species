#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 2 11:20:49 2020
Model learning using Mouse / Human data

@author: chendi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow

import losses
import utils
import plot_utils

# Hyperparameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001 * 0.1

platform_txt = 'keras_version/'
combining_weight = 0.0052 
combining_weight_txt = '00052'
subfolder = 'transfer_learning'
regularization_weight = 0 #0.01
weight_flag = 'minmax'
cell_lineage = ''
input_bp_txt = ''#'_600bp_with_overlap'

#-------------------------------------------#
# Step 1. Curate Data                
#-------------------------------------------#
# species = 'Mouse'
species = 'Human'
## Load one-hot encoding sequence feature data using order _ACGT_
input_features = np.load('../data/' + species + '_Data/one_hot_seqs_ACGT' + input_bp_txt + '.npy')
input_features = input_features.astype(np.float32)
peak_names = np.load('../data/' + species + '_Data/peak_names' + input_bp_txt + '.npy')
input_labels = np.load('../data/' + species + '_Data/cell_type_array.npy')
input_labels = input_labels.astype(np.float32)
class_names = utils.read_class_names('../data/' + species + '_Data/cell_type_names.txt', species)

num_classes = len(class_names)
print(input_features.shape)
print(input_labels.shape)

## Generate the dataset split
if species == 'Mouse':
    test_size_number = 15786
if species == 'Human':
    test_size_number = 25802
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels, train_names, test_names = train_test_split(
    input_features, input_labels, peak_names, test_size=test_size_num, random_state=42) 

train_features, valid_features, train_labels, valid_labels, train_names, valid_names = train_test_split(
    train_features, train_labels, train_names, test_size=0.05, random_state=42) 

max_value = np.max(np.max(train_labels, axis=-1))
min_value = np.min(np.min(train_labels, axis=-1))
#min_value = 0 # when using min clip, mouse will have capped lower bound problem

#-------------------------------------------#
# Step 2. Select the Architecture and Train
#-------------------------------------------#
def create_model(input_size, num_classes, batch_size, min_value, max_value, combining_weight, weight_flag, regularization_weight): 
    ## Build models based on Basset
    from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout, Activation, BatchNormalization
    from tensorflow.keras.models import Sequential     

    model = Sequential()
    model.add(Conv1D(filters=300, kernel_size=19, padding='same',# , activation='relu'
                 input_shape=(input_size, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=3, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv1D(filters=200, kernel_size=11, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=4, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv1D(filters=200, kernel_size=7, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=4, padding='same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(Dropout(0.3)) 

    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(Dropout(0.3)) 

    model.add(Dense(num_classes)) 

    opt = tensorflow.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False) 
    model.compile(loss=losses.combine_loss_by_sample(combining_weight),  
                  optimizer=opt,  
                  metrics=['mse', losses.pearson_loss, losses.r2_score])   

    return model


TL_manner = 'finetune' 
run_num = 'Basset_adam' + input_bp_txt + '_lr0001_dropout_03_conv_relu_max_BN_convfc_batch_' + str(batch_size) + \
          '_loss_combine'  + combining_weight_txt + '_bysample' + '_' + 'epoch' + str(num_epochs) + '_best_separatelayer'  
model_name = 'model' + '_' + TL_manner + '_M2H_01lr' + '_' + run_num + '/'

## Create output directory
folder = '../'
output_directory = os.path.join(folder + 'results/models/keras_version', subfolder, model_name)
directory = os.path.dirname(output_directory)
if not os.path.exists(directory):
    print('Creating directory %s' % output_directory)
    os.makedirs(directory)
else:
     print('Directory %s exists' % output_directory)

input_size = train_features.shape[1]
# Load pre-trained model
pretrained_model_name = 'model_Mouse_' + 'Basset_adam' + input_bp_txt + '_lr0001_dropout_03_conv_relu_max_BN_convfc_batch_' + str(batch_size) + \
          '_loss_combine001' + '_bysample' + '_epoch' + str(num_epochs) + '_best_separatelayer'  
json_file = open(folder + 'results/models/' + platform_txt + 'weighted_loss/' + pretrained_model_name + '/whole_model_best.json', 'r')
model_json = json_file.read()
json_file.close()
model = tensorflow.keras.models.model_from_json(model_json) 
model.load_weights(folder + 'results/models/' + platform_txt + 'weighted_loss/' + pretrained_model_name + '/whole_model_best_weights.h5')
print('Loaded model from disk')
# For fine-tuning purpose, truncate the original last layer
model._layers.pop() 
model.outputs = [model.layers[-1].output]
from tensorflow.keras.layers import Dense
model.add(Dense(num_classes))
if TL_manner == 'frozen':
    for layer in model.layers[:-1]:
        print(layer.name)
        layer.trainable = False
model.summary()

opt = tensorflow.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False) 
model.compile(loss=losses.combine_loss_by_sample(combining_weight), 
                optimizer=opt,  
                metrics=['mse', losses.pearson_loss, losses.r2_score]) 

checkpoint_path_weights = output_directory + 'cp.ckpt' 

## Create checkpoint callback
cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_path_weights, 
                                                save_weights_only=True,
                                                save_best_only=True,
                                                verbose=1)

tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir=os.path.join(output_directory, 'logs', '{}'.format(time())),
                                            histogram_freq=1, 
                                            batch_size=batch_size,
                                            write_graph=True,
                                            write_grads=True,
                                            write_images=False,
                                            update_freq='batch'
                                             ) 

history = model.fit(train_features, train_labels, batch_size=batch_size,
                    epochs=num_epochs, verbose=1, validation_data=(valid_features, valid_labels),
                    callbacks=[cp_callback, tensorboard]) 

# Save the whole model-latest
model.save(output_directory + 'whole_model_latest.h5')
model_json = model.to_json()
with open(output_directory + 'whole_model_latest.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights(output_directory + 'whole_model_latest_weights.h5')

# Save the training/validation curve  
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(output_directory + "training_valid_loss.svg")
plt.close()

plt.figure()
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model mean squared error')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(output_directory + "training_valid_metric_mse.svg")
plt.close()

plt.figure()
plt.plot(history.history['pearson_loss'])
plt.plot(history.history['val_pearson_loss'])
plt.title('model pearson loss')
plt.ylabel('pearson loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(output_directory + "training_valid_metric_corr.svg")
plt.close()

plt.figure()
plt.plot(history.history['r2_score'])
plt.plot(history.history['val_r2_score'])
plt.title('model r2 score')
plt.ylabel('r2 score')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(output_directory + "training_valid_metric_r2_score.svg")
plt.close()

#-------------------------------------------#
# Step 3. Evaluate
#-------------------------------------------#
# Using the model with the latest results
predicted_labels = model.predict(np.stack(test_features))
title = "basset_cor_hist_latest.svg"
correlations = plot_utils.plot_cors(test_labels, predicted_labels, output_directory, title)
    
# Using the Best weights 
model = create_model(input_size, num_classes, batch_size, min_value, max_value, combining_weight, weight_flag, regularization_weight)
model.load_weights(checkpoint_path_weights)
model.save(output_directory + 'whole_model_best.h5')
model_json = model.to_json()
with open(output_directory + 'whole_model_best.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights(output_directory + 'whole_model_best_weights.h5')

predicted_labels = model.predict(np.stack(test_features))
# save predictions and variables
np.save(output_directory + 'predictions.npy', predicted_labels)
np.save(output_directory + 'test_data.npy', test_features)
np.save(output_directory + 'test_labels.npy', test_labels)
np.save(output_directory + 'test_OCR_names.npy', test_names)

title = "basset_cor_hist_best.svg"
correlations = plot_utils.plot_cors(test_labels, predicted_labels, output_directory, title)
np.save(output_directory + 'correlations.npy', correlations)
plot_utils.plot_corr_variance(test_labels, correlations, output_directory)
quantile_indx = plot_utils.plot_cors_piechart(correlations, test_labels, output_directory) 
plot_utils.plot_random_predictions(test_labels, predicted_labels, correlations, quantile_indx, test_names, output_directory, len(class_names), class_names)
plot_utils.plot_random_predictions(test_labels, predicted_labels, correlations, quantile_indx, test_names, output_directory, len(class_names), class_names, scale=False)

# Clear out session
tensorflow.keras.backend.clear_session()