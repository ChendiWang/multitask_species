#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:02:37 2019

@author: chendi
"""
# ## Step 0. Import packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
#import requests
#import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time

import tensorflow as tf

import plot_utils
import losses

# Hyperparameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001
weight = 0.01
     
#-------------------------------------------#
# Step 1. Curate Data                
#-------------------------------------------#
species = 'Mouse'
input_features = np.load('../data/' + species + '_Data/one_hot_seqs_ACGT.npy')
input_features = input_features.astype(np.float32)
input_features = np.swapaxes(input_features,1,2) # Input dim is different from pytorch
peak_names = np.load('../data/' + species + '_Data/peak_names.npy')
species_tag = np.ones((len(peak_names)))
input_labels = np.load('../data/' + species + '_Data/cell_type_array.npy')
with open('../data/' + species + '_Data/cell_type_names.txt','r') as class_names_file:        
    class_names = []
    for line in class_names_file:
        line = line.strip() # remove /n at the end
        task_num, class_name = line.split()
        class_names.append(class_name)
    class_names = list(filter(None, class_names))  # This removes empty entries
num_classes = len(class_names)

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels, train_names, test_names, train_species_tag, test_species_tag = train_test_split(
    input_features, input_labels, peak_names, species_tag, test_size=15786, random_state=42) 

train_features, valid_features, train_labels, valid_labels, train_names, valid_names, train_species_tag, valid_species_tag = train_test_split(
    train_features, train_labels, train_names, train_species_tag, test_size=0.05, random_state=42)
  
#-------------------------------------------#
# Step 2. Select the Architecture and Train
#-------------------------------------------#
def create_model(input_size, num_classes): 
    ## Build models based on Basset
    from tensorflow.keras.layers import Input, Conv1D, Dense, MaxPooling1D, Flatten, Dropout, Activation, BatchNormalization# BatchNormalization(different in keras standalone)
    
    feature_input = Input(shape=(input_size, 4), name='input')
    nn = Conv1D(filters=300, kernel_size=19, padding='same', name='conv_1')(feature_input)
    nn = Activation('relu', name='relu_1')(nn)
    nn = MaxPooling1D(pool_size=3, strides=3, padding='same', name='maxpool_1')(nn)
    nn = BatchNormalization(name='BN_1')(nn)
    
    nn = Conv1D(filters=200, kernel_size=11, padding='same', name='conv_2')(nn)
    nn = Activation('relu', name='relu_2')(nn)
    nn = MaxPooling1D(pool_size=4, strides=4, padding='same', name='maxpool_2')(nn)
    nn = BatchNormalization(name='BN_2')(nn)
    
    nn = Conv1D(filters=200, kernel_size=7, padding='same', name='conv_3')(nn)
    nn = Activation('relu', name='relu_3')(nn)
    nn = MaxPooling1D(pool_size=4, strides=4, padding='same', name='maxpool_3')(nn)
    nn = BatchNormalization(name='BN_3')(nn)
    
    nn = Flatten(name='flatten')(nn)
    
    nn = Dense(1000, name='dense_1')(nn)
    nn = Activation('relu', name='relu_4')(nn)
    nn = Dropout(0.3, name='dropout_1')(nn)
    
    nn = Dense(1000, name='dense_2')(nn)
    nn = Activation('relu', name='relu_5')(nn)
    nn = Dropout(0.3, name='dropout_2')(nn)
    
    result = Dense(num_classes, name='dense_out')(nn)
    
    model = tf.keras.models.Model(inputs=feature_input, 
                                  outputs=result)  
    
    opt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False) 
    
    model.compile(loss=losses.pearson_loss, 
                  optimizer=opt, 
                  metrics=['mse', losses.pearson_loss]) 

    return model


# ## Model training 
# We will further divide the training set into a training and validation set. 
# We will train only on the reduced training set, but plot the loss curve on both the training and validation sets. 
# __Early stopping__: Once the loss for the validation set stops improving or gets worse throughout the learning cycles, 
# it is time to stop training because the model has already converged and may be just overfitting.


run_num = 'Basset_adam_lr0001_dropout_03_conv_relu_max_BN_batch_' + str(batch_size) + \
    '_loss_pearson_' + 'epoch' + str(num_epochs) + '_best_separatelayer'
model_name = 'model' + species + '_' + run_num + '_nn'#+ 'PoissionNLLoss_'

# Create output directory
folder = '/media/chendi/DATA2/projects/XDNN/mhTransferLearning/immuneMH/'
#folder = '../'
output_directory = folder + 'results/models/keras_version/' + model_name + '/'
directory = os.path.dirname(output_directory)
if not os.path.exists(directory):
    print('Creating directory %s' % output_directory)
    os.makedirs(directory)
else:
     print('Directory %s exists' % output_directory)

checkpoint_path_weights = output_directory + 'cp.ckpt' 

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_weights, 
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=output_directory+'logs/{}'.format(time()),
                         histogram_freq=128,
                         batch_size=batch_size,
                         write_graph=True,
                         write_grads=True,
                         write_images=False,
                         update_freq='batch')

input_size = train_features.shape[1]
model = create_model(input_size, num_classes)
history = model.fit(train_features, train_labels, batch_size=batch_size,
                    epochs=num_epochs, verbose=1, validation_data=(valid_features, valid_labels), 
                    callbacks=[cp_callback, tensorboard])

# Save the whole model-latest
model.save(output_directory + 'whole_model_latest.h5')

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
plt.savefig(output_directory + "training_valid_metric.svg")
plt.close()

#-------------------------------------------#
# Step 3. Evaluate
#-------------------------------------------#
# Using the model with the latest results
predicted_labels = model.predict(np.stack(test_features))

title = "basset_cor_hist_latest.svg"
correlations = plot_utils.plot_cors(test_labels, predicted_labels, output_directory, title)

# Using the Best weights 
model = create_model(input_size, num_classes)
model.load_weights(checkpoint_path_weights)
model.save(output_directory + 'whole_model_best.h5')
predicted_labels = model.predict(np.stack(test_features))

title = "basset_cor_hist_best.svg"
correlations = plot_utils.plot_cors(test_labels, predicted_labels, output_directory, title)

plot_utils.plot_corr_variance(test_labels, correlations, output_directory)

quantile_indx = plot_utils.plot_cors_piechart(correlations, test_labels, output_directory)

plot_utils.plot_random_predictions(test_labels, predicted_labels, correlations, quantile_indx, test_names, output_directory, len(class_names), class_names)

#save predictions
np.save(output_directory + 'predictions.npy', predicted_labels)

#save correlations
np.save(output_directory + 'correlations.npy', correlations)

np.save(output_directory + 'test_data.npy', test_features)
np.save(output_directory + 'test_labels.npy', test_labels)
np.save(output_directory + 'test_OCR_names.npy', test_names)

tf.keras.backend.clear_session()