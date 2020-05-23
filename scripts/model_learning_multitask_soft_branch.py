#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb. 26 14:04:02 2020
Multitask for learning mouse and human together
@author: chendi
"""
# ## Step 0. Import packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split

import utils
import plot_utils
import losses
# import custom_layer

# Hyperparameters
num_epochs = 10
batch_size = 100 
learning_rate = 0.001
combining_weight = 0.01
combining_weight_txt = '001_00052'
regularization_weight = 0
regularization_txt = ''
weight_flag = 'minmax'
overlapped_celltype_txt = '_overlapped_celltype'
     
#-------------------------------------------#
# Step 1. Curate Data                
#-------------------------------------------#
species_1 = 'Mouse'
input_features_1 = np.load('../data/' + species_1 + '_Data/one_hot_seqs_ACGT.npy')
input_features_1 = input_features_1.astype(np.float32)
if input_features_1.shape[2] > input_features_1.shape[1]:
    input_features_1 = np.swapaxes(input_features_1,1,2)
peak_names_1 = np.load('../data/' + species_1 + '_Data/peak_names.npy')
species_tags_1 = np.ones((len(peak_names_1)))
input_labels_1 = np.load('../data/' + species_1 + '_Data/cell_type_array.npy')
input_labels_1 = input_labels_1.astype(np.float32)
# print(np.min(np.min(input_labels_1)))
with open('../data/' + species_1 + '_Data/cell_type_names.txt','r') as class_names_file:        
    class_names_1 = []
    for line in class_names_file:
        line = line.strip() # remove /n at the end
        task_num, class_name = line.split()
        class_names_1.append(class_name)
    class_names_1 = list(filter(None, class_names_1))  # This removes empty entries
num_classes_1 = len(class_names_1)

species_2 = 'Human'
input_features_2 = np.load('../data/' + species_2 + '_Data/one_hot_seqs_ACGT.npy')
input_features_2 = input_features_2.astype(np.float32)
if input_features_2.shape[2] > input_features_2.shape[1]:
    input_features_2 = np.swapaxes(input_features_2,1,2)
peak_names_2 = np.load('../data/' + species_2 + '_Data/peak_names.npy')
species_tags_2 = 2*np.ones((len(peak_names_2)))
input_labels_2 = np.load('../data/' + species_2 + '_Data/cell_type_array.npy')
input_labels_2 = input_labels_2.astype(np.float32)
# print(np.min(np.min(input_labels_2)))
with open('../data/' + species_2 + '_Data/cell_type_names.txt','r') as class_names_file:        
    class_names_2 = class_names_file.read().split('\t')
    class_names_2[-1] = class_names_2[-1].strip() # remove the last newline
    class_names_2 = list(filter(None, class_names_2))  # This removes empty entries
num_classes_2 = len(class_names_2)

# Find the overlapped subset of mouse/human cell types for training
if overlapped_celltype_txt: 
    with open('../data/Human_Data/mouse_human_cell_type_names.txt','r') as mapping_file:
        mapping = []
        for line in mapping_file:
            line = line.strip()
            mouse_cell, human_cell = line.split()        
            mapping.append([mouse_cell, human_cell])
        mapping = list(filter(None, mapping))  # This removes empty entries
    mapping = np.stack(mapping)
    # For Mouse
    selected_cells = np.unique(mapping[:,0])  
    index = np.in1d(class_names_1, selected_cells).nonzero()    
    selected_cells = [class_names_1[i] for i in index[0]]
    class_names_1 = selected_cells
    input_labels_1 = np.squeeze(input_labels_1[:, index], axis=1)
    num_classes_1 = len(class_names_1)
    del selected_cells, index
    print(input_features_1.shape)
    print(input_labels_1.shape)
    # For Human
    selected_cells = np.unique(mapping[:,1]) 
    index = np.in1d(class_names_2, selected_cells).nonzero()    
    selected_cells = [class_names_2[i] for i in index[0]]
    class_names_2 = selected_cells
    input_labels_2 = np.squeeze(input_labels_2[:, index], axis=1)
    num_classes_2 = len(class_names_2)
    del selected_cells, index
    print(input_features_2.shape)
    print(input_labels_2.shape)

# TODO: Find the orthologous gene locations, and separate from train and valid and test
# Each batch has random mixture of mouse and human samples
# Pad the labels of the opposite class with 0
padded_input_labels_1 = np.zeros((input_labels_1.shape[0], num_classes_2), dtype=np.float32)
input_labels_1 = np.concatenate((input_labels_1, padded_input_labels_1), axis=1)

padded_input_labels_2 = np.zeros((input_labels_2.shape[0], num_classes_1), dtype=np.float32)
input_labels_2 = np.concatenate((padded_input_labels_2, input_labels_2), axis=1)

train_features_1, test_features_1, train_labels_1, test_labels_1, train_names_1, test_names_1, train_tags_1, test_tags_1 = train_test_split(
    input_features_1, input_labels_1, peak_names_1, species_tags_1, test_size=15786, random_state=42) # 15786

train_features_2, test_features_2, train_labels_2, test_labels_2, train_names_2, test_names_2, train_tags_2, test_tags_2 = train_test_split(
    input_features_2, input_labels_2, peak_names_2, species_tags_2, test_size=25802, random_state=42) # 25802

num_classes = num_classes_1 + num_classes_2
class_names = class_names_1 + class_names_2

test_features = np.concatenate((test_features_1, test_features_2), axis=0)
test_labels = np.concatenate((test_labels_1, test_labels_2), axis=0)
test_names = np.concatenate((test_names_1, test_names_2), axis=0)
test_tags = np.concatenate((test_tags_1, test_tags_2), axis=0)

train_features = np.concatenate((train_features_1, train_features_2), axis=0)
train_labels = np.concatenate((train_labels_1, train_labels_2), axis=0)
train_names = np.concatenate((train_names_1, train_names_2), axis=0)
train_tags = np.concatenate((train_tags_1, train_tags_2), axis=0)

# Keep the human/mouse ratio in all the batches
# Solution: shuffle each dataset first; then arrange them as 1.6345 ~ every 3 human +  every 2 mouse
ratio = len(species_tags_2) / len(species_tags_1) 
train_index_1 = np.arange(len(train_tags_1))
train_index_2 = np.arange(len(train_tags_2)) + len(train_tags_1)

# This magic number to bring the num to the same as the other task
insert_index = np.where(np.mod(train_index_1, ratio)>=0.958058/ratio)[0] #0.958058
# Use -1 as place holder to balance two datasets
train_index_1_insert = np.insert(train_index_1, insert_index, -1)
train_index_2_insert = train_index_2
# Pad the ends with value -1
if train_index_1_insert.size < train_index_2_insert.size:
    train_index_1_insert = np.append(train_index_1_insert, -1*np.ones(train_index_2_insert.size-train_index_1_insert.size, dtype=np.int64))
elif train_index_1_insert.size > train_index_2_insert.size:
    train_index_2_insert = np.append(train_index_2_insert, -1*np.ones(train_index_1_insert.size-train_index_2_insert.size, dtype=np.int64))
# Interlace the index from two tasks
reorder_index = np.vstack((train_index_1_insert, train_index_2_insert)).reshape((-1,), order='F')
reorder_index = reorder_index[reorder_index!=-1]

# Reorder the training set to mix two dataset by number ratio
train_features = train_features[reorder_index,:,:]
train_labels = train_labels[reorder_index,:]
train_names = train_names[reorder_index]
train_tags = train_tags[reorder_index]

train_features, valid_features, train_labels, valid_labels, train_names, valid_names, train_tags, valid_tags = train_test_split(
    train_features, train_labels, train_names, train_tags, test_size=0.05, random_state=42, shuffle=False)

del reorder_index, train_index_1, train_index_2, insert_index, train_index_1_insert, train_index_2_insert
del train_features_1, train_features_2, train_labels_1, train_labels_2, train_names_1, train_names_2, train_tags_1, train_tags_2
del test_features_1, test_features_2, test_labels_1, test_labels_2, test_names_1, test_names_2, test_tags_1, test_tags_2
del input_features_1, input_features_2, input_labels_1, input_labels_2, peak_names_1, peak_names_2, species_tags_1, species_tags_2

# Prepare information for the masking etc.
input_size = train_features.shape[1]

# Make the label mask based on tags
label_mask_task1_template = np.hstack([np.ones(num_classes_1), np.zeros(num_classes_2)])
label_mask_task2_template = np.hstack([np.zeros(num_classes_1), np.ones(num_classes_2)])
train_labels_mask = np.zeros(train_labels.shape)
valid_labels_mask = np.zeros(valid_labels.shape)
test_labels_mask = np.zeros(test_labels.shape)
for i in range(train_labels.shape[0]):
    train_labels_mask[i, :] = label_mask_task1_template if train_tags[i]==1 else label_mask_task2_template
for i in range(valid_labels.shape[0]):
    valid_labels_mask[i, :] = label_mask_task1_template if valid_tags[i]==1 else label_mask_task2_template
for i in range(test_labels.shape[0]):
    test_labels_mask[i, :] = label_mask_task1_template if test_tags[i]==1 else label_mask_task2_template

percent_share = [0.9, 0.8, 0.7, 0.65, 0.6]


#-------------------------------------------#
# Step 2. Select the Architecture and Train
#-------------------------------------------#
def create_model(input_size, num_classes_1, num_classes_2, batch_size, combining_weight, weight_flag, ratio=1.0, percent_share=None): #, input_tag
    ## Build models based on Basset
    from tensorflow.keras.layers import Input, Conv1D, Dense, MaxPooling1D, Flatten, Dropout, Activation, BatchNormalization
    from tensorflow.keras.layers import Masking, Add, Multiply, RepeatVector, Reshape, Permute, Lambda

    conv1_dim, conv2_dim, conv3_dim, fc1_dim, fc2_dim = 300, 200, 200, 1000, 1000

    def cross_stitch(inputs):
        in_1 = inputs[0]
        in_2 = inputs[1]
        share_rate = inputs[2]
        in_1 = Lambda(lambda x: x * share_rate)(in_1)
        in_2 = Lambda(lambda x: x * (1.0-share_rate))(in_2)
        return Add()([in_1, in_2])


    inputs = Input(shape=(input_size, 4), name='inputs')
    inputs1, inputs2 = inputs, inputs

    nn1_in = Conv1D(filters=conv1_dim, kernel_size=19, padding='same', name='conv_1_task1')(inputs1)
    nn2_in = Conv1D(filters=conv1_dim, kernel_size=19, padding='same', name='conv_1_task2')(inputs2)
    
    # Add in cross-stitch operation
    nn1 = Lambda(cross_stitch, name='cross_stitch_1_task1')([nn1_in, nn2_in, percent_share[0]])
    nn2 = Lambda(cross_stitch, name='cross_stitch_1_task2')([nn2_in, nn1_in, percent_share[0]])

    nn1 = Activation('relu', name='relu_1_task1')(nn1)    
    nn1 = MaxPooling1D(pool_size=3, strides=3, padding='same', name='maxpool_1_task1')(nn1)
    nn1 = BatchNormalization(name='BN_1_task1')(nn1)

    nn2 = Activation('relu', name='relu_1_task2')(nn2)     
    nn2 = MaxPooling1D(pool_size=3, strides=3, padding='same', name='maxpool_1_task2')(nn2)
    nn2 = BatchNormalization(name='BN_1_task2')(nn2)
    
    nn1_in = Conv1D(filters=conv2_dim, kernel_size=11, padding='same', name='conv_2_task1')(nn1)
    nn2_in = Conv1D(filters=conv2_dim, kernel_size=11, padding='same', name='conv_2_task2')(nn2)

    nn1 = Lambda(cross_stitch, name='cross_stitch_2_task1')([nn1_in, nn2_in, percent_share[1]])
    nn2 = Lambda(cross_stitch, name='cross_stitch_2_task2')([nn2_in, nn1_in, percent_share[1]])

    nn1 = Activation('relu', name='relu_2_task1')(nn1)
    nn1 = MaxPooling1D(pool_size=4, strides=4, padding='same', name='maxpool_2_task1')(nn1)
    nn1 = BatchNormalization(name='BN_2_task1')(nn1)
    
    nn2 = Activation('relu', name='relu_2_task2')(nn2)
    nn2 = MaxPooling1D(pool_size=4, strides=4, padding='same', name='maxpool_2_task2')(nn2)
    nn2 = BatchNormalization(name='BN_2_task2')(nn2)
    
    nn1_in = Conv1D(filters=conv3_dim, kernel_size=7, padding='same', name='conv_3_task1')(nn1)
    nn2_in = Conv1D(filters=conv3_dim, kernel_size=7, padding='same', name='conv_3_task2')(nn2)

    nn1 = Lambda(cross_stitch, name='cross_stitch_3_task1')([nn1_in, nn2_in, percent_share[2]])
    nn2 = Lambda(cross_stitch, name='cross_stitch_3_task2')([nn2_in, nn1_in, percent_share[2]])

    nn1 = Activation('relu', name='relu_3_task1')(nn1)
    nn1 = MaxPooling1D(pool_size=4, strides=4, padding='same', name='maxpool_3_task1')(nn1)
    nn1 = BatchNormalization(name='BN_3_task1')(nn1)
    
    nn2 = Activation('relu', name='relu_3_task2')(nn2)
    nn2 = MaxPooling1D(pool_size=4, strides=4, padding='same', name='maxpool_3_task2')(nn2)
    nn2 = BatchNormalization(name='BN_3_task2')(nn2)
    
    nn1 = Flatten(name='flatten_task1')(nn1)
    nn2 = Flatten(name='flatten_task2')(nn2)

    nn1_in = Dense(fc1_dim, name='dense_1_task1')(nn1)
    nn2_in = Dense(fc1_dim, name='dense_1_task2')(nn2)

    nn1 = Lambda(cross_stitch, name='cross_stitch_4_task1')([nn1_in, nn2_in, percent_share[3]])
    nn2 = Lambda(cross_stitch, name='cross_stitch_4_task2')([nn2_in, nn1_in, percent_share[3]])

    nn1 = BatchNormalization(name='BN_4_task1')(nn1)
    nn1 = Activation('relu', name='relu_4_task1')(nn1)
    nn1 = Dropout(0.3, name='dropout_1_task1')(nn1)
    
    nn2 = BatchNormalization(name='BN_4_task2')(nn2)
    nn2 = Activation('relu', name='relu_4_task2')(nn2)
    nn2 = Dropout(0.3, name='dropout_1_task2')(nn2) 

    nn1_in = Dense(fc2_dim, name='dense_2_task1')(nn1)
    nn2_in = Dense(fc2_dim, name='dense_2_task2')(nn2)

    nn1 = Lambda(cross_stitch, name='cross_stitch_5_task1')([nn1_in, nn2_in, percent_share[4]])
    nn2 = Lambda(cross_stitch, name='cross_stitch_5_task1')([nn2_in, nn1_in, percent_share[4]])

    nn1 = BatchNormalization(name='BN_5_task1')(nn1)
    nn1 = Activation('relu', name='relu_5_task1')(nn1)
    nn1 = Dropout(0.3, name='dropout_2_task1')(nn1)  

    nn2 = BatchNormalization(name='BN_5_task2')(nn2)
    nn2 = Activation('relu', name='relu_5_task2')(nn2)
    nn2 = Dropout(0.3, name='dropout_2_task2')(nn2)    
       
    ## two head version:
    labels_mask_1 = Input(shape=([num_classes_1]), dtype='float32', name='labels_mask_1')
    mask_1 = Masking(mask_value=0.0, name='mask_1')(labels_mask_1) 
    labels_mask_2 = Input(shape=([num_classes_2]), dtype='float32', name='labels_mask_2')
    mask_2 = Masking(mask_value=0.0, name='mask_2')(labels_mask_2)     

    result_1_ori = Dense(num_classes_1, name='dense_out_1_ori')(nn1)
    result_2_ori = Dense(num_classes_2, name='dense_out_2_ori')(nn2)

    result_1 = Multiply(name='dense_out_1')([result_1_ori, mask_1])
    result_2 = Multiply(name='dense_out_2')([result_2_ori, mask_2])

    model = tf.keras.models.Model(inputs=[inputs, labels_mask_1, labels_mask_2],
                                  outputs=[result_1, result_2]) 

    loss_combined = {
        'dense_out_1': losses.combine_loss_by_sample(0.01),
        'dense_out_2': losses.combine_loss_by_sample(0.0052)
    }
    lossWeights = {'dense_out_1': ratio, 'dense_out_2': 1.0} 

    metric = {
        'dense_out_1': ['mse', losses.pearson_loss, losses.r2_score],
        'dense_out_2': ['mse', losses.pearson_loss, losses.r2_score]
    }

    opt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False) 
   
    model.compile(optimizer=opt,
                  loss=loss_combined,
                  loss_weights=lossWeights,
                  metrics=metric)
    return model

run_num = 'Basset_adam_lr0001_dropout_03_conv_relu_max_BN_convfc_batch' + str(batch_size) + \
          '_loss_combine' + combining_weight_txt + '_bysample' + '_' + 'epoch' + str(num_epochs) + \
          '_best_separatelayer_soft_branched_cross_stitch' + overlapped_celltype_txt 
model_name = 'model' + '_multitask_MaH_ratiobased_' + run_num 

# Create output directory
folder = '/media/chendi/DATA2/projects/XDNN/mhTransferLearning/immuneMH/'
#folder = '../'
output_directory = folder + 'results/models/keras_version/multitask/' + model_name + '/'
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
                         histogram_freq=1,
                         batch_size=batch_size,
                         write_graph=True,
                         write_grads=True,
                         write_images=False,
                         update_freq='batch')

model = create_model(input_size, num_classes_1, num_classes_2, batch_size, combining_weight, weight_flag, ratio=1.0, percent_share=percent_share)
model.summary()
history = model.fit([train_features, train_labels_mask[:, :num_classes_1], train_labels_mask[:, -num_classes_2:]], 
                    [train_labels[:, :num_classes_1], train_labels[:, -num_classes_2:]], 
                    batch_size=batch_size,
                    epochs=num_epochs, 
                    verbose=1, 
                    validation_data=([valid_features, valid_labels_mask[:, :num_classes_1], valid_labels_mask[:, -num_classes_2:]], 
                                     [valid_labels[:, :num_classes_1], valid_labels[:, -num_classes_2:]]), 
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
plt.plot(history.history['dense_out_1_loss'])
plt.plot(history.history['val_dense_out_1_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(output_directory + "training_valid_loss_1.svg")
plt.close()

plt.figure()
plt.plot(history.history['dense_out_2_loss'])
plt.plot(history.history['val_dense_out_2_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(output_directory + "training_valid_loss_2.svg")
plt.close()

plt.figure()
plt.plot(history.history['dense_out_1_mean_squared_error'])
plt.plot(history.history['val_dense_out_1_mean_squared_error'])
plt.title('model mean squared error')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(output_directory + "training_valid_metric_mse_1.svg")
plt.close()

plt.figure()
plt.plot(history.history['dense_out_2_mean_squared_error'])
plt.plot(history.history['val_dense_out_2_mean_squared_error'])
plt.title('model mean squared error')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(output_directory + "training_valid_metric_mse_2.svg")
plt.close()

plt.figure()
plt.plot(history.history['dense_out_1_pearson_loss'])
plt.plot(history.history['val_dense_out_1_pearson_loss'])
plt.title('model pearson loss')
plt.ylabel('pearson loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(output_directory + "training_valid_metric_pearson_loss_1.svg")
plt.close()

plt.figure()
plt.plot(history.history['dense_out_2_pearson_loss'])
plt.plot(history.history['val_dense_out_2_pearson_loss'])
plt.title('model pearson loss')
plt.ylabel('pearson loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(output_directory + "training_valid_metric_pearson_loss_2.svg")
plt.close()

plt.figure()
plt.plot(history.history['dense_out_1_r2_score'])
plt.plot(history.history['val_dense_out_1_r2_score'])
plt.title('model r2 score')
plt.ylabel('r2 score')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(output_directory + "training_valid_metric_r2_score_1.svg")
plt.close()

plt.figure()
plt.plot(history.history['dense_out_2_r2_score'])
plt.plot(history.history['val_dense_out_2_r2_score'])
plt.title('model r2 score')
plt.ylabel('r2 score')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(output_directory + "training_valid_metric_r2_score_2.svg")
plt.close()

#-------------------------------------------#
# Step 3. Evaluate
#-------------------------------------------#
# Using the model with the latest results
[predicted_labels_1, predicted_labels_2] = model.predict([test_features, test_labels_mask[:, :num_classes_1], test_labels_mask[:, -num_classes_2:]])

title = "basset_cor_hist_latest_class_1.svg"
correlations_1 = plot_utils.plot_cors(test_labels[test_tags==1, :][:, :num_classes_1], predicted_labels_1[test_tags==1, :], output_directory, title)

title = "basset_cor_hist_latest_class_2.svg"
correlations_2 = plot_utils.plot_cors(test_labels[test_tags==2, :][:, -num_classes_2:], predicted_labels_2[test_tags==2, :], output_directory, title)

# Using the Best weights 
model = create_model(input_size, num_classes_1, num_classes_2, batch_size, combining_weight, weight_flag, ratio=1.0, percent_share=percent_share)
model.load_weights(checkpoint_path_weights)
model.save(output_directory + 'whole_model_best.h5')
model_json = model.to_json()
with open(output_directory + 'whole_model_best.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights(output_directory + 'whole_model_best_weights.h5')

[predicted_labels_1, predicted_labels_2] = model.predict([test_features, test_labels_mask[:, :num_classes_1], test_labels_mask[:, -num_classes_2:]])

title = "basset_cor_hist_best_class_1.svg"
correlations_1 = plot_utils.plot_cors(test_labels[test_tags==1, :][:, :num_classes_1], predicted_labels_1[test_tags==1, :], output_directory, title)

title = "basset_cor_hist_best_class_2.svg"
correlations_2 = plot_utils.plot_cors(test_labels[test_tags==2, :][:, -num_classes_2:], predicted_labels_2[test_tags==2, :], output_directory, title)


quantile_indx_1 = plot_utils.plot_cors_piechart(correlations_1, test_labels[test_tags==1, :][:, :num_classes_1], output_directory, 'basset_cor_pie_class_1.svg')

plot_utils.plot_random_predictions(test_labels[test_tags==1, :][:, :num_classes_1], predicted_labels_1[test_tags==1, :], correlations_1, quantile_indx_1, test_names[test_tags==1], output_directory, len(class_names_1), class_names_1, 'class_1_')

plot_utils.plot_random_predictions(test_labels[test_tags==1, :][:, :num_classes_1], predicted_labels_1[test_tags==1, :], correlations_1, quantile_indx_1, test_names[test_tags==1], output_directory, len(class_names_1), class_names_1, 'class_1_', scale=False)

quantile_indx_2 = plot_utils.plot_cors_piechart(correlations_2, test_labels[test_tags==2, :][:, -num_classes_2:], output_directory, 'basset_cor_pie_class_2.svg')

plot_utils.plot_random_predictions(test_labels[test_tags==2, :][:, -num_classes_2:], predicted_labels_2[test_tags==2, :], correlations_2, quantile_indx_2, test_names[test_tags==2], output_directory, len(class_names_2), class_names_2, 'class_2_')

plot_utils.plot_random_predictions(test_labels[test_tags==2, :][:, -num_classes_2:], predicted_labels_2[test_tags==2, :], correlations_2, quantile_indx_2, test_names[test_tags==2], output_directory, len(class_names_2), class_names_2, 'class_2_', scale=False)

#save predictions
np.save(output_directory + 'predictions_class_1.npy', predicted_labels_1[test_tags==1, :])
np.save(output_directory + 'predictions_class_2.npy', predicted_labels_2[test_tags==2, :])

#save correlations
np.save(output_directory + 'correlations_class_1.npy', correlations_1)
np.save(output_directory + 'correlations_class_2.npy', correlations_2)

np.save(output_directory + 'test_data_class_1.npy', test_features[test_tags==1, :, :])
np.save(output_directory + 'test_labels_class_1.npy', test_labels[test_tags==1, :][:, :num_classes_1])
np.save(output_directory + 'test_OCR_names_class_1.npy', test_names[test_tags==1])

np.save(output_directory + 'test_data_class_2.npy', test_features[test_tags==2, :, :])
np.save(output_directory + 'test_labels_class_2.npy', test_labels[test_tags==2, :][:, -num_classes_2:])
np.save(output_directory + 'test_OCR_names_class_2.npy', test_names[test_tags==2])

# Evaluate prediction magnitude
# Class 1
plt.clf()
plt.hist(np.max(predicted_labels_1[test_tags==1, :], axis=-1) - np.min(predicted_labels_1[test_tags==1, :], axis=-1), bins=50)
try:
    plt.title("Average minmax = {%f}" % np.mean(np.max(predicted_labels_1[test_tags==1, :], axis=-1) - np.min(predicted_labels_1[test_tags==1, :], axis=-1)))
except Exception as e:
    print("could not set the title for graph")
    print(e)
plt.ylabel("Frequency")
plt.xlabel("Max - Min")
plt.savefig(output_directory + 'selected_model_predicted_labels_minmax_hist_testset_class_1.svg')
plt.close()

plt.clf()
plt.hist(np.max(test_labels[test_tags==1, :][:, :num_classes_1], axis=-1) - np.min(test_labels[test_tags==1, :][:, :num_classes_1], axis=-1), bins=50)
try:
    plt.title("Average minmax = {%f}" % np.mean(np.max(test_labels[test_tags==1, :][:, :num_classes_1], axis=-1) - np.min(test_labels[test_tags==1, :][:, :num_classes_1], axis=-1)))
except Exception as e:
    print("could not set the title for graph")
    print(e)
plt.ylabel("Frequency")
plt.xlabel("Max - Min")
plt.savefig(output_directory + 'selected_model_ground_truth_labels_minmax_hist_testset_class_1.svg')
plt.close()

plt.clf()
plt.hist(np.max(predicted_labels_1[test_tags==1, :], axis=-1), bins=50)
try:
    plt.title("Average max = {%f}" % np.mean(np.max(predicted_labels_1[test_tags==1, :], axis=-1)))
except Exception as e:
    print("could not set the title for graph")
    print(e)
plt.ylabel("Frequency")
plt.xlabel("Max")
plt.savefig(output_directory + 'selected_model_predicted_labels_max_hist_testset_class_1.svg')
plt.close()

plt.clf()
plt.hist(np.max(test_labels[test_tags==1, :][:, -num_classes_1:], axis=-1), bins=50)
try:
    plt.title("Average max = {%f}" % np.mean(np.max(test_labels[test_tags==1, :][:, -num_classes_1:], axis=-1)))
except Exception as e:
    print("could not set the title for graph")
    print(e)
plt.ylabel("Frequency")
plt.xlabel("Max")
plt.savefig(output_directory + 'selected_model_ground_truth_labels_max_hist_testset_class_1.svg')
plt.close()

# Class 2
plt.clf()
plt.hist(np.max(predicted_labels_2[test_tags==2, :], axis=-1) - np.min(predicted_labels_2[test_tags==2, :], axis=-1), bins=50)
try:
    plt.title("Average minmax = {%f}" % np.mean(np.max(predicted_labels_2[test_tags==2, :], axis=-1) - np.min(predicted_labels_2[test_tags==2, :], axis=-1)))
except Exception as e:
    print("could not set the title for graph")
    print(e)
plt.ylabel("Frequency")
plt.xlabel("Max - Min")
plt.savefig(output_directory + 'selected_model_predicted_labels_minmax_hist_testset_class_2.svg')
plt.close()

plt.clf()
plt.hist(np.max(test_labels[test_tags==2, :][:, -num_classes_2:], axis=-1) - np.min(test_labels[test_tags==2, :][:, -num_classes_2], axis=-1), bins=50)
try:
    plt.title("Average minmax = {%f}" % np.mean(np.max(test_labels[test_tags==2, :][:, -num_classes_2:], axis=-1) - np.min(test_labels[test_tags==2, :][:, -num_classes_2:], axis=-1)))
except Exception as e:
    print("could not set the title for graph")
    print(e)
plt.ylabel("Frequency")
plt.xlabel("Max - Min")
plt.savefig(output_directory + 'selected_model_ground_truth_labels_minmax_hist_testset_class_2.svg')
plt.close()

plt.clf()
plt.hist(np.max(predicted_labels_2[test_tags==2, :], axis=-1), bins=50)
try:
    plt.title("Average max = {%f}" % np.mean(np.max(predicted_labels_2[test_tags==2, :], axis=-1)))
except Exception as e:
    print("could not set the title for graph")
    print(e)
plt.ylabel("Frequency")
plt.xlabel("Max")
plt.savefig(output_directory + 'selected_model_predicted_labels_max_hist_testset_class_2.svg')
plt.close()

plt.clf()
plt.hist(np.max(test_labels[test_tags==2, :][:, -num_classes_2:], axis=-1), bins=50)
try:
    plt.title("Average max = {%f}" % np.mean(np.max(test_labels[test_tags==2, :][:, -num_classes_2:], axis=-1)))
except Exception as e:
    print("could not set the title for graph")
    print(e)
plt.ylabel("Frequency")
plt.xlabel("Max")
plt.savefig(output_directory + 'selected_model_ground_truth_labels_max_hist_testset_class_2.svg')
plt.close()

tf.keras.backend.clear_session()