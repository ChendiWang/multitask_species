#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12400:33 2020
Nullify the filters one at a time to compute influence
@author: chendi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

import losses

def create_filter_model(original_model, input_size, target_layer_name): 
    # # Build models based on Basset
    from tensorflow.keras.layers import Input, Multiply      

    # # Automatically extract the original layers
    custom_mask_filter_in = Input(shape=([input_size, 300]), name='custom_mask_filter_in')       
    feature_input = Input(shape=(input_size, 4), name='feature_input') # feature_input = original_model.input
    layer_names = []
    for i, layer in enumerate(original_model.layers):
        original_model.layers[i].trainable = False
        # print(layer.name)
        layer_names.append(layer.name)
        if i == 0:
            nn = original_model.layers[i](feature_input)
        else:
            if layer.name == target_layer_name: # This can be a list of layer names
                nn = Multiply(name='activation_convlayer1_filter_masked')([nn, custom_mask_filter_in])
            nn = original_model.layers[i](nn)

    model = tf.keras.models.Model(inputs=[feature_input, custom_mask_filter_in], outputs=nn)

    # # Deep copy the weights from original model to this model by layer name
    for layer_name in layer_names:
        model.get_layer(layer_name).set_weights(weights=original_model.get_layer(layer_name).get_weights())    

    return model 


def create_filter_model_multitask(original_model, num_classes, input_size, target_layer_name): 
    from tensorflow.keras.layers import Input, Multiply   
    # Automatically extract the original layers
    custom_mask_filter_in = Input(shape=([input_size, 300]), name='custom_mask_filter_in')       
    feature_input = Input(shape=(input_size, 4), name='feature_input') # feature_input = original_model.input
    layer_names = []
    for i, layer in enumerate(original_model.layers):
        original_model.layers[i].trainable = False
        print(layer.name)
        layer_names.append(layer.name)    
        if layer.name == target_layer_name: # This can be a list of layer names
            nn = Multiply(name='activation_convlayer1_filter_masked')([nn, custom_mask_filter_in])
            nn = layer(nn)
        else:
            nn = layer(layer.input)
    model = tf.keras.models.Model(inputs=[feature_input, custom_mask_filter_in], outputs=nn)

    # # Deep copy the weights from original model to this model by layer name
    for layer_name in layer_names:
        model.get_layer(layer_name).set_weights(weights=original_model.get_layer(layer_name).get_weights())    

    return model 


if __name__ == "__main__":

    print("Utils for computing the influence.")

    