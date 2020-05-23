# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:55:05 2019
Loss functions
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.contrib.distributions import percentile
import numpy as np

def pearson_loss(obs, pred):
    mean_obs = K.mean(obs, axis=1, keepdims=True)
    mean_pred = K.mean(pred, axis=1, keepdims=True)
    obs_mean, pred_mean = obs - mean_obs, pred - mean_pred

    obs_norm = tf.nn.l2_normalize(obs_mean, axis=1)
    pred_norm = tf.nn.l2_normalize(pred_mean, axis=1)

    return tf.losses.cosine_distance(obs_norm, pred_norm, axis=1) # , reduction=tf.losses.Reduction.MEAN


def pearson_loss_by_sample(obs, pred):
    mean_obs = K.mean(obs, axis=1, keepdims=True)
    mean_pred = K.mean(pred, axis=1, keepdims=True)
    obs_mean, pred_mean = obs - mean_obs, pred - mean_pred

    obs_norm = tf.nn.l2_normalize(obs_mean, axis=1)
    pred_norm = tf.nn.l2_normalize(pred_mean, axis=1)

    return tf.losses.cosine_distance(obs_norm, pred_norm, axis=1, reduction=tf.losses.Reduction.NONE)

# Fixed weighting for combining by each sample
def combine_loss_by_sample(weight):
    def custom_combine_loss(obs, pred): 
        loss_mse = tf.keras.losses.mean_squared_error(obs, pred) # vector of a batch
        loss_mse = tf.expand_dims(loss_mse, 1)
        loss_p = pearson_loss_by_sample(obs, pred) # vector of a batch
        loss = loss_p + weight*loss_mse 
        return K.mean(loss, axis=-1) # should this be 0 or 1? 0 is the average loss across the batch, why 1 still works, a vector of all loss in the batch
    return custom_combine_loss

   
def poisson_loss(obs, pred):
   return tf.nn.log_poisson_loss(obs, tf.log(pred), compute_full_loss=True)


def huber_loss(delta):
    def custom_loss(obs, pred):
        loss = tf.losses.huber_loss(obs, pred, delta=delta)
        return loss
    return custom_loss


def quantile_loss(theta):
    def custom_loss(obs, pred):
        loss = tf.where(obs >= pred, theta*(tf.math.abs(obs-pred)), (1-theta)*(tf.math.abs(obs-pred)))
        return tf.math.reduce_sum(loss)
    return custom_loss


def r2_score_loss(y_true, y_pred): # in example_metrics: verified with r2_score_numpy, gives the same result
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return -1 * ( 1 - SS_res/(SS_tot + K.epsilon()) ) #epsilon=1e-8


def r2_score(y_true, y_pred): # in example_metrics: verified with r2_score_numpy, gives the same result
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) ) #epsilon=1e-8


def r2_score_numpy(y_true, y_pred): # verified with sklearn.metrics.r2_score, gives the same result
    ssres = np.sum(np.square(y_true - y_pred))
    sstot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - ssres / sstot


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
#        print('Hamming score: {0}'.format(hamming_score(y_true, y_pred)))
#        print('Subset accuracy: {0}'.format(sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))
#        print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_true, y_pred)))
    return np.mean(acc_list)



if __name__ == "__main__":
	print('This is util functions for losses.')