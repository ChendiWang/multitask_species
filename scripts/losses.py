# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:55:05 2019
Loss functions
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.contrib.distributions import percentile
#from keras import backend as K
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


def weighted_pearson(obs, pred):
    mean_obs = K.mean(obs, axis=1, keepdims=True)
    mean_pred = K.mean(pred, axis=1, keepdims=True)
    obs_mean, pred_mean = obs - mean_obs, pred - mean_pred

    obs_norm = tf.nn.l2_normalize(obs_mean, axis=1)
    pred_norm = tf.nn.l2_normalize(pred_mean, axis=1)
    
    mean, var = tf.nn.moments(obs, axes=[1])
    var = tf.math.sqrt(var)
    var = tf.expand_dims(var, 1)
    var_sum = tf.math.reduce_sum(var)
    loss = tf.losses.cosine_distance(obs_norm, pred_norm, weights=var, axis=1, reduction='weighted_sum')
    loss = tf.div_no_nan(loss, var_sum)
    return loss


def weighted_mse(obs, pred):
    mean, var = tf.nn.moments(obs, axes=[1])
    var = tf.math.sqrt(var)
    var = tf.expand_dims(var, 1)
    var_sum = tf.math.reduce_sum(var)
    loss = tf.losses.mean_squared_error(obs, pred, weights=var, reduction='weighted_sum')
    loss = tf.div_no_nan(loss, var_sum)
    return loss


def inverse_weighted_mse(obs, pred):
    mean, var = tf.nn.moments(obs, axes=[1])
    var = tf.math.sqrt(var)
    var = tf.math.divide(1, var)
    var = tf.expand_dims(var, 1)
    var_sum = tf.math.reduce_sum(var)
    loss= tf.losses.mean_squared_error(obs, pred, weights=var, reduction='weighted_sum')
    loss = tf.div_no_nan(loss, var_sum)
    return loss


# Fixed weighting for combining
def combine_loss(weight): # combining the average loss within the batch
    def custom_combine_loss(obs, pred): 
        loss_mse = tf.keras.losses.mean_squared_error(obs, pred) # vector of a batch
        loss_p = pearson_loss(obs, pred) # mean value of the whole batch
        loss = loss_p + weight*loss_mse 
        return loss
    return custom_combine_loss


# Fixed weighting for combining by each sample
def combine_loss_by_sample(weight):
    def custom_combine_loss(obs, pred): 
        loss_mse = tf.keras.losses.mean_squared_error(obs, pred) # vector of a batch
        loss_mse = tf.expand_dims(loss_mse, 1)
        loss_p = pearson_loss_by_sample(obs, pred) # vector of a batch
        loss = loss_p + weight*loss_mse 
        return K.mean(loss, axis=-1) # should this be 0 or 1? 0 is the average loss across the batch, why 1 still works, a vector of all loss in the batch
    return custom_combine_loss


def fixed_weighted_combine(weight):
    def custom_weighted_combine_loss(obs, pred): 
        loss_mse = weighted_mse(obs, pred)
        loss_p = weighted_pearson(obs, pred)
        loss = loss_p + weight*loss_mse 
        return loss
    return custom_weighted_combine_loss


def fixed_inverse_weighted_combine(weight):
    def custom_inverse_weighted_combine_loss(obs, pred): 
        loss_mse = inverse_weighted_mse(obs, pred)
        loss_p = weighted_pearson(obs, pred)
#        weight = 0.001
        loss = loss_p + weight*loss_mse 
        return loss
    return custom_inverse_weighted_combine_loss


# adaptive weighting for combining
def inverse_weighted_combine_per_sample(obs, pred):
    mean_obs = K.mean(obs, axis=1, keepdims=True)
    mean_pred = K.mean(pred, axis=1, keepdims=True)
    obs_mean, pred_mean = obs - mean_obs, pred - mean_pred

    obs_norm = tf.nn.l2_normalize(obs_mean, axis=1)
    pred_norm = tf.nn.l2_normalize(pred_mean, axis=1)
    
    mean, var = tf.nn.moments(obs, axes=[1])
    var = tf.math.sqrt(var)    
    var = tf.expand_dims(var, 1)
    var_sum_p = tf.math.reduce_sum(var)
    loss_p = tf.losses.cosine_distance(obs_norm, pred_norm, weights=var, axis=1, reduction='weighted_sum')
    
    var_inverse = tf.math.divide(1, var) # inverse the weight for mse
    var_inverse_sum_m = tf.math.reduce_sum(var_inverse)
    loss_m = tf.losses.mean_squared_error(obs, pred, weights=var_inverse, reduction='weighted_sum')
    loss = tf.div_no_nan(loss_p + loss_m, var_sum_p + var_inverse_sum_m)
    return loss


def inverse_weighted_combine(batch_size, combining_weight, weight_flag): #_sample_sum_to_batch
# TODO: dynamic=True
    def custom_loss(obs, pred): 
        loss = 0.0
        i = 0
#        K.shape is not helpful for debugging because it is just a symbol, but it is the one to actually use in your code.
#        K.int_shape is helpful for debugging and checking, but is just some static metadata and won't normally reflect the current batch size.

#        def f1(): return i
#        def f2(): return K.shape(obs)[0]
#        while tf.cond(tf.less(tf.cast(i, tf.float32), obs), f1, f2): 
        
#        def custom_shape(tensor):
#            s = tensor.shape
#            return tuple([s[j].value for j in range(0, len(s))])
        
        while i < batch_size: #obs.get_shape()[0]:#K.shape(obs)[0]: #batch_size: 
            # The last batch might not have the fixed batch size...
#            if i == K.shape(obs)[0]: 
#                break
            mean_obs = K.mean(obs[i, :], keepdims=True) 
            mean_pred = K.mean(pred[i, :], keepdims=True)
            obs_mean, pred_mean = obs[i, :] - mean_obs, pred[i, :] - mean_pred
        
            obs_norm = tf.nn.l2_normalize(obs_mean) 
            pred_norm = tf.nn.l2_normalize(pred_mean)
            
            if weight_flag == 'var':
                mean, var = tf.nn.moments(obs[i, :], axes=[0]) 
                weight = var
            if weight_flag == 'std':
                mean, var = tf.nn.moments(obs[i, :], axes=[0])
                var = tf.math.sqrt(var) 
                weight = var
            if weight_flag == 'percentile9505':
                percentile_var = percentile(obs[i, :], q=95.) - percentile(obs[i, :], q=5.)
                weight = percentile_var
            if weight_flag == 'minmax':
                minmax_var = K.max(obs[i, :]) - K.min(obs[i, :])
                weight = minmax_var
            if weight_flag == 'minmaxsquare':
                minmax_var = K.max(obs[i, :]) - K.min(obs[i, :])
                weight = K.square(minmax_var) # when <1, this will be even smaller, which favors mse
            
            weight_inverse = tf.math.divide(1, weight) # inverse the weight for mse
            
            loss_p = tf.losses.cosine_distance(obs_norm, pred_norm, axis=0) #, weights=var, axis=0, reduction='weighted_sum'
            loss_m = tf.losses.mean_squared_error(obs[i, :], pred[i, :]) #, weights=var_inverse, reduction='weighted_sum'
#            loss_per_sample = tf.div_no_nan(weight * loss_p + weight_inverse * loss_m, weight + weight_inverse) 
            loss_per_sample = tf.div_no_nan(1 / combining_weight * weight * loss_p + combining_weight * weight_inverse * loss_m, 1 / combining_weight * weight + combining_weight * weight_inverse) 
            loss = tf.math.add(loss, loss_per_sample)  
            i += 1
        loss = tf.math.divide(loss, i) #.div_no_nan
        return loss
    return custom_loss
  
    
def double_weighted_combine(batch_size, combining_weight, weight_flag):
    def custom_loss_double_weighted(obs, pred): 
        loss = 0.0
        i = 0   
#        mean, var = tf.nn.moments(obs, axes=[1])
#        sample_weight = tf.math.sqrt(var)
#        sample_weight = tf.expand_dims(sample_weight, 1)
        # Use minmax as sample weight
        sample_weight = tf.expand_dims(K.max(obs, axis=-1) - K.min(obs, axis=-1), 1)
#        loss_per_sample = tf.Variable(tf.zeros(batch_size, tf.float32))
        while i < batch_size: #obs.get_shape()[0]:#K.shape(obs)[0]: #batch_size: 
            # The last batch might not have the fixed batch size...
#            if i == K.shape(obs)[0]: 
#                break
            mean_obs = K.mean(obs[i, :], keepdims=True) 
            mean_pred = K.mean(pred[i, :], keepdims=True)
            obs_mean, pred_mean = obs[i, :] - mean_obs, pred[i, :] - mean_pred
        
            obs_norm = tf.nn.l2_normalize(obs_mean) 
            pred_norm = tf.nn.l2_normalize(pred_mean)
            
            if weight_flag == 'var':
                mean, var = tf.nn.moments(obs[i, :], axes=[0]) 
                weight = var
            if weight_flag == 'std':
                mean, var = tf.nn.moments(obs[i, :], axes=[0])
                var = tf.math.sqrt(var) 
                weight = var
            if weight_flag == 'percentile9505':
                percentile_var = percentile(obs[i, :], q=95.) - percentile(obs[i, :], q=5.)
                weight = percentile_var
            if weight_flag == 'minmax':
                minmax_var = K.max(obs[i, :]) - K.min(obs[i, :])
                weight = minmax_var
            if weight_flag == 'minmaxsquare':
                minmax_var = K.max(obs[i, :]) - K.min(obs[i, :])
                weight = K.square(minmax_var) # when <1, this will be even smaller, which favors mse
            
            weight_inverse = tf.math.divide(1, weight) # inverse the weight for mse
            
            loss_p = tf.losses.cosine_distance(obs_norm, pred_norm, axis=0) #, weights=var, axis=0, reduction='weighted_sum'
            loss_m = tf.losses.mean_squared_error(obs[i, :], pred[i, :]) #, weights=var_inverse, reduction='weighted_sum'
            loss_per_sample = tf.math.multiply(sample_weight[i], tf.div_no_nan(weight * loss_p + weight_inverse * loss_m, weight + weight_inverse)) 
            loss = tf.math.add(loss, loss_per_sample)
            i += 1
        loss = tf.div_no_nan(loss, tf.reduce_sum(sample_weight))
        return loss
    return custom_loss_double_weighted


def custom_weighted_loss(combining_weight, axis=-1):
    def custom_loss(obs, pred):
        x_square = K.mean(pred * pred, axis=axis)#K.square(pred): same results as x * x
#        x_square = K.sum(pred * pred, axis=axis)
        y_square = K.mean(obs * obs, axis=axis)
#        y_square = K.sum(obs * obs, axis=axis)
        obs_normed = K.l2_normalize(obs - K.mean(obs, axis=axis, keepdims=True), axis=axis)
        pred_normed = K.l2_normalize(pred - K.mean(pred, axis=axis, keepdims=True), axis=axis)
        x_y_normed = K.sum(obs_normed * pred_normed, axis=axis)
        x_y = K.mean(obs * pred, axis=axis)
#        loss = combining_weight * x_square - 2 * combining_weight * x_y - (1 - combining_weight) * x_y_normed + combining_weight * y_square
        loss = combining_weight * x_square - 2 * combining_weight * x_y - (1 - 3 * combining_weight) * x_y_normed
#        loss = x_square - 10 * x_y # -2xy
#        loss = - combining_weight * x_y - (1 - combining_weight) * x_y_normed
#        loss = - K.mean((obs - K.mean(obs, axis=axis, keepdims=True)) * (pred - K.mean(pred, axis=axis, keepdims=True)), axis=axis)
        return loss
    return custom_loss


def custom_inverse_weighted_loss(batch_size, combining_weight, weight_flag): #_sample_sum_to_batch
# TODO: dynamic=True
    def custom_loss(obs, pred): 
        loss = 0.0
        i = 0
        
        while i < batch_size: 

            mean_obs = K.mean(obs[i, :], keepdims=True) 
            mean_pred = K.mean(pred[i, :], keepdims=True)
            obs_mean, pred_mean = obs[i, :] - mean_obs, pred[i, :] - mean_pred
        
            obs_norm = tf.nn.l2_normalize(obs_mean) 
            pred_norm = tf.nn.l2_normalize(pred_mean)
            
            if weight_flag == 'var':
                mean, var = tf.nn.moments(obs[i, :], axes=[0]) 
                weight = var
            if weight_flag == 'std':
                mean, var = tf.nn.moments(obs[i, :], axes=[0])
                var = tf.math.sqrt(var) 
                weight = var
            if weight_flag == 'percentile9505':
                percentile_var = percentile(obs[i, :], q=95.) - percentile(obs[i, :], q=5.)
                weight = percentile_var
            if weight_flag == 'minmax':
                minmax_var = K.max(obs[i, :]) - K.min(obs[i, :])
                weight = minmax_var
            if weight_flag == 'minmaxsquare':
                minmax_var = K.max(obs[i, :]) - K.min(obs[i, :])
                weight = K.square(minmax_var) # when <1, this will be even smaller, which favors mse
            
            weight_inverse = tf.math.divide(1, weight) # inverse the weight for mse
            
            x_square = K.mean(pred[i, :] * pred[i, :])
            x_y_normed = K.sum(obs_norm * pred_norm)
            x_y = K.mean(obs[i, :] * pred[i, :])
            
            loss_per_sample = tf.div_no_nan(weight_inverse * x_square -2 * weight_inverse * x_y - weight * x_y_normed, weight + 3 * weight_inverse) 
#            loss_per_sample = tf.div_no_nan(1 / combining_weight * weight * x_y_normed + combining_weight * weight_inverse * x_square - 2 * combining_weight * weight_inverse * x_y, 1 / combining_weight * weight + 3 * combining_weight * weight_inverse) 

            loss = tf.math.add(loss, loss_per_sample)  
            i += 1
        loss = tf.math.divide(loss, i) #.div_no_nan
        return loss
    return custom_loss

    
def poisson_loss(obs, pred):
   return tf.nn.log_poisson_loss(obs, tf.log(pred), compute_full_loss=True)


def cross_continjaccard(obs, pred):
   union = np.sum(np.maximum(np.abs(obs), np.abs(pred)))
   intersection = np.minimum(np.abs(obs), np.abs(pred))
   signs = np.sign(obs)*np.sign(pred)
   conjac = np.sum(signs*intersection)/union
   to_return = np.log(1/(0.5*np.maximum(conjac,0))-1)
   return to_return


def cross_continjaccard_loss(obs, pred):
   union = tf.math.reduce_sum(tf.math.maximum(tf.math.abs(obs), tf.math.abs(pred)), axis=-1)
   intersection = tf.math.minimum(tf.math.abs(obs),tf.math.abs(pred))
   signs = tf.math.sign(obs)*tf.math.sign(pred)
   conjac = tf.math.reduce_sum(signs*intersection, axis=-1)/union
   # Convert similarity to loss: #between [-1,1] to [0,1]
   to_return = 1-(tf.reduce_mean(conjac)+1)/2
#   epsilon = 1e-6
#   to_return = tf.math.log(1/(0.5*tf.math.maximum(conjac,0) + epsilon) - 1)
#   to_return = tf.math.log(1/(0.25*(conjac+1) + epsilon) - 1)
   return to_return


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


def kd_pearson_loss_by_batch(y_pred_teacher, temperature):    
    def custom_loss(y_true, y_pred):         
        # criterion(y_pred, y_true) + exp(-t*criterion(teacher_true, y_true))*(criterion(y_pred, teacher_true) - criterion(y_pred, y_true))
        loss = pearson_loss(y_true, y_pred) + K.exp(-temperature*pearson_loss(y_true, y_pred_teacher))*(pearson_loss(y_pred_teacher, y_pred) - pearson_loss(y_true, y_pred))
        return loss
    return custom_loss


def kd_combine_loss_by_batch(y_pred_teacher, combining_weight, temperature):    
    def custom_combine_loss(y_1, y_2):
        loss_mse = tf.keras.losses.mean_squared_error(y_1, y_2)
        loss_p = pearson_loss(y_1, y_2)
        loss = loss_p + combining_weight*loss_mse 
        return loss

    def custom_loss(y_true, y_pred):         
        # criterion(y_pred, y_true) + exp(-t*criterion(teacher_true, y_true))*(criterion(y_pred, teacher_true) - criterion(y_pred, y_true))
        return custom_combine_loss(y_true, y_pred) + K.exp(-temperature*custom_combine_loss(y_true, y_pred_teacher))*(custom_combine_loss(y_pred_teacher, y_pred) - custom_combine_loss(y_true, y_pred))

    return custom_loss


def kd_upper_bound_pearson_loss_by_batch(y_pred_teacher, alpha=0.5):
    hard_weight = alpha
    soft_weight = 1.0 - alpha
    
    def custom_loss(y_true, y_pred):        
        hard_loss = pearson_loss(y_true, y_pred)
        soft_loss = pearson_loss(y_pred_teacher, y_pred)
        teacher_loss = pearson_loss(y_true, y_pred_teacher)
        
        bound_condition = tf.greater(hard_loss, teacher_loss)
        L_teacher = tf.where(bound_condition, soft_loss, tf.zeros_like(soft_loss))

        L_final = (hard_weight * hard_loss) + (soft_weight * L_teacher)

        return L_final

    return custom_loss


def kd_upper_bound_combine_loss_by_batch(y_pred_teacher, combining_weight, alpha=0.5):
    hard_weight = alpha
    soft_weight = 1.0 - alpha

    def custom_combine_loss(y_1, y_2):
        loss_mse = tf.keras.losses.mean_squared_error(y_1, y_2)
        loss_p = pearson_loss(y_1, y_2)
        loss = loss_p + combining_weight*loss_mse 
        return loss

    def custom_loss(y_true, y_pred):        
        hard_loss = custom_combine_loss(y_true, y_pred)
        soft_loss = custom_combine_loss(y_pred_teacher, y_pred)
        teacher_loss = custom_combine_loss(y_true, y_pred_teacher)
        
        bound_condition = tf.greater(hard_loss, teacher_loss)
        L_teacher = tf.where(bound_condition, soft_loss, tf.zeros_like(soft_loss))

        L_final = (hard_weight * hard_loss) + (soft_weight * L_teacher)

        return L_final

    return custom_loss


def kd_pearson_loss(y_pred_teacher, temperature):
    def custom_loss(y_true, y_pred):         
        # criterion(y_pred, y_true) + exp(-t*criterion(teacher_true, y_true))*(criterion(y_pred, teacher_true) - criterion(y_pred, y_true))
        loss = pearson_loss_by_sample(y_true, y_pred) + tf.multiply(K.exp(-temperature*pearson_loss_by_sample(y_true, y_pred_teacher)), (pearson_loss_by_sample(y_pred_teacher, y_pred) - pearson_loss_by_sample(y_true, y_pred)))
        return K.mean(loss, axis=-1)
    return custom_loss


def kd_combine_loss(y_pred_teacher, combining_weight, temperature): 
    def custom_combine_loss(y_1, y_2):
        # loss_mse = tf.losses.mean_squared_error(y_1, y_2, reduction=tf.losses.Reduction.NONE)
        loss_mse = tf.keras.losses.mean_squared_error(y_1, y_2)
        loss_mse = tf.expand_dims(loss_mse, 1)
        loss_p = pearson_loss_by_sample(y_1, y_2)
        loss = loss_p + combining_weight*loss_mse 
        return loss

    def custom_loss(y_true, y_pred):         
        # criterion(y_pred, y_true) + exp(-t*criterion(teacher_true, y_true))*(criterion(y_pred, teacher_true) - criterion(y_pred, y_true))
        loss = custom_combine_loss(y_true, y_pred) + tf.multiply(K.exp(-temperature*custom_combine_loss(y_true, y_pred_teacher)), (custom_combine_loss(y_pred_teacher, y_pred) - custom_combine_loss(y_true, y_pred)))
        return K.mean(loss, axis=-1)

    return custom_loss


def kd_upper_bound_combine_loss(y_pred_teacher, combining_weight, alpha=0.5):
    hard_weight = alpha
    soft_weight = 1.0 - alpha

    def custom_combine_loss(y_1, y_2):
        # loss_mse = tf.losses.mean_squared_error(y_1, y_2, reduction=tf.losses.Reduction.NONE)
        loss_mse = tf.keras.losses.mean_squared_error(y_1, y_2)
        loss_mse = tf.expand_dims(loss_mse, 1)
        loss_p = pearson_loss_by_sample(y_1, y_2)
        loss = loss_p + combining_weight*loss_mse 
        return loss

    def custom_loss(y_true, y_pred):        
        hard_loss = custom_combine_loss(y_true, y_pred)
        soft_loss = custom_combine_loss(y_pred_teacher, y_pred)
        teacher_loss = custom_combine_loss(y_true, y_pred_teacher)
        # This needs to be a vector by batch
        bound_condition = tf.greater(hard_loss, teacher_loss)
        L_teacher = tf.where(bound_condition, soft_loss, tf.zeros_like(soft_loss))

        L_final = (hard_weight * hard_loss) + (soft_weight * L_teacher)

        return K.mean(L_final, axis=-1)

    return custom_loss

# Attentive imitation loss for knowledge distillation 
def attentive_imitation_pearson_loss(y_pred_teacher, alpha=0.5, eta=1.0):
    # # eta is a normalization parameter that can be retrieved from
    # # subtracting the maximum and the minimum of teacher loss on the training set
    hard_weight = alpha
    soft_weight = 1.0 - alpha
    
    def custom_loss(y_true, y_pred):
        # This needs to be a vector for a batch        
        hard_loss = pearson_loss_by_sample(y_true, y_pred)        
        soft_loss = pearson_loss_by_sample(y_pred_teacher, y_pred) 
        teacher_loss = pearson_loss_by_sample(y_true, y_pred_teacher)

        phi = tf.subtract(tf.ones_like(teacher_loss), tf.multiply(teacher_loss, 1.0 / eta))
        # phi is a vector for a batch
        L_final = (hard_weight * hard_loss) + (soft_weight * tf.multiply(phi, soft_loss))
        L_final = K.mean(L_final, axis=-1)
        return L_final

    return custom_loss


def attentive_imitation_combine_loss(y_pred_teacher, combining_weight, alpha=0.5, eta=1.0):
    # # eta is a normalization parameter that can be retrieved from
    # # subtracting the maximum and the minimum of teacher loss on the training set
    hard_weight = alpha
    soft_weight = 1.0 - alpha

    def custom_combine_loss(y_1, y_2):
        # loss_mse = tf.losses.mean_squared_error(y_1, y_2, reduction=tf.losses.Reduction.NONE)
        loss_mse = tf.keras.losses.mean_squared_error(y_1, y_2)
        loss_mse = tf.expand_dims(loss_mse, 1)
        loss_p = pearson_loss_by_sample(y_1, y_2)
        loss = loss_p + combining_weight*loss_mse 
        return loss
    
    def custom_loss(y_true, y_pred):
        # This needs to be a vector for a batch        
        hard_loss = custom_combine_loss(y_true, y_pred)        
        soft_loss = custom_combine_loss(y_pred_teacher, y_pred) 
        teacher_loss = custom_combine_loss(y_true, y_pred_teacher)

        phi = tf.subtract(tf.ones_like(teacher_loss), tf.multiply(teacher_loss, 1.0 / eta))
        # phi is a vector for a batch
        L_final = (hard_weight * hard_loss) + (soft_weight * tf.multiply(phi, soft_loss))
        L_final = K.mean(L_final, axis=-1)
        return L_final

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


# This loss is not working for now
def mse_multiply_per_class(obs, pred):
    loss_per_class_per_sample = tf.losses.mean_squared_error(obs, pred, reduction=tf.losses.Reduction.NONE)
    # This multiply func returns Nan - because the mul returns inf: too many terms; 18 term works
    # TODO: take average within a lineage, then use multiple between lineage?
    loss_per_sample = tf.reduce_prod(loss_per_class_per_sample, axis=-1) #[:, :19] 
    loss = K.mean(loss_per_sample, axis=-1)                          
    return loss


# # This loss is not working for now
# def mse_per_class(batch_size, num_classes):
#     def custom_loss(obs, pred):
#         loss_batch = 0.0
#         for i in range(batch_size):
#             loss = 1.0
#             for j in range(num_classes):
#                 # loss = tf.add(loss, tf.losses.mean_squared_error(obs[i, j], pred[i, j]))  
#                 # This multiply func returns Nan - because the mul returns inf: too many terms
#                 loss = tf.multiply(loss, tf.losses.mean_squared_error(obs[i, j], pred[i, j]))                 
#             loss_batch = tf.add(loss_batch, loss)
#         return tf.divide(loss_batch, batch_size*num_classes)
#     return custom_loss


# This loss is not working for now
#def spearman_rankcorr_loss(batch_size):
#    def custom_loss(obs, pred): 
#        loss =  -1 * batch_size * tf.reduce_sum(tf.multiply(obs, pred)) - (tf.reduce_sum(obs) * tf.reduce_sum(pred))
#        divisor = tf.sqrt(
#                (batch_size * tf.reduce_sum(tf.square(obs)) - tf.square(tf.reduce_sum(obs))) * 
#                (batch_size * tf.reduce_sum(tf.square(pred)) - tf.square(tf.reduce_sum(pred)))
#                )
#        loss = tf.truediv(loss, divisor) 
#        return loss
#    return custom_loss
    

#def spearman_rankcorr_loss(batch_size, num_classes):
#    def custom_loss(obs, pred):
#        predictions_rank = tf.nn.top_k(pred, k=num_classes, sorted=True, name='prediction_rank').indices
#        real_rank = tf.nn.top_k(obs, k=num_classes, sorted=True, name='real_rank').indices
#        rank_diffs = predictions_rank - real_rank
#        rank_diffs_squared_sum = tf.reduce_sum(rank_diffs * rank_diffs)
#        six = tf.constant(6)
#        one = tf.constant(1.0)
#        numerator = tf.cast(six * rank_diffs_squared_sum, dtype=tf.float32)
#        divider = tf.cast(num_classes * num_classes * num_classes - num_classes, dtype=tf.float32)
#        loss_batch = one - numerator / divider
#        return loss_batch
#    return custom_loss


#def spearman_rankcorr_loss(batch_size):
#    def custom_loss(obs, pred):
#        loss = 0.0
#        from scipy.stats import spearmanr
#        for i in range(batch_size):
#            loss_temp = (tf.py_function(spearmanr, [pred[i, :], obs[i, :]], Tout = tf.float32))
#            loss = tf.add(loss, loss_temp)
#        loss = tf.div_no_nan(loss, batch_size)
#        return loss
#    return custom_loss
    

#def spearman_rankcorr_loss(obs, pred):
#    from scipy.stats import spearmanr
#    return (tf.py_function(spearmanr, [pred, obs], Tout = tf.float32))