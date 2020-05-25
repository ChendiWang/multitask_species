from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import numpy as np

import matplotlib
from matplotlib.colors import LogNorm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error

import utils

    
def plot_training_valid_loss(training_loss, valid_loss, loss1_txt, loss2_txt, output_directory):
    #plot line plot of loss function per epoch
    plt.figure(figsize=(16, 9))
    plt.plot(np.arange(len(training_loss)), training_loss)
    plt.plot(valid_loss)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend([loss1_txt, loss2_txt], fontsize=18)
    plt.savefig(output_directory + loss1_txt + '_' + loss2_txt + '_loss.svg')
    plt.close()


def plot_cors(obs, pred, output_directory, title='basset_cor_hist.svg'):
    correlations = []
    vars = []
    for i in range(len(pred)):
        var = np.var(obs[i, :])
        vars.append(var)
        x = np.corrcoef(pred[i, :], obs[i, :])[0, 1]
        correlations.append(x)

    weighted_cor = np.dot(correlations, vars) / np.sum(vars)
    print('weighted_cor is {}'.format(weighted_cor))

    nan_cors = [value for value in correlations if math.isnan(value)]
    print("number of NaN values: %d" % len(nan_cors))
    correlations = [value for value in correlations if not math.isnan(value)]

 
    plt.clf()
    plt.hist(correlations, bins=30)
    plt.axvline(np.mean(correlations), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(0, color='k', linestyle='solid', linewidth=2)
    try:
        plt.title("histogram of correlation.  Avg cor = {%f}" % np.mean(correlations))
    except Exception as e:
        print("could not set the title for graph")
        print(e)
    plt.ylabel("Frequency")
    plt.xlabel("correlation")
    plt.savefig(output_directory + title)
    plt.close()

    return correlations


def plot_mses(obs, pred, output_directory, title='basset_mse_hist.svg'):
    mses = []
    var_all = []
    for i in range(len(pred)):
        var = np.var(obs[i, :])
        var_all.append(var)
        x = mean_squared_error(obs[i, :], pred[i, :])
        mses.append(x)

    weighted_mse = np.dot(mses, var_all) / np.sum(var_all)
    print('weighted_mse is {}'.format(weighted_mse))

    nan_mses = [value for value in mses if math.isnan(value)]
    print("number of NaN values in MSE: %d" % len(nan_mses))
    mses = [value for value in mses if not math.isnan(value)]
 
    plt.clf()
    plt.hist(mses, bins=30)
    plt.axvline(np.mean(mses), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(0, color='k', linestyle='solid', linewidth=2)
    try:
        plt.title("histogram of mse.  Avg mse = {%f}" % np.mean(mses))
    except Exception as e:
        print("could not set the title for graph")
        print(e)
    plt.ylabel("Frequency")
    plt.xlabel("MSE")
    plt.savefig(output_directory + title)
    plt.close()

    return mses


def plot_cors_piechart(correlations, eval_labels, output_directory, title=None):
    ind_collection = []
    Q0_idx = []
    Q1_idx = []
    Q2_idx = []
    Q3_idx = []
    Q4_idx = []
    Q5_idx = []
    ind_collection.append(Q0_idx)
    ind_collection.append(Q1_idx)
    ind_collection.append(Q2_idx)
    ind_collection.append(Q3_idx)
    ind_collection.append(Q4_idx)
    ind_collection.append(Q5_idx)
    
    for i, x in enumerate(correlations):
        if x > 0.75:
            Q1_idx.append(i)
            if x > 0.9:
                Q0_idx.append(i)
        elif x > 0.5 and x <= 0.75:
            Q2_idx.append(i)
        elif x > 0.25 and x <= 0.5:
            Q3_idx.append(i)
        elif x > 0 and x <= 0.25:
            Q4_idx.append(i)
        elif x < 0:
            Q5_idx.append(i)
        
    # pie chart of correlations distribution
    pie_labels = "cor>0.75", "0.5<cor<0.75", "0.25<cor<0.5", "0<cor<0.25", 'cor<0'
    sizes = [len(Q1_idx), len(Q2_idx), len(Q3_idx), len(Q4_idx), len(Q5_idx)]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'red']
    explode = (0.1, 0, 0, 0, 0)  # explode 1st slice
    plt.pie(sizes, explode=explode, labels=pie_labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('correlation_pie')
    if not title:
        title = "basset_cor_pie.svg"
    plt.savefig(output_directory + title)
    plt.close()
    
    # Plot relation between SD/IQR vs prediction performance
#    Q0 = eval_labels[Q0_idx]
    Q1 = eval_labels[Q1_idx]
    Q2 = eval_labels[Q2_idx]
    Q3 = eval_labels[Q3_idx]
    Q4 = eval_labels[Q4_idx]
    Q5 = eval_labels[Q5_idx]
    
    sd1 = np.std(Q1, axis=1)
    sd2 = np.std(Q2, axis=1)
    sd3 = np.std(Q3, axis=1)
    sd4 = np.std(Q4, axis=1)
    sd5 = np.std(Q5, axis=1)
    
    qr1 = np.percentile(Q1, 75, axis=1) - np.percentile(Q1, 25, axis=1)
    qr2 = np.percentile(Q2, 75, axis=1) - np.percentile(Q2, 25, axis=1)
    qr3 = np.percentile(Q3, 75, axis=1) - np.percentile(Q3, 25, axis=1)
    qr4 = np.percentile(Q4, 75, axis=1) - np.percentile(Q4, 25, axis=1)
    qr5 = np.percentile(Q5, 75, axis=1) - np.percentile(Q5, 25, axis=1)
    
    mean_sds = []
    mean_sd1 = np.mean(sd1)
    mean_sd2 = np.mean(sd2)
    mean_sd3 = np.mean(sd3)
    mean_sd4 = np.mean(sd4)
    mean_sd5 = np.mean(sd5)
    mean_sds.append(mean_sd1)
    mean_sds.append(mean_sd2)
    mean_sds.append(mean_sd3)
    mean_sds.append(mean_sd4)
    mean_sds.append(mean_sd5)
    print('1st sd: {0}, 2nd sd: {1}, 3rd sd: {2}, 4th sd: {3}'.format(mean_sd1, mean_sd2, mean_sd3, mean_sd4))
    
    mean_qrs = []
    mean_qr1 = np.mean(qr1)
    mean_qr2 = np.mean(qr2)
    mean_qr3 = np.mean(qr3)
    mean_qr4 = np.mean(qr4)
    mean_qr5 = np.mean(qr5)
    mean_qrs.append(mean_qr1)
    mean_qrs.append(mean_qr2)
    mean_qrs.append(mean_qr3)
    mean_qrs.append(mean_qr4)
    mean_qrs.append(mean_qr5)
    print('1st qr: {0}, 2nd qr: {1}, 3rd qr: {2}, 4th qr: {3}'.format(mean_qr1, mean_qr2, mean_qr3, mean_qr4))
    
    x_axis = np.arange(5)
    width = 0.3
    xticks = ["cor>0.75", "0.5<cor<0.75", "0.25<cor<0.5", "0<cor<0.25", 'cor<0']
    plt.figure(figsize=(16, 9))
    plt.bar(x_axis, mean_sds, width, color='#fc8d91', edgecolor='none', label='standard deviation')
    plt.bar(x_axis + width, mean_qrs, width, color='#f7d00e', edgecolor='none', label='interquartile range')
    plt.xticks(x_axis + width, xticks, fontsize=16)
    plt.title('Comparison among good and bad peaks')
    plt.xlabel('peaks class', fontsize=18)
    plt.ylabel('average', fontsize=18)
    plt.legend()
    plt.savefig(output_directory + "basset_SD_IQR.svg")
    plt.close()
    
    return ind_collection


def plot_corr_variance(labels, correlations, output_directory):
    #compute variance:
    variance = np.var(labels, axis=1)

    #plot scatterplot of variance-correlations
    plt.figure(figsize=(16, 9))
    plt.scatter(variance, correlations)
    plt.xlabel('Peak variance', fontsize=18)
    plt.ylabel('Prediction-ground truth correlation', fontsize=18)
    plt.savefig(output_directory + "variance_correlation_plot.svg")
    plt.close()

    #plot 2D scatterplot of variance-correlations
    plt.figure(figsize=(16, 9))
    plt.hist2d(variance, correlations, bins=100)
    plt.xlabel('Peak variance', fontsize=18)
    plt.ylabel('Prediction-ground truth correlation', fontsize=18)
    plt.colorbar()
    plt.savefig(output_directory + "variance_correlation_hist2D.svg")
    plt.close()

    #plot 2D log transformed scatterplot of variance-correlations
    plt.figure(figsize=(16, 9))
    plt.hist2d(variance, correlations, bins=100, norm=LogNorm())
    plt.xlabel('Peak variance', fontsize=18)
    plt.ylabel('Prediction-ground truth correlation', fontsize=18)
    plt.colorbar()
    plt.savefig(output_directory + "variance_correlation_loghist2d.svg")
    plt.close()


#get cell-wise correlations
def plot_cell_cors(obs, pred, cell_labels, output_directory,num_classes):
    correlations = []
    for i in range(pred.shape[1]):
        x = np.corrcoef(pred[:, i], obs[:, i])[0, 1]
        correlations.append(x)

    #plot cell-wise correlation histogram
    plt.clf()
    plt.hist(correlations, bins=30)
    plt.axvline(np.mean(correlations), color='k', linestyle='dashed', linewidth=2)
    try:
        plt.title("histogram of correlation.  Avg cor = {0:.2f}".format(np.mean(correlations)))
    except Exception as e:
        print("could not set the title for graph")
        print(e)
    plt.ylabel("Frequency")
    plt.xlabel("Correlation")
    plt.savefig(output_directory + "basset_cell_wise_cor_hist.svg")
    plt.close()
    
    #plot cell-wise correlations by cell type
    plt.clf()
    plt.bar(np.arange(num_classes), correlations)
    plt.title("Correlations by Cell Type")
    plt.ylabel("Correlation")
    plt.xlabel("Cell Type")
    plt.xticks(np.arange(num_classes), cell_labels, rotation='vertical', fontsize=3.5) 
    plt.savefig(output_directory + "cellwise_cor_bargraph.svg")
    plt.close()
    
    return correlations


# plot some predictions vs ground_truth on test set
def plot_random_predictions(eval_labels, predictions, correlations, ind_collection, eval_names, output_directory, num_classes, cell_names, title=None, scale=True):
    for n in range(3):
        mum_plt_row = 1
        mum_plt_col = 1
        num_plt = mum_plt_row * mum_plt_col
        # 3 plots for each correlation categories
        for k in range(len(ind_collection) - 1):
            if len(ind_collection[k + 1]) < num_plt: continue
            idx = random.sample(ind_collection[k + 1], num_plt)
    
            y_samples_eval = eval_labels[idx]
            predicted_classes = predictions[idx]
            sample_names = eval_names[idx]
    
            # Plot
            x_axis = np.arange(num_classes)
            if cell_names==[]:
                xticks = []
            else:
                xticks = cell_names
                                   
            plt.figure(1)
            width = 0.35
            for i in range(num_plt):
                plt.figure(figsize=(16, 9))
                plt.subplot(mum_plt_row, mum_plt_col, i + 1)
                plt.bar(x_axis, y_samples_eval[i], width, color='#f99fa1', edgecolor='none', label='true activity')
                if scale:
                    plt.bar(x_axis + width, utils.minmax_scale(predicted_classes[i], y_samples_eval[i]), width, color='#014ead',
                            edgecolor='none', label='prediction')
                    scale_txt = '_normalized'
                else:
                    plt.bar(x_axis + width, predicted_classes[i], width, color='#014ead',
                            edgecolor='none', label='prediction')   
                    scale_txt = '_original'
                plt.xticks(x_axis + width, xticks, rotation='vertical', fontsize=9)
                plt.title('{0}, correlation = {1:.3f}'.format(sample_names[i], correlations[idx[i]]))
                plt.xlabel('cell type', fontsize=12)
                plt.ylabel(scale_txt + ' activity', fontsize=12)
                plt.legend()
    
    
            fig = plt.gcf()
            fig.tight_layout()
            if not title:
                title = ''
            plt.savefig(output_directory + title + "basset_cor_q{0}{1}".format(k, n + 1) + scale_txt + ".svg", bbox_inches='tight')
    
            plt.close()
            
            
# Plot individual prediction cases
def plot_predictions(eval_labels, predictions, norm_flag, correlations, mses, eval_names, output_directory, num_classes, cell_names, file_txt):
    for idx in range(len(eval_labels)):   
        y_samples_eval = eval_labels[idx]
        predicted_classes = predictions[idx]
        sample_names = eval_names[idx]

        x_axis = np.arange(num_classes)
        xticks = cell_names
            
        plt.figure()
        width = 0.35

        plt.figure(figsize=(16, 9))
#        plt.subplot(mum_plt_row, mum_plt_col, i + 1)
        plt.bar(x_axis, y_samples_eval, width, color='#f99fa1', edgecolor='none', label='true activity')
        if norm_flag:
            plt.bar(x_axis + width, utils.minmax_scale(predicted_classes, y_samples_eval), width, color='#014ead',
                    edgecolor='none', label='prediction')
            ylabel_text = 'Normalized activity'
        else:
            plt.bar(x_axis + width, predicted_classes, width, color='#014ead',
                edgecolor='none', label='prediction') 
            ylabel_text = 'Activity'
        plt.xticks(x_axis + width, xticks, rotation='vertical', fontsize=9)
        plt.title('{0}, correlation = {1:.4f}, mse = {2:.4f}'.format(sample_names, correlations[idx], mses[idx]))
        plt.xlabel('Cell type', fontsize=12)
        plt.ylabel(ylabel_text, fontsize=12)
        plt.legend()

        fig = plt.gcf()
        fig.tight_layout()
        plt.savefig(output_directory + sample_names + file_txt + '.svg', bbox_inches='tight')
        plt.close()
        
        
def plot_predictions_subplot(eval_labels, predictions, correlations, mses, eval_names, output_directory, num_classes, cell_names, file_txt):
    for idx in range(len(eval_labels)):   
        y_samples_eval = eval_labels[idx]
        predicted_classes = predictions[idx]
        sample_names = eval_names[idx]

        # Plot
        x_axis = np.arange(num_classes)
        xticks = cell_names

        plt.figure()
        width = 0.5

        fig, axs = plt.subplots(2, figsize=(24, 12))
#        fig.suptitle('{0}, correlation = {1:.3f}'.format(sample_names, correlations[idx]), fontsize=18)
        line_labels = ['true activity', 'prediction']
        l0 = axs[0].bar(x_axis, y_samples_eval, width, color='#f99fa1') #, edgecolor='none', label='true activity'
        l1 = axs[1].bar(x_axis, utils.minmax_scale(predicted_classes, y_samples_eval), width, color='#014ead') #, edgecolor='none', label='prediction'
        axs[0].set_title('{0}, correlation = {1:.4f}, mse = {2:.4f}'.format(sample_names, correlations[idx], mses[idx]), fontsize=30)
        
        fig.legend([l0, l1],     # The line objects
           labels=line_labels,   # The labels for each line
           loc="upper right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           fontsize=12  # Title for the legend
           )

        plt.xticks(x_axis, xticks, rotation='vertical', fontsize=15)
        plt.xlabel('Cell type', fontsize=24)
        plt.ylabel('Normalized activity', fontsize=24)

        fig = plt.gcf()
        fig.tight_layout()
        plt.savefig(output_directory + sample_names + file_txt + '_subplot.svg', bbox_inches='tight')
        plt.close()
        
            
def plot_filt_corr_change(filt_pred, labels, correlations, output_directory, square_flag=True):  
    filt_corr = []
    corr_change = []
    for i in range(len(filt_pred)):
        pred = filt_pred[i,:,:]
        label = labels[i]
        corr_original = np.full(filt_pred.shape[1], correlations[i])

        #compute correlation between label and each prediction
        def pearson_corr(pred, label):
            return np.corrcoef(pred, label)[0,1]
        corr = np.apply_along_axis(pearson_corr, 1, pred, label)
        filt_corr.append(corr) 
        
        #compute difference in correlation between original model and leave-one-filter-out results
        if square_flag:
            change = np.square(corr-corr_original)
        else:
            change = corr_original-corr
        corr_change.append(change)
        
    #convert filt_corr and corr_change from list to array
    filt_corr = np.stack(filt_corr, axis=0)
    corr_change = np.stack(corr_change, axis=0)
    
    # plot histogram of correlation values of all models
    plt.clf()
    plt.hist(filt_corr.flatten(), bins=30)
    plt.axvline(np.mean(filt_corr), color='k', linestyle='dashed', linewidth=2)
    try:
        plt.title("histogram of correlation.  Avg cor = {0:.2f}".format(np.mean(filt_corr)))
    except Exception as e:
        print("could not set the title for graph")
        print(e)
    plt.ylabel("Frequency")
    plt.xlabel("Correlation")
    plt.savefig(output_directory + "filt_corr_hist.svg")
    plt.close()
    
    # corr_change = np.sum(corr_change, axis=0)
    # # Change to Average by the size of samples 
    # # Keep both versions
    corr_change_mean = np.mean(corr_change, axis=0) 
    # # Change to number of nonzero samples
    corr_change_mean_act = np.ma.masked_equal(corr_change, 0)
    corr_change_mean_act = np.mean(corr_change_mean_act, axis=0)
    corr_change_mean_act = corr_change_mean_act.filled(0.0)

    # # Plot the distribution of correlation change
    plt.clf()
    plt.hist(corr_change_mean.flatten(), bins=30)
    plt.axvline(np.mean(corr_change_mean), color='k', linestyle='dashed', linewidth=2)
    try:
        plt.title("histogram of correlation.  Avg cor = {0:.2f}".format(np.mean(corr_change_mean)))
    except Exception as e:
        print("could not set the title for graph")
        print(e)
    plt.ylabel("Frequency")
    plt.xlabel("Correlation")
    plt.savefig(output_directory + "corr_change_hist.svg")
    plt.close()
    
    # # Plot bar graph of correlation change
    plt.clf()
    plt.bar(np.arange(filt_pred.shape[1]), corr_change_mean)
    plt.title("Influence of filters on model predictions")
    plt.ylabel("Influence")
    plt.xlabel("Filter")
    plt.savefig(output_directory + "corr_change_bar_graph.svg")
    plt.close()
    
    return filt_corr, corr_change, corr_change_mean, corr_change_mean_act


def infl_celltype_by_activation(infl):
    infl_sum_celltype = np.sum(np.absolute(infl), axis=-1) # If there's no change in all celltype, mask out this OCR
    infl_sum_celltype = infl_sum_celltype > 0 # Turn into a binary mask if there is change
    infl_sum_celltype = np.expand_dims(infl_sum_celltype, -1)
    infl_sum_celltype = np.repeat(infl_sum_celltype, infl.shape[-1], axis=-1)
    masked_infl = np.ma.masked_equal(infl_sum_celltype, 0)
    del infl_sum_celltype
    infl_mean_act = np.multiply(infl, masked_infl)
    del masked_infl, infl
    infl_mean_act = np.mean(infl_mean_act, axis=0).squeeze()
    infl_mean_act = infl_mean_act.filled(0.0)
    return infl_mean_act


def plot_filt_infl(pred, filt_pred, output_directory, cell_labels=None):
    # Expand pred array to be nx300x81
    pred = np.expand_dims(pred, 1)
    pred = np.repeat(pred, filt_pred.shape[1], axis=1)
    
    # influence per celltype with signed version; result 300x81 array of influences
    infl = pred - filt_pred  
    # Average by the size of samples
    infl_signed_mean = np.mean(infl, axis=0).squeeze()   
    # Change to number of nonzero samples
    infl_signed_mean_act = infl_celltype_by_activation(infl)    

    # Compute the sum of absolute/squares of differences between pred and filt_pred; result 300x81 array of influences
    # infl = np.square(filt_pred - pred) 
    infl_absolute_mean = np.mean(np.absolute(infl), axis=0).squeeze() 
    infl_absolute_mean_act = infl_celltype_by_activation(np.absolute(infl))
    
    # plot histogram
    plt.clf()
    plt.hist(infl_absolute_mean.flatten(), bins=30)
    plt.axvline(np.mean(infl_absolute_mean), color='k', linestyle='dashed', linewidth=2)
    try:
        plt.title("histogram of filter influence.  Avg influence = {0:.2f}".format(np.mean(infl_absolute_mean)))
    except Exception as e:
        print("could not set the title for graph")
        print(e)
    plt.ylabel("Frequency")
    plt.xlabel("")
    plt.savefig(output_directory + "filt_infl_celltype_hist.svg")
    plt.close()

    # plot heatmap
    plt.clf()
    infl_absolute_mean[infl_absolute_mean==0] = np.nan    
    sns.heatmap(np.log10(infl_absolute_mean), mask=np.isnan(infl_absolute_mean))
    plt.title("log10 of Influence of filters on each cell type prediction")
    plt.ylabel("Filter")
    plt.xlabel("Cell Type")
    if not cell_labels:
        cell_labels = []
    plt.xticks(np.arange(len(cell_labels)), cell_labels, rotation='vertical', fontsize=3.0)
    plt.savefig(output_directory + "sns_filt_infl_celltype_log10_heatmap.svg")
    plt.close()

    return infl, infl_signed_mean, infl_signed_mean_act, infl_absolute_mean, infl_absolute_mean_act

  
def plot_confusion_matrix(obs, preds, output_directory, num_classes):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    import itertools
    
    test_labels_max = obs
    predicted_labels_max = np.argmax(preds, axis=1)
    #print('max test labels:\n',test_labels_max.T)
    #print('max predicted labels:\n',predicted_labels_max.T)
    
    accuracy = accuracy_score(test_labels_max, predicted_labels_max) # , normalize=False
    print('Test set accuracy:\n',accuracy)
    
    target_names = ['class 0', 'class 1']
    print(classification_report(test_labels_max, predicted_labels_max, target_names=target_names))
    # TODO: Add in sample weights
    
    cm = confusion_matrix(test_labels_max, predicted_labels_max)
    print('Confusion matrix:\n',cm)
    
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    
    plt.clf()
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.xticks([0, 1]); plt.yticks([0, 1])
    plt.grid(False)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > 0.5 else 'black')
    plt.savefig(output_directory + "confusion_matrix.svg")
    
    
def plot_scatter(x, y, output_directory, title, xlabel, ylabel, save_title, same_limit=False, c=None):
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 9))
    if same_limit:
        min_limit = np.minimum(np.min(x), np.min(y)) - 0.01
        max_limit = np.maximum(np.max(x), np.max(y)) + 0.01
        plt.xlim(min_limit, max_limit)
        plt.ylim(min_limit, max_limit)
    else:
        plt.xlim(np.min(x)-0.01, np.max(x)+0.01)
        plt.ylim(np.min(y)-0.01, np.max(y)+0.01)
    if c is not None:
        ax.scatter(x, y, cmap=plt.get_cmap('jet'), c=c)
    else:
        ax.scatter(x, y, cmap=plt.get_cmap('jet'))
    plt.title(title, fontsize=18) 
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18) 
    plt.savefig(output_directory + save_title)
    plt.close()
        

def get_memes(activations, sequences, y, output_directory, num_classes, flag_coding_order, threshold=0.5, flag_weighted=False, num_filter=300):
    #find the threshold value for activation
    if activations.shape[-1] != 251:
        activations = np.swapaxes(activations, 1, 2)
    if sequences.shape[-1] != 251:
        sequences = np.swapaxes(sequences, 1, 2)
        
    if flag_weighted:
        activation_threshold = threshold*np.amax(activations, axis=(2))
    else:
        activation_threshold = threshold*np.amax(activations, axis=(0, 2))          

    #pad sequences:
    num_pad = 9
    npad = ((0, 0), (0, 0), (num_pad, num_pad))
    sequences = np.pad(sequences, pad_width=npad, mode='constant', constant_values=0)
    
    pwm = np.zeros((num_filter, 4, 19))
    pfm = np.zeros((num_filter, 4, 19))
    nsamples = activations.shape[0]
    
    OCR_matrix = np.zeros((num_filter, y.shape[0]))
    activation_indices = []
    activated_OCRs = np.zeros((num_filter, num_classes))
    n_activated_OCRs = np.zeros(num_filter)
    total_seq = np.zeros(num_filter)
    filter_to_ind_dict = {}
    
    for i in range(num_filter):
        #create list to store 19 bp sequences that activated filter
        act_seqs_list = []
        act_OCRs_tmp = []
        list_seq_and_start=[]
        for j in range(nsamples):
            # find all indices where filter is activated
            if flag_weighted:
                indices = np.where(activations[j,i,:] > activation_threshold[j,i])
            else:
                indices = np.where(activations[j,i,:] > activation_threshold[i])

            #save ground truth peak heights of OCRs activated by each filter  
            if indices[0].shape[0]>0:
                act_OCRs_tmp.append(y[j, :])
                OCR_matrix[i, j] = 1 
                
            for start in indices[0]:
                activation_indices.append(start)
                end = start+19
                act_seqs_list.append(sequences[j,:,start:end]) 
                list_seq_and_start.append((j, start-num_pad))
                
        filter_to_ind_dict[i]=list_seq_and_start
        
        #convert act_seqs from list to array
        if act_seqs_list:
          act_seqs = np.stack(act_seqs_list)
          pwm_tmp = np.sum(act_seqs, axis=0)
          pfm_tmp=pwm_tmp
          total = np.sum(pwm_tmp, axis=0)
          # Avoid divide by zero runtime error
          # pwm_tmp = np.nan_to_num(pwm_tmp/total) # Original way will raise runtime error         
          with np.errstate(divide='ignore', invalid='ignore'):
              pwm_tmp = np.true_divide(pwm_tmp,total) # When certain position of total is 0, ignore error
              pwm_tmp[pwm_tmp == np.inf] = 0
              pwm_tmp = np.nan_to_num(pwm_tmp)
          
          #permute pwm from A, T, G, C order to A, C, G, T order
          if flag_coding_order == 'ATGC':
              order = [0, 3, 2, 1]
          if flag_coding_order == 'ACGT':
              order = [0, 1, 2, 3]
          pwm[i,:,:] = pwm_tmp[order, :]
          pfm[i,:,:] = pfm_tmp[order, :]
          
          #store total number of sequences that activated that filter
          total_seq[i] = len(act_seqs_list)

          #save mean OCR activation
          act_OCRs_tmp = np.stack(act_OCRs_tmp)
          activated_OCRs[i, :] = np.mean(act_OCRs_tmp, axis=0)

          #save the number of activated OCRs
          n_activated_OCRs[i] = act_OCRs_tmp.shape[0]
    
    #TODO: delete the following line: activated_OCRs is already an array
    activated_OCRs = np.stack(activated_OCRs)

    #write motifs to meme format
    #PWM file:
    if flag_weighted:
        meme_file = open(output_directory + "filter_motifs_pwm_weighted.meme", 'w')
    else:
        meme_file = open(output_directory + "filter_motifs_pwm.meme", 'w')
    meme_file.write("MEME version 4 \n")

    #PFM file:
    if flag_weighted:
        meme_file_pfm = open(output_directory + "filter_motifs_pfm_weighted.meme", 'w')
    else:
        meme_file_pfm = open(output_directory + "filter_motifs_pfm.meme", 'w')
    meme_file_pfm.write("MEME version 4 \n")

    for i in range(0, num_filter):
        if np.sum(pwm[i,:,:]) > 0:
          meme_file.write("\n")
          meme_file.write("MOTIF filter%s \n" % i)
          meme_file.write("letter-probability matrix: alength= 4 w= %d \n" % np.count_nonzero(np.sum(pwm[i,:,:], axis=0)))

          meme_file_pfm.write("\n")
          meme_file_pfm.write("MOTIF filter%s \n" % i)
          meme_file_pfm.write("letter-probability matrix: alength= 4 w= %d \n" % np.count_nonzero(np.sum(pwm[i,:,:], axis=0)))

          for j in range(0, 19):
              if np.sum(pwm[i,:,j]) > 0:
                meme_file.write(str(pwm[i,0,j]) + "\t" + str(pwm[i,1,j]) + "\t" + str(pwm[i,2,j]) + "\t" + str(pwm[i,3,j]) + "\n")
                meme_file_pfm.write(str(pfm[i,0,j]) + "\t" + str(pfm[i,1,j]) + "\t" + str(pfm[i,2,j]) + "\t" + str(pfm[i,3,j]) + "\n")
      
    meme_file.close()
    meme_file_pfm.close()
    
    #plot indices of first position in sequence that activates the filters
    activation_indices_array = np.stack(activation_indices)
    
    plt.clf()
    plt.hist(activation_indices_array.flatten(), bins=260)
    plt.title("histogram of position indices.")
    plt.ylabel("Frequency")
    plt.xlabel("Position")
    plt.savefig(output_directory + "position_hist.svg")
    plt.close()
    
    #plot total sequences that activated each filter
    #TODO: delete the following line: total_seq is already an array
    total_seq_array = np.stack(total_seq)
    
    plt.clf()
    plt.bar(np.arange(num_filter), total_seq_array)
    plt.title("Number of sequences activating each filter")
    plt.ylabel("N sequences")
    plt.xlabel("Filter")
    plt.savefig(output_directory + "nseqs_bar_graph.svg")
    plt.close()
    
    return filter_to_ind_dict, pwm, activation_indices_array, total_seq_array, activated_OCRs, n_activated_OCRs, OCR_matrix


#convert mouse model predictions to human cell predictions
def mouse2human(mouse_predictions, mouse_cell_types, mapping, method='average'):
    
    human_cells = np.unique(mapping[:,1])
    
    human_predictions = np.zeros((mouse_predictions.shape[0], human_cells.shape[0]))
    
    for i, celltype in enumerate(human_cells):
        matches = mapping[np.where(mapping[:,1] == celltype)][:,0]
        idx = np.in1d(mouse_cell_types, matches).nonzero() #mouse_cell_types[:,1] Edited by Chendi
        if method == 'average':
            human_predictions[:, i] = np.mean(mouse_predictions[:, idx], axis=2).squeeze()
        if method == 'max':
            human_predictions[:, i] = np.max(mouse_predictions[:, idx], axis=2).squeeze()
        if method == 'median':
            human_predictions[:, i] = np.median(mouse_predictions[:, idx], axis=2).squeeze()
   
    return human_predictions, human_cells    
