#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:00:50 2019

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


folder = '../results/motifs/keras_version/frozen'
species = 'Human'
data_aug_txt = ''#'_rc_shiftleft_shiftright'
cisbp_txt = '_otherspecies'
flag_combined_species = True#False#
branch_tag = '_hard'
influence_version_txt = '_mean'# '_mean_activated'#

if flag_combined_species:
    species_text = species + '/'
    if branch_tag == '_hard':
        model_name = 'model_multitask_MaH_ratiobased_Basset_adam_lr0001_dropout_03_conv_relu_max_BN_convfc_batch100_loss_combine001_00052_bysample_epoch10_best_separatelayer_hard_branched' + data_aug_txt
    if branch_tag == '_soft':
        model_name = 'model_multitask_MaH_ratiobased_Basset_adam_lr0001_dropout_03_conv_relu_max_BN_convfc_batch100_loss_combine001_00052_bysample_epoch10_best_separatelayer_soft_branched_cross_stitch' + data_aug_txt
else:
    species_text = ''
    if species == 'Mouse':
        model_name = 'model_' + species + '_Basset_adam_lr0001_dropout_03_conv_relu_max_BN_convfc_batch_100_loss_combine001_bysample_epoch10_best_separatelayer' + data_aug_txt
    if species == 'Human':
        model_name = 'model_' + species + '_Basset_adam_lr0001_dropout_03_conv_relu_max_BN_convfc_batch_100_loss_combine00052_bysample_epoch10_best_separatelayer' + data_aug_txt

output_directory = os.path.join(folder, model_name) + '/well_predicted/' + species_text

# Read in filter influence
filter_influence = np.load(output_directory + 'filter_influence' + influence_version_txt + '.npy')
filter_num = []
for i in range (0, len(filter_influence)):
    filter_num.append('filter'+str(i))
df_influence = pd.DataFrame({'Filter': filter_num, 'Influence': filter_influence})

if not os.path.isfile(output_directory + 'num_runs_tomtom.npy'):
        df_influence['num_runs_tomtom'] = 'default value'
else:
    num_runs_tomtom = np.load(output_directory + 'num_runs_tomtom.npy')
    df_influence['num_runs_tomtom'] = num_runs_tomtom

# Distribution of reproducibility
if os.path.isfile(output_directory + 'num_runs_tomtom.npy'):
    num_max = 20
    plt.clf()
    plt.hist(num_runs_tomtom/num_max, bins=num_max)
    plt.axvline(np.mean(num_runs_tomtom/num_max), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(0, color='k', linestyle='solid', linewidth=2)
    ave_repro = float(np.mean(num_runs_tomtom))/float(num_max)
    try:
        plt.title('Histogram of Reproducibility.  Avg repro ratio= {%f}' %ave_repro)
    except Exception as e:
        print('could not set the title for graph')
        print(e)
    plt.ylabel('Frequency')
    plt.xlabel('Reproducibility by ' + str(num_max) + ' fold')
    plt.savefig(output_directory + 'hist_runs_tomtom_' + str(num_max) + 'fold.svg')
    plt.close()

df_IC = pd.read_pickle(output_directory + 'info_content.pkl')
df_TomTom = pd.read_pickle(output_directory + 'top_match_TomTom_qED1.pkl')
df_CISBP = pd.read_pickle(output_directory + 'top_match_CISBP_qED1' + cisbp_txt + '.pkl')
# Match the filters' order using 'Filter' column
df_TomTom.rename(columns={'Query_ID':'Filter'}, inplace=True)
merged = pd.merge(df_influence, pd.merge(df_IC, df_TomTom, how = 'inner', on = 'Filter', suffixes=('_IC', '_TomTom')), how = 'inner', on = 'Filter', suffixes=('_Influence', ''))
merged = pd.merge(merged, df_CISBP[['Query_ID', 'TF_Name']], how = 'inner', left_on = 'Filter', right_on = 'Query_ID')

# Plot influence + IC + qvalue + reproducibility
flag_infl_log = True
infl_log_txt = ''
if flag_infl_log:
    merged['Influence'] = np.log10(merged['Influence'])
    infl_log_txt = 'Log10'
    
qvalue_neg_log = -np.log10(merged['q-value'])
qvalue_neg_log.fillna(0, inplace=True)

plt.clf()
fig, ax = plt.subplots(figsize=(9, 9))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(np.min(merged['IC'])-0.2, np.max(merged['IC'])+0.5)
plt.ylim(np.min(merged['Influence'])-0.1, np.max(merged['Influence'])+0.1) # 0.001 for original values; 0.1 for log 
# Produce a legend with the unique size to indicate the reproducibility, or probability of matching to CIS-BP
if os.path.isfile(output_directory + 'num_runs_tomtom.npy'):    
    scatter = ax.scatter(merged['IC'], merged['Influence'], c = merged['num_runs_tomtom'], s = qvalue_neg_log*20, cmap=plt.get_cmap('jet')) 
    handles, labels = scatter.legend_elements(prop='colors', num = 11)
    labels=[]
    labels = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    legend1 = ax.legend(handles, labels, loc='lower right', title='Reproducibility', bbox_to_anchor=(1.12,0))
    ax.add_artist(legend1)
else:
    scatter = ax.scatter(merged['IC'], merged['Influence'], c = qvalue_neg_log, s = qvalue_neg_log*20, cmap=plt.get_cmap('jet')) 
    handles, labels = scatter.legend_elements(prop='colors', num = 5)
    labels=[]
    labels = ['very low', 'low', 'medium', 'high', 'very high']
    legend1 = ax.legend(handles, labels, loc='lower right', title='Probability of matching to CIS-BP', bbox_to_anchor=(1.1,-0.01))
    ax.add_artist(legend1)

# Produce a legend with the unique color to indicate the intensity/probability of matching to CIS-BP
label_values = [np.min(merged['q-value']), 1e-10, 1e-7, 0.001, 0.01, 0.03, 0.05, 0.1, 0.5, 1]
values = -np.log10(label_values)*20
handles = []
labels = []
f = matplotlib.ticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
fmt = matplotlib.ticker.FuncFormatter(g)
g_plain = lambda x,pos : "${}$".format(f._formatSciNotation('%.2f' % x))
fmt_plain = matplotlib.ticker.FuncFormatter(g_plain)
#fmt.create_dummy_axis()
#fmt.set_bounds(np.min(label_values), np.max(label_values))
for val, lab in zip(values, label_values):
    size = np.sqrt(val)
    if np.isclose(size, 0.0):
        continue
    h = matplotlib.lines.Line2D([0], [0], ls="", color='k', ms=size, marker="o", alpha=0.5)
    handles.append(h)
    if lab>0.001:
        l = fmt_plain(lab)
    else:
        l = fmt(lab)
    labels.append(l)            
legend2 = ax.legend(handles, labels, loc="upper right", title="Q value matching to CIS-BP", bbox_to_anchor=(0.35, 1))#(1.1,1.15)
plt.xlabel("Information Content", fontsize=18) #, fontname='Helvetica'
plt.ylabel("Filter Influence - " + infl_log_txt, fontsize=18) 
for i, txt in enumerate(merged['Query_ID']):    
    if merged['Influence'][i] > 0.01*np.min(merged['Influence']) or merged['IC'][i] > 12 or merged['q-value'][i] < 0.01:
        TF_txt = merged['TF_Name'][i]
#        txt = txt + ':' + TF_txt # Show both filter number and motif names
        txt = TF_txt
        ax.annotate(txt, (merged['IC'][i] + 0.2, merged['Influence'][i])) #+ 0.005         
plt.savefig(output_directory + infl_log_txt + 'Influence' + influence_version_txt + '_IC_qvalue_plot' + cisbp_txt + filter_IC_txt + '.svg') #Filter vs TF 
plt.close() 