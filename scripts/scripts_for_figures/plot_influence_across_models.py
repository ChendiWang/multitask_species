#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:06:30 2020

@author: chendi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

#######species comparison######
# influences of mouse vs. human for each filter from combined species model
output_directory_both = '../results/models/keras_version/frozen/result_figures/'
data_aug_txt = '_rc_shiftleft_shiftright'
branch_tag = '_hard'
infl_version_text = '_mean'
model_name = 'model_multitask_MaH_ratiobased_Basset_adam_lr0001_dropout_03_conv_relu_max_BN_convfc_batch100_loss_combine001_00052_bysample_epoch10_best_separatelayer_hard_branched' + data_aug_txt

output_directory_mouse = '../results/motifs/keras_version/frozen/' + \
                          model_name + '/well_predicted/Mouse/'
output_directory_human = '../results/motifs/keras_version/frozen/' + \
                          model_name + '/well_predicted/Human/'
                          
influence_mouse = np.load(output_directory_mouse + 'filter_influence' + infl_version_text + '.npy')
influence_human = np.load(output_directory_human + 'filter_influence' + infl_version_text + '.npy')
influence_mouse[influence_mouse==0] = np.nan
influence_mouse = np.log10(influence_mouse)
influence_mouse[np.isnan(influence_mouse)] = 0
                
influence_human[influence_human==0] = np.nan
influence_human = np.log10(influence_human)
influence_human[np.isnan(influence_human)] = 0

influence_difference_threshold = 0.35

# Find the similarity between two models
similarity_matrix = np.load(output_directory_human + 'similarity_to_mouse_trim02outlier.npy')
nan_mouse = np.where(influence_mouse==0)[0]
nan_human = np.where(influence_human==0)[0]
for i in range(len(nan_mouse)):
    similarity_matrix = np.insert(similarity_matrix, nan_mouse[i], values=0, axis=0)
for i in range(len(nan_human)):
    similarity_matrix = np.insert(similarity_matrix, nan_human[i], values=0, axis=1)
similarity = similarity_matrix.diagonal()

filter_num = []
for i in range(300):
    filter_num.append('filter'+str(i))
    
df_influence = pd.DataFrame({'Filter': filter_num, 'influence_mouse': influence_mouse, 'influence_human': influence_human})

cisbp_txt = '_otherspecies'#'self_species'
df_CISBP_mouse = pd.read_pickle(output_directory_mouse + 'top_match_CISBP_qED1' + cisbp_txt + '.pkl')
df_CISBP_human = pd.read_pickle(output_directory_human + 'top_match_CISBP_qED1' + cisbp_txt + '.pkl')

merged = pd.merge(df_influence, df_CISBP_mouse[['Query_ID', 'TF_Name', 'Family_Name']], how = 'outer', left_on = 'Filter', right_on = 'Query_ID')
merged = pd.merge(merged, df_CISBP_human[['Query_ID', 'TF_Name', 'Family_Name']], how = 'outer', left_on = 'Filter', right_on = 'Query_ID', suffixes=('_mouse', '_human'))

del merged['Query_ID_mouse']
del merged['Query_ID_human']

merged_mouse = merged
num_runs_tomtom = np.load(output_directory_mouse + 'num_runs_tomtom.npy')
merged_mouse['num_runs_tomtom'] = num_runs_tomtom / 20
mouse_ic = pd.read_pickle(output_directory_mouse + 'info_content.pkl')
merged_wo_nan_mouse = merged_mouse.loc[merged_mouse['influence_mouse'] != 0]
merged_wo_nan_mouse = pd.merge(merged_wo_nan_mouse, mouse_ic, how = 'inner', on = 'Filter')

merged_human = merged
num_runs_tomtom = np.load(output_directory_human + 'num_runs_tomtom.npy')
merged_human['num_runs_tomtom'] = num_runs_tomtom / 20
human_ic = pd.read_pickle(output_directory_human + 'info_content.pkl')
merged_wo_nan_human = merged_human.loc[merged_human['influence_human'] != 0]
merged_wo_nan_human = pd.merge(merged_wo_nan_human, human_ic, how = 'inner', on = 'Filter')

merged = merged.replace(np.nan, '-', regex=True)

min_value = np.minimum(np.min(merged['influence_mouse']), np.min(merged['influence_human'])) + 0.1
max_value = np.maximum(np.max(merged['influence_mouse']), np.max(merged['influence_human'])) + 0.1
# For display: should be -inf
min_value = min_value - 1
merged['influence_mouse'].loc[merged['influence_mouse']==0] = min_value + 0.1
merged['influence_human'].loc[merged['influence_human']==0] = min_value + 0.1

# Add in motif names
plt.clf()
fig, ax = plt.subplots(figsize=(9, 9))

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value) 
scatter = ax.scatter(merged['influence_mouse'], merged['influence_human']) 
ax.plot([0, 1], [0, 1], ls='--', color = 'r', transform=ax.transAxes)
plt.title('Influence on the same fiters from combined species model', fontsize=18) 
plt.xlabel('Log10 influence for Mouse', fontsize=18)
plt.ylabel('Log10 influence for Human', fontsize=18) 
for i, txt in enumerate(merged['Filter']):   
    if np.absolute(merged['influence_mouse'][i] - merged['influence_human'][i]) > influence_difference_threshold: 
        txt = merged['TF_Name_mouse'][i] + ':' + merged['TF_Name_human'][i]
        ax.annotate(txt, (merged['influence_mouse'][i] + 0.1, merged['influence_human'][i])) 
plt.tight_layout()
plt.savefig(output_directory_both + 'log10_influence_across_species' + branch_tag + '_TF_name.svg') 
plt.close()

# Add in Family names
plt.clf()
fig, ax = plt.subplots(figsize=(11, 9))
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value) 
scatter = ax.scatter(merged['influence_mouse'], merged['influence_human'], c=similarity, cmap='jet') 
handles, labels = scatter.legend_elements(prop='colors', num = 11)
labels = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
legend1 = ax.legend(handles, labels, loc='upper right', title='Similarity', bbox_to_anchor=(1.2,1))
ax.add_artist(legend1)
ax.plot([0, 1], [0, 1], ls='--', color = 'k', transform=ax.transAxes)
plt.title('Influence on the same fiters from combined species model', fontsize=18) 
plt.xlabel('Log10 influence for Mouse', fontsize=18)
plt.ylabel('Log10 influence for Human', fontsize=18) 
for i, txt in enumerate(merged['Filter']):   
    if np.absolute(merged['influence_mouse'][i] - merged['influence_human'][i]) > influence_difference_threshold: 
        TF_txt = merged['Family_Name_mouse'][i] + '//' + merged['Family_Name_human'][i]
#        ax.annotate(txt + ':' + TF_txt, (merged['influence_mouse'][i] + 0.1, merged['influence_human'][i]), fontsize=12) 
        ax.annotate(TF_txt, (merged['influence_mouse'][i] + 0.1, merged['influence_human'][i]-0.05), fontsize=14) 
plt.tight_layout()
plt.savefig(output_directory_both + 'log10_influence_across_species' + branch_tag + '_family_name.svg') 
plt.close()

# Color code using the reproducibility
reproducibility_human = np.load(output_directory_human + 'num_runs_tomtom.npy')
reproducibility_mouse = np.load(output_directory_mouse + 'num_runs_tomtom.npy')

reproducibility = np.zeros(reproducibility_human.shape)
reproducibility[0:15] = reproducibility_mouse[0:15]
reproducibility[15:30] = reproducibility_human[15:30]
for i in range(30, 300):
    reproducibility[i] = np.maximum(reproducibility_human[i], reproducibility_mouse[i])

plt.clf()
fig, ax = plt.subplots(figsize=(10.5, 9))
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value) 
scatter = ax.scatter(merged['influence_mouse'], merged['influence_human'], c=reproducibility, cmap='jet') 
handles, labels = scatter.legend_elements(prop='colors', num = 11)
labels = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
legend1 = ax.legend(handles, labels, loc='upper right', title='Reproducibility', bbox_to_anchor=(1.2,1))
ax.add_artist(legend1)
ax.plot([0, 1], [0, 1], ls='--', color = 'k', transform=ax.transAxes)
plt.title('Influence on the same fiters from combined species model', fontsize=18) 
plt.xlabel('Log10 influence for Mouse', fontsize=18)
plt.ylabel('Log10 influence for Human', fontsize=18) 
for i, txt in enumerate(merged['Filter']):   
    if np.absolute(merged['influence_mouse'][i] - merged['influence_human'][i]) > influence_difference_threshold: 
        TF_txt = merged['Family_Name_mouse'][i] + ':' + merged['Family_Name_human'][i]
        ax.annotate(txt + ':' + TF_txt, (merged['influence_mouse'][i] + 0.1, merged['influence_human'][i])) 
plt.tight_layout()
plt.savefig(output_directory_both + 'log10_influence_across_species' + branch_tag + '_family_name_color_reproducibility.svg') 
plt.close()