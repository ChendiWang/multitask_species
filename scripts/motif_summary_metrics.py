#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 4 15:57:44 2019

@author: chendi
"""
#motif_summary.xlsx has a bunch of information about the motifs
#1. output from the tomtom comparison to CIS-BP.  
#2. additional information about the transcription factors that was pulled from CIS-BP.
#3. filter influence
#4. number of input OCRs containing that motif
#5. number of motifs extracted from k different training iterations of the model that match to this particular motif
#6. number of models (out of k) that contain a matching motif; essentially reproducibility  of the filter when re-training the model
#7. Information Content

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

# Create output directory
folder = '../'
num_filters = 300
infl_celltype_version_text = '_signed'
infl_version_text = '_mean'
species = 'Human'
data_aug_txt = '_rc_shiftleft_shiftright'
set_txt = '_topmatch'#'_wholeset'#
set_threshold_txt = '005'
cisbp_txt = '_otherspecies'#'_selfspecies'#
cisbp_plus_txt = '_plus'
flag_combined_species = False
branch_tag = '_hard'

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

output_directory = folder + 'results/motifs/keras_version/frozen/' + model_name + '/well_predicted/' + species_text

# The Database being compared to
if species == 'Mouse':
    base_file_tag = 'Mus'
if species == 'Human':
    base_file_tag = 'Homo'
# Read in TomTom file
target_file_path = output_directory + 'TomTomED' + set_threshold_txt + base_file_tag + 'LOGO/'
df = pd.read_csv(target_file_path + 'tomtom.tsv', sep="\t", header=0)
df = df[:-3]
# Find the Top matched filters
if set_txt == '_topmatch':
    df = df.drop_duplicates(subset = 'Query_ID')

# Prefer the direct TF, but sometimes D only exist in other species. Download other species and combine them to pick up TF_Status = D
# Load in CIS-BP info file: _plus contains all motifs for a given TF, which includes all direct motifs, and all inferred motifs above the threshold.
if cisbp_txt == '_otherspecies':
    df_mus = pd.read_csv(folder + 'data/CIS_BP/Mus_musculus_2019_05_27_11_12_pm/TF_Information.txt', sep="\t", header=0)
    df_homo = pd.read_csv(folder + 'data/CIS_BP/Homo_sapiens_2019_05_31_12_38_am/TF_Information.txt', sep="\t", header=0)
    df_cae = pd.read_csv(folder + 'data/CIS_BP/Caenorhabditis_elegans_2019_05_31_1_29_am/TF_Information.txt', sep="\t", header=0)
    df_str = pd.read_csv(folder + 'data/CIS_BP/Strongylocentrotus_purpuratus_2019_05_31_2_15_am/TF_Information.txt', sep="\t", header=0)
    df_dro = pd.read_csv(folder + 'data/CIS_BP/Drosophila_melanogaster_2019_05_31_2_17_am/TF_Information.txt', sep="\t", header=0)
    df_mel = pd.read_csv(folder + 'data/CIS_BP/Meleagris_gallopavo_2019_05_31_2_22_am/TF_Information.txt', sep="\t", header=0)
    df_pbm = pd.read_csv(folder + 'data/CIS_BP/PBM_CONSTRUCTS_2019_05_31_2_25_am/TF_Information.txt', sep="\t", header=0)
    df_mono = pd.read_csv(folder + 'data/CIS_BP/Monodelphis_domestica_2019_05_31_2_27_am/TF_Information.txt', sep="\t", header=0)
    df_sac = pd.read_csv(folder + 'data/CIS_BP/Saccharomyces_cerevisiae_2019_05_31_11_22_am/TF_Information.txt', sep="\t", header=0)
    df_CISBP_all_motifs = pd.concat([pd.concat([pd.concat([pd.concat([pd.concat([pd.concat([pd.concat([pd.concat([df_mus, df_homo]), df_cae]), df_str]), df_dro]), df_mel]), df_pbm]), df_mono]), df_sac])
if cisbp_txt == '_selfspecies':
    df_CISBP_all_motifs = pd.read_csv(folder + 'data/CIS_BP/Mus_musculus_2019_05_27_11_12_pm/TF_Information_all_motifs' + cisbp_plus_txt + '.txt', sep="\t", header=0)

merged_match = pd.merge(df, df_CISBP_all_motifs[['Motif_ID', 'TF_Name', 'Family_Name', 'TF_Status', 'Motif_Type']], how = 'inner', left_on = 'Target_ID', right_on = 'Motif_ID', suffixes=('_result_motif', '_cisbp_motif'))
if set_txt == '_topmatch_directly_infer':
    merged_match = merged_match.loc[merged_match['TF_Status'] == 'D']
if set_txt == '_topmatch':
    merged_match = merged_match.drop_duplicates(subset = 'Query_ID')

# Add logo file name
merged_match['sort_index'] = merged_match['Query_ID'].str[6:]
merged_match['sort_index'] = pd.to_numeric(merged_match['sort_index'])
merged_match = merged_match.sort_values(by='sort_index')
del merged_match['sort_index']
del merged_match['Motif_ID']
merged_match.reset_index(inplace=True)
del merged_match['index']
merged_match['logo_file_name'] = pd.Series('align_' + merged_match['Query_ID'] + '_0_+' + merged_match['Target_ID'] + '.eps', index=merged_match.index)

# Compute Information Content
[motifs, motif_names] = utils.read_meme(output_directory + 'filter_motifs_pwm.meme')
info_content_df = utils.compute_ic(motifs, motif_names)

# Collect influence information
filter_num = []
for i in range (0, num_filters):
    filter_num.append('filter'+str(i))
filter_num = np.stack(filter_num)

if not os.path.isfile(output_directory + 'filter_influence' + infl_version_text + '.npy'):    
    # Use when there is no info about Influence
    influence_df = pd.DataFrame(data=[filter_num]).T
    influence_df.columns=['Filter']
    influence_txt = '_without_influence'
else:
    # Use when there is info about Influence
    influence_txt = '_with_celltype_influence'
    influence = np.load(output_directory + 'filter_influence' + infl_version_text + '.npy')
    influence_cellwise = np.load(output_directory + 'filter_cellwise_influence' + infl_celltype_version_text + infl_version_text + '.npy')   
    cell_names = utils.read_class_names('../data/' + species + '_Data/cell_type_names.txt', species)     
    influence_df = pd.DataFrame(data=influence_cellwise,
                                columns=cell_names)
    influence_df.insert(loc=0, column='Influence', value=pd.to_numeric(influence))
    influence_df.insert(loc=0, column='Filter', value=filter_num)
       
# Activation stats for OCRs
num_seqs = np.loadtxt(output_directory + 'nseqs_per_filters.txt')
influence_df['num_seqs'] = num_seqs

activated_OCRs = np.load(output_directory + 'activated_OCRs.npy')
Ave_OCR_activity = np.mean(activated_OCRs, axis = 1)
influence_df['Ave_OCR_activity'] = Ave_OCR_activity

Num_Act_OCRs = np.load(output_directory + 'n_activated_OCRs.npy')
influence_df['Num_Act_OCRs'] = Num_Act_OCRs

# Add in reproducibility information
if not os.path.isfile(output_directory + 'num_runs_tomtom.npy'):
    influence_df['num_runs_tomtom'] = 'default value'
else:
    num_runs_tomtom = np.load(output_directory + 'num_runs_tomtom.npy')
    influence_df['num_runs_tomtom'] = num_runs_tomtom    
    
# merge dataframes using outer to keep all 300 filters
merged = pd.merge(influence_df, info_content_df, how = 'outer', on = 'Filter')

merged = pd.merge(merged_match, merged, how = 'outer', left_on = 'Query_ID', right_on = 'Filter')
cols = list(merged)
cols.insert(0, cols.pop(cols.index('Filter')))
merged = merged.loc[:, cols]
del merged['Query_ID']
merged.rename(columns={'Filter':'Query_ID'}, inplace=True)

# Save motif summary file
merged.to_excel(output_directory + 'motif_summary' + influence_txt + infl_celltype_version_text + infl_version_text + set_txt + set_threshold_txt + cisbp_txt + cisbp_plus_txt + '.xlsx')