#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 4 15:57:44 2019

@author: chendi
"""
#motif_summary.xlsx has a bunch of information about the motifs
#Columns B-K are output from the tomtom comparison to CIS-BP.  
#Columns L-M are additional information about the transcription factors that I pulled from CIS-BP.
#Column N - negative log q-value 
#Column O - DeepLIFT is another approach for extracting motifs from the model
#Column P - filter influence
#Column Q - filter influence based on high importance filters only
#Column R-AE - influence averaged over cell lineages (e.g. B-cells, T-cells)
#Column AF - number of times given motif is encountered in the data
#Column AG - average measured peak height over all OCRs containing that motif
#Column AH - number of input OCRs containing that motif
#Column AI - number of motifs extracted from 10 different training iterations of the model that match to this particular motif
#Column AJ -  number of models (out of 10) that contain a matching motif; essentially reproducibility  of the filter when re-training the model
#Columan AK - Information Content
#Column AL - Number of bases in motif with information content >0.2
#Column AM  - position of last informative base minus position of first informative base
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create output directory
folder = '../'
num_filters = 300
infl_celltype_version_text = '_signed'# '_absolute'#
infl_version_text = '_mean'#'_mean_activated'
species = 'Human'
data_aug_txt = '_rc_shiftleft_shiftright'
set_txt = '_topmatch'#'_wholeset'#
set_threshold_txt = '005'
cisbp_txt = '_otherspecies'#'_selfspecies'#
cisbp_plus_txt = '_plus'
flag_combined_species = False#True#
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
target_file_path = output_directory + 'TomTomED' + set_threshold_txt + base_file_tag + 'LOGO/' # Use ED1 for less strict cisbp matching
df = pd.read_csv(target_file_path + 'tomtom.tsv', sep="\t", header=0)
# Delete the last three lines (For TomTom information)
df = df[:-3]
# Find the Top matched filters
if set_txt == '_topmatch':
    df = df.drop_duplicates(subset = 'Query_ID')

# Prefer the direct TF, but sometimes D only exist in other species? download other species and combine them to pick up TF_Status = D
# Load in CIS-BP info file: _plus contains all motifs for a given TF, which includes all direct motifs, and all inferred motifs above the threshold.
if cisbp_txt == '_otherspecies':
    df_mus = pd.read_csv(folder + 'data/CIS_BP/Mus_musculus_2019_05_27_11_12_pm/TF_Information.txt', sep="\t", header=0)
    # M4471, Pax5 from Homo_sapiens
    df_homo = pd.read_csv(folder + 'data/CIS_BP/Homo_sapiens_2019_05_31_12_38_am/TF_Information.txt', sep="\t", header=0)
    # M1110 from Caenorhabditis elegans
    df_cae = pd.read_csv(folder + 'data/CIS_BP/Caenorhabditis_elegans_2019_05_31_1_29_am/TF_Information.txt', sep="\t", header=0)
    # M0756 Strongylocentrotus purpuratus
    df_str = pd.read_csv(folder + 'data/CIS_BP/Strongylocentrotus_purpuratus_2019_05_31_2_15_am/TF_Information.txt', sep="\t", header=0)
    # M4850 Drosophila melanogaster
    df_dro = pd.read_csv(folder + 'data/CIS_BP/Drosophila_melanogaster_2019_05_31_2_17_am/TF_Information.txt', sep="\t", header=0)
    # MELEAGRIS GALLOPAVO
    df_mel = pd.read_csv(folder + 'data/CIS_BP/Meleagris_gallopavo_2019_05_31_2_22_am/TF_Information.txt', sep="\t", header=0)
    # PBM CONSTRUCTS
    df_pbm = pd.read_csv(folder + 'data/CIS_BP/PBM_CONSTRUCTS_2019_05_31_2_25_am/TF_Information.txt', sep="\t", header=0)
    # Monodelphis domestica
    df_mono = pd.read_csv(folder + 'data/CIS_BP/Monodelphis_domestica_2019_05_31_2_27_am/TF_Information.txt', sep="\t", header=0)
    # Saccharomyces cerevisiae
    df_sac = pd.read_csv(folder + 'data/CIS_BP/Saccharomyces_cerevisiae_2019_05_31_11_22_am/TF_Information.txt', sep="\t", header=0)
    df_CISBP_all_motifs = pd.concat([pd.concat([pd.concat([pd.concat([pd.concat([pd.concat([pd.concat([pd.concat([df_mus, df_homo]), df_cae]), df_str]), df_dro]), df_mel]), df_pbm]), df_mono]), df_sac])
if cisbp_txt == '_selfspecies':
    df_CISBP_all_motifs = pd.read_csv(folder + 'data/CIS_BP/Mus_musculus_2019_05_27_11_12_pm/TF_Information_all_motifs' + cisbp_plus_txt + '.txt', sep="\t", header=0)

merged_match = pd.merge(df, df_CISBP_all_motifs[['Motif_ID', 'TF_Name', 'Family_Name', 'TF_Status', 'Motif_Type']], how = 'inner', left_on = 'Target_ID', right_on = 'Motif_ID', suffixes=('_result_motif', '_cisbp_motif'))
if set_txt == '_topmatch_directlyinfer':
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

# # Add in deeplift match
merged_match['DeepLIFT_matches'] = 'default value'

# Information Content
#read in meme file
with open(output_directory + 'filter_motifs_pwm.meme') as fp:
    line = fp.readline()
    motifs=[]
    motif_names=[]
    while line:
        #determine length of next motif
        if line.split(" ")[0]=='MOTIF':
            #add motif number to separate array
            motif_names.append(line.split(" ")[1])           
            
            #get length of motif
            line2=fp.readline().split(" ")
            motif_length = int(float(line2[5]))
            
            #read in motif 
            current_motif=np.zeros((19, 4)) # Edited pad shorter ones with 0
            for i in range(motif_length):
                current_motif[i,:] = fp.readline().split("\t")
            
            motifs.append(current_motif)

        line = fp.readline()
        
    motifs = np.stack(motifs)  
    motif_names = np.stack(motif_names)

#set background frequencies of nucleotides
bckgrnd = [0.25, 0.25, 0.25, 0.25]

#compute information content of each motif
info_content = []
position_ic = []
for i in range(motifs.shape[0]): 
    length = motifs[i,:,:].shape[0]
    position_wise_ic = np.subtract(np.sum(np.multiply(motifs[i,:,:],np.log2(motifs[i,:,:] + 0.00000000001)), axis=1),np.sum(np.multiply(bckgrnd,np.log2(bckgrnd))))                                    
    position_ic.append(position_wise_ic)
    ic = np.sum(position_wise_ic, axis=0)
    info_content.append(ic)
    
info_content = np.stack(info_content)
position_ic = np.stack(position_ic)

#length of motif with high info content
n_info = np.sum(position_ic>0.2, axis=1)

#"length of motif", i.e. difference between first and last informative base
ic_idx = pd.DataFrame(np.argwhere(position_ic>0.2), columns=['row', 'idx']).groupby('row')['idx'].apply(list)
motif_length = []
for row in ic_idx:
    motif_length.append(np.max(row)-np.min(row)+1) 

motif_length = np.stack(motif_length)

#create pandas data frame:
info_content_df = pd.DataFrame(data=[motif_names, info_content, n_info, pd.to_numeric(motif_length)]).T
info_content_df.columns=['Filter', 'IC', 'Num_Informative_Bases', 'Motif_Length']

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
    if infl_version_text == '_diff':
        influence_cellwise_mean = np.load(output_directory + 'filter_cellwise_influence' + infl_celltype_version_text + '_mean' + '.npy')
        influence_cellwise_mean_act = np.load(output_directory + 'filter_cellwise_influence' + infl_celltype_version_text + '_mean_activated' + '.npy')
        influence_cellwise = influence_cellwise_mean_act - influence_cellwise_mean
    else:
        influence_cellwise = np.load(output_directory + 'filter_cellwise_influence' + infl_celltype_version_text + infl_version_text + '.npy')

    # Read cell_type_names = class names
    with open('../data/' + species + '_Data/cell_type_names.txt','r') as class_names_file:
        if species == 'Mouse':
            class_names = []
            for line in class_names_file:
                line = line.strip() # remove /n at the end
                task_num, class_name = line.split()
                class_names.append(class_name)
            class_names = list(filter(None, class_names))  # This removes empty entries 
        elif species == 'Human':
            class_names = class_names_file.read().split('\t')
            class_names[-1] = class_names[-1].strip() # remove the last newline
            class_names = list(filter(None, class_names))  # This removes empty entries
    cell_names = class_names
    if model_name == 'model2_Human_data':
        influence_df = pd.DataFrame(data=influence_cellwise,
                                    columns=list(np.arange(influence_cellwise.shape[1])))
    else:        
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
#del merged['Filter'] # Keep 'Filter' to know which ones are inactive filters; and remove Query_ID
del merged['Query_ID']
merged.rename(columns={'Filter':'Query_ID'}, inplace=True)

# Save motif summary file
merged.to_excel(output_directory + 'motif_summary' + influence_txt + infl_celltype_version_text + infl_version_text + set_txt + set_threshold_txt + cisbp_txt + cisbp_plus_txt + '.xlsx')