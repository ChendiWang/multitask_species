#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:35:18 2019

@author: chendi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict

try:
    import cPickle as pickle
except:
    import pickle
    
import utils


folder = '../'
# ref_genome = 'mm10'
# species = 'mouse'
ref_genome = 'hg19'
species = 'human'
immgen_name_flag = True

# Exclude the gender chromosomes
if species == 'mouse':
    num_chromo = 19
    interested_region_file = '../data/' + species.capitalize() + '_Data/ImmGenATAC1219.peak.filteredSM0.05.bed'
if species == 'human':
    num_chromo = 22
    interested_region_file = '../data/' + species.capitalize() + '_Data/26August2017_EJCsamples_allReads_250bp.bed'     

# Parse faste file and turn into dictionary
sequence_path = folder + species + 'GenomeSequencingConsortium/'
result_path = folder + ref_genome + '_results/'

if not os.path.exists(result_path + 'chr_dict.pickle'):
    chr_dict = dict()
    for chromo in range(1, num_chromo + 1): 
        print(chromo)
        chr_file_path = sequence_path + ref_genome + '/chromosomes/chr{}.fa'.format(chromo)
        chr_dict.update(SeqIO.to_dict(SeqIO.parse(open(chr_file_path), 'fasta')))
    pickle.dump(chr_dict, open(os.path.join(result_path, 'chr_dict.pickle'), 'wb'))
else:
    chr_dict = pickle.load(open(os.path.join(result_path, 'chr_dict.pickle'), 'rb'))

print('chr_dict ready')

target_chr = ['chr{}'.format(i) for i in range(1, num_chromo + 1)]

# Load interested region from bed file
original_window_size = 251
window_size = original_window_size#600
tag_selection = ''#'_' + str(window_size) + 'bp_with_overlap'

positions = defaultdict(list)
with open(interested_region_file) as f:
    for line in f:
        name, chromo, start, stop = line.split()
        positions[name].append((chromo, int(start), int(stop)))

# Extract sequences and other information
short_seq_records = []
one_hot_seqs = []
strings = []
invalid_ids = []
peak_names_features = []
start_point = [] 
chromo_list = []

extra_len = int(np.floor((window_size - original_window_size)/2))
num_overlap_case = 0   
midpoint_comp = -1 * window_size
chromo_comp = next(iter(positions.items()))[-1][-1][0]

for name_index in positions:
    for (chromo, start, stop) in positions[name_index]:
        if immgen_name_flag:
            name = name_index # For Immgen dataset
        else:                   
            name = chromo + '_start_' + str(start) # For other datasets
        if chromo in target_chr:
            long_seq_record = chr_dict[chromo]
            long_seq = long_seq_record.seq
            total_len = len(long_seq)
            print(chromo + ':' + str(total_len))
            alphabet = long_seq.alphabet
            
            if extra_len == 0: # no extension
                short_seq = str(long_seq)[start-1:stop] # -1: 0 index
            else:
                short_seq = str(long_seq)[start-1-extra_len:stop+extra_len+1] # -1: 0 index
            
            # Test if extended window include more than one original smaller window
            if chromo_comp != chromo:
                chromo_comp = chromo
                midpoint_comp = -1 * window_size
            if (stop + start)/2 - midpoint_comp < window_size: 
                num_overlap_case = num_overlap_case + 1
                # TODO: remove window that have more than one OCR peaks
            midpoint_comp = (stop + start)/2 
            one_hot_seq = utils.one_hot_encode_along_channel_axis(short_seq, onehot_axis=1)
            string = short_seq.lower()
                        
            # Detect if there are any Ns in the sequence
            if np.min(np.sum(one_hot_seq, axis=1)) == 1 and one_hot_seq.shape[0] == window_size: # it is valid sequence without Ns
                peak_names_features.append(name)
                start_point.append(start)
                strings.append(string)
                one_hot_seqs.append(one_hot_seq)
                short_seq_record = SeqRecord(Seq(short_seq, alphabet), id=name, description='')
                short_seq_records.append(short_seq_record)
                chromo_list.append(chromo)
            else:
                invalid_ids.append(name) 
        else:
            invalid_ids.append(name)

one_hot_seqs = np.stack(one_hot_seqs) 
peak_names_features = np.stack(peak_names_features)
start_point = np.stack(start_point)
chromo_list = np.stack(chromo_list)

#~~~~~ Skipped this section if only to extract sequences from bed file~~~~~#
# Write corresponding cell peak intensity file
# When have the original intensity full list file (peak_names, cell_type_array)
# import pandas as pd
# peak_names_labels_df = pd.read_csv('../data/' + species.capitalize() + '_Data/quantilenorm_log2_countData.txt', sep="\t", header=0)
# peak_names_labels = peak_names_labels_df['peakID'].values.astype(str)

peak_names_labels = np.load('../data/' + species.capitalize() + '_Data/peak_names.npy').astype(str)
input_labels = np.load('../data/' + species.capitalize() + '_Data/cell_type_array.npy')
input_labels = input_labels.astype(np.float32)

# Take the intersection of two sets of variables
peak_names_intersect = np.intersect1d(peak_names_features, peak_names_labels)
# In this specific case this version has the right order (monotonic), same as the slow way
idx_features = np.isin(peak_names_features, peak_names_intersect)
idx_labels = np.isin(peak_names_labels, peak_names_intersect)

## Slow way which makes sure the order between feature and labels is matching, 
## by following peak_names_intersect, but this way interrupts the chromo order (could convert into int and reorder...)
#idx_labels= []
#idx_features= []
#for name_temp in peak_names_intersect:
#    idx_features_temp = np.where(peak_names_labels==name_temp)
#    idx_labels_temp = np.where(peak_names_labels==name_temp)
#    idx_features.append(idx_features_temp[0][0])
#    idx_labels.append(idx_labels_temp[0][0])
#idx_features = np.stack(idx_features)
#idx_labels = np.stack(idx_labels)

one_hot_seqs = one_hot_seqs[idx_features, :, :]
peak_names_features = peak_names_features[idx_features]
chromo_list = chromo_list[idx_features]
input_labels = input_labels[idx_labels, :]
peak_names_labels = peak_names_labels[idx_labels]

# Write to file
if np.sum(peak_names_features != peak_names_labels) > 0:
    print("Order of peaks not matching for sequences/intensities!")
else:
    np.save(result_path + 'cell_type_array' + tag_selection + '.npy', input_labels)
    np.save(result_path + 'one_hot_seqs_ACGT' + tag_selection + '.npy', one_hot_seqs)
    np.save(result_path + 'peak_names' + tag_selection + '.npy', peak_names_features)
    np.save(result_path + 'chromosome' + tag_selection + '.npy', chromo_list)
#    np.save(result_path + 'peak_names_' + tag_selection + '.npy', peak_names_intersect) # when using the slow way

with open(result_path + 'invalid_ids' + tag_selection + '.txt', 'w') as f:
    f.write(json.dumps(invalid_ids))
#np.savetxt(result_path + 'start_point_' + tag_selection + '.txt', start_point, fmt='%i', delimiter=',')
#np.save(result_path + 'sequence_strings_' + tag_selection + '.npy', strings)
#np.save(result_path + 'short_seq_records_' + tag_selection + '.npy', short_seq_records)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#