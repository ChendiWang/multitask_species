#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:15:23 2020

@author: chendi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd

species = 'human'
ref_genome = 'hg19'
tag_selection = '2015_DeepSea_ISM'
flag_first_run = True

folder = '../'
fasta_folder = '../' + species + 'GenomeSequencingConsortium/' + ref_genome + '/chromosomes/'

# Select the SNP dataset
dataset = 'GRASP'
# dataset = 'GWAS'

# SNP set
if dataset == 'GRASP':
    df = pd.read_csv('../2015_08_DeepSEA/41592_2015_BFnmeth3547_MOESM649_ESM.csv', sep=',', skiprows=2, header=0, index_col=0) 
    label_pos_txt = 'eQTL'
    label_neg_txt = ''
if dataset == 'GWAS':
    df = pd.read_csv('../2015_08_DeepSEA/41592_2015_BFnmeth3547_MOESM650_ESM.csv', sep=',', skiprows=2, header=0, index_col=0) 
    label_pos_txt = 'GWAS Catalog'
    label_neg_txt = ''

output_directory = folder + 'results/evolution/' + dataset + '/'

directory = os.path.dirname(output_directory)  
if not os.path.exists(directory):
    print('Creating directory %s' % output_directory)
    os.makedirs(directory)
else:
    print('Directory %s exists' % output_directory)
    
file_num = 1
pos_num = 78613
vcf_file_path_neg = output_directory + 'vcf_file_neg' + str(file_num) + '.vcf'
vcf_file_neg = open(vcf_file_path_neg, 'w')
vcf_file_neg.write('##fileformat=VCFv4.3' + '\n')
vcf_file_neg.write('#CHROM' + '\t' + 'POS' + '\t' + 'ID' + '\t' + 'REF' + '\t' + 'ALT' + '\n')    
    
for ind, rows in df.iterrows():
    print(ind)
    chromo, midpoint, ref, alt, label = rows.chr, rows.pos, rows.ref, rows.alt, rows.label    
    if label == label_pos_txt:
        continue
    else:
        if (ind - pos_num)%80000 == 0 and ind > pos_num:
            vcf_file_neg.close()
            file_num = file_num + 1
            vcf_file_path_neg = output_directory + 'vcf_file_neg' + str(file_num) + '.vcf'
            vcf_file_neg = open(vcf_file_path_neg, 'w')
            vcf_file_neg.write('##fileformat=VCFv4.3' + '\n')
            vcf_file_neg.write('#CHROM' + '\t' + 'POS' + '\t' + 'ID' + '\t' + 'REF' + '\t' + 'ALT' + '\n')
                               
        vcf_file_neg.write(chromo + '\t' + str(midpoint) + '\t' + 'id' + '\t' + ref + '\t' + alt + '\n')
    
vcf_file_neg.close()