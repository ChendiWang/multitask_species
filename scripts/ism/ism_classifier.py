#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 5 09:54:48 2020

@author: chendi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import xgboost as xgb

# Logistic regression classifier to tell two classes pos vs. neg
species = 'Human'
ref_genome = 'hg19'
tag_selection = '2015_DeepSea_ISM'
flag_overlapped_celltype = False

folder = '../'
subfolder = 'frozen/'
data_augmentation = 'rc_shiftleft_shiftright/'
#run_num = 'Basset_adam_lr0001_dropout_03_conv_relu_max_BN_convfc_batch_100_loss_combine00052_bysample_epoch10_best_separatelayer_' + data_augmentation
#model_name = 'model' + '_' + species + '_' + run_num
model_name = 'model_multitask_MaH_ratiobased_Basset_adam_lr0001_dropout_03_conv_relu_max_BN_convfc_batch100_loss_combine001_00052_bysample_epoch10_best_separatelayer_hard_branched_' + data_augmentation

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

output_directory = folder + 'results/motifs/keras_version/'  + subfolder + model_name + \
        tag_selection + '/' + dataset + '/ISM_stats' + '/'
#sad = np.load(output_directory + 'sad.npy')
sad_pos = np.load(output_directory + 'sad_pos.npy')
sad_neg = np.load(output_directory + 'sad_neg.npy')
# Use subset of negative set
#sad_neg = sad_neg[0:sad_pos.shape[0],:]
sad = np.concatenate((sad_pos, sad_neg), axis=0)

pred_before = np.load(output_directory + 'pred_before.npy')
pred_after = np.load(output_directory + 'pred_after.npy')

X_abs_diff = np.absolute(sad) # |ref - alt|
X_rel_diff = pred_before / pred_after # ref/alt

X = np.concatenate((X_abs_diff, X_rel_diff), axis=1)

model_feature_name = []
for i in range (0, X.shape[1]):
    model_feature_name.append(str(i))
    
df_X = pd.DataFrame(X, columns=model_feature_name)
df = pd.concat([df, df_X], axis=1)
df['keyID'] = df['chr'].str[3:] + '-' + df['pos'].astype(str) + '-' + df['ref'].astype(str) + '-' + df['alt'].astype(str)
df = df.drop_duplicates(subset = 'keyID')

# Add in evolutionary features # imputng missing vaues: 0/nan
output_directory_evo = folder + 'results/evolution/' + dataset + '/'
pos_scores = pd.read_csv(output_directory_evo + 'pos.tsv.gz', sep='\t', skiprows=1, header=0, low_memory=False) 
pos_scores['keyID'] = pos_scores['#Chr'].astype(str) + '-' + pos_scores['Pos'].astype(str) + '-' + pos_scores['Ref'].astype(str) + '-' + pos_scores['Alt'].astype(str)
pos_scores = pos_scores.drop_duplicates(subset = 'keyID')

for i in range(1,15): # 14 neg result files
    df_tmp = pd.read_csv(output_directory_evo + 'neg' + str(i) + '.tsv.gz', sep='\t', skiprows=1, header=0, low_memory=False) 
    df_tmp['keyID'] = df_tmp['#Chr'].astype(str) + '-' + df_tmp['Pos'].astype(str) + '-' + df_tmp['Ref'].astype(str) + '-' + df_tmp['Alt'].astype(str)
    df_tmp = df_tmp .drop_duplicates(subset = 'keyID')
    if i == 1:
        neg_scores = df_tmp 
    else:
        neg_scores = neg_scores.append(df_tmp)
del df_tmp

df_evo = pos_scores.append(neg_scores)
df_evo = df_evo.drop_duplicates(subset = 'keyID')

# Merged two df to get the intersection of non-missing values
merged = pd.merge(df, df_evo[['keyID', '#Chr', 'Pos', 'Ref', 'Alt', 'priPhCons', 'priPhyloP', 'GerpN', 'GerpRS', 'GerpS']], how = 'inner', on = 'keyID', suffixes=('_model', '_evolution'))
#merged = merged.dropna()
merged['priPhCons'] = merged['priPhCons'].fillna(0.115)
merged['priPhyloP'] = merged['priPhyloP'].fillna(-0.033)
merged['GerpN'] = merged['GerpN'].fillna(1.909)
merged['GerpRS'] = merged['GerpRS'].fillna(0)
merged['GerpS'] = merged['GerpS'].fillna(-0.200)

# Tag examples with labels: 0 as positive set and 1 as negative set                              
pos_df = merged.loc[merged['label']==label_pos_txt]
neg_df = merged.loc[merged['label']!=label_pos_txt]
num_pos = len(pos_df) 
num_neg = len(neg_df)

tags_pos = np.zeros(num_pos).astype(np.int32) # sad_pos.shape[0] num_pos
tags_neg = np.ones(num_neg).astype(np.int32)
tags = np.concatenate((tags_pos, tags_neg))
y = tags

# Get Features
evo_feature_name = ['priPhCons', 'priPhyloP', 'GerpN', 'GerpRS'] # 'GerpS'
feature_name = model_feature_name + evo_feature_name
#feature_name = evo_feature_name
features = merged.loc[:, feature_name].values

trainX, testX, trainy, testy = train_test_split(features, y, test_size=0.1, random_state=0)

# Feature scaling - separate training and testing scaling
scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)

testX = scaler.transform(testX)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]
# fit a model
class_weight_dict = {0: 1.0, 1: sad_neg.shape[0]/sad_pos.shape[0]}#None#
#model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', penalty='l2', C=1.0, class_weight = class_weight_dict)
model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1, max_depth = 5, reg_lambda=0.1)

model.fit(trainX, trainy)
lr_probs = model.predict_proba(testX)
lr_probs = lr_probs[:, 1]

print('Accuracy score:')
print(model.score(testX, testy))

probs = model.predict_proba(testX)
preds = model.predict(testX)
cm = confusion_matrix(testy, preds)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
# Summary report
print(classification_report(testy, preds))

# Calculate scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# Calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# Plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Predict class values
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
lr_f1, lr_auc = f1_score(testy, preds), auc(lr_recall, lr_precision)
# Summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# Plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()