# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 00:33:46 2022

@author: DELL
"""
import os
os.chdir("D:/1. Master's/7. Thesis/1. thesis/datasets")
import autotime
%load_ext autotime
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from math import log
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from sklearn.utils import *
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from numpy import mean
import re, math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from collections import Counter

trans = pd.read_csv (r"banks\v2.1\data\bank_mixed\transactions.csv\transactions.csv")
acc = pd.read_csv(r"banks\v2.1\data\bank_mixed\accounts.csv\accounts.csv")
acc_feature = pd.read_csv(r"banks\v2.1\features\bank_mixed\features.csv\all_features.csv")
acc_feature = acc_feature[['account','sum_amount', 'in_count', 'sum_in_amount',
        'max_in_amount', 'avg_in_amount', 'out_count', 'sum_out_amount',
        'max_out_amount', 'avg_out_amount', 'min_interval', 'avg_interval',
        'max_interval', 'sg_num_accts', 'sg_num_txs',
        'sg_min_amount', 'sg_max_amount', 'sg_avg_amount', 'sg_sum_amount',
        'sg_all_period', 'sg_all_amount_diff',
        'sg_all_amount_ratio', 'sg_all_date_diff', 'sg_amount_ordered',
        'sg_date_ordered', 'sg_in_count', 'sg_out_count', 'sg_in_amount',
        'sg_out_amount', 'sg_acct_amount_diff', 'sg_acct_amount_ratio',
        'sg_acct_period', 'sg_acct_date_diff', 'sg_depth', 'gs_num_accts',
        'gs_num_txs', 'gs_min_amount', 'gs_max_amount', 'gs_avg_amount',
        'gs_sum_amount', 'gs_all_period', 'gs_all_amount_diff',
        'gs_all_amount_ratio', 'gs_all_date_diff', 'gs_amount_ordered',
        'gs_date_ordered', 'gs_in_count', 'gs_out_count', 'gs_in_amount',
        'gs_out_amount', 'gs_acct_amount_diff', 'gs_acct_amount_ratio',
        'gs_acct_period', 'gs_acct_date_diff', 'gs_depth']]
acc= pd.merge(acc, acc_feature, how='left', left_on=['acct_id'], right_on=['account'])

merge = pd.merge(trans,acc, how='left', left_on ='orig_acct', right_on='acct_id')#index = trans_id
#remove duplicate column and alert_id 
merge.drop(['acct_id','account','alert_id','dsply_nm','orig_acct', 'bene_acct'], axis=1, inplace =True)
#drop empty columns 
merge.drop(merge.columns[merge.apply(lambda col: (col.isnull().sum()/len(col)) ==1)], axis=1, inplace = True)
merge.drop(merge.columns[merge.apply(lambda col: (len(col.unique()) ==1))], axis=1, inplace = True)
merge['is_sar'] = merge.apply(lambda row: row['is_sar'] == True, axis=1).astype(int)
merge['prior_sar_count'] = merge.apply(lambda row: row['prior_sar_count'] == True, axis=1).astype(int)

#turn timestamps into numbers
merge['tran_timestamp'] = pd.to_datetime(merge['tran_timestamp'])
merge['tran_timestamp'] = merge['tran_timestamp'].apply(lambda x:x.toordinal())
merge.loc[:,~merge.columns.isin(['tran_id','tran_timestamp' ,'is_sar','prior_sar_count','bank_id'])] = StandardScaler().fit_transform(merge.loc[:,~merge.columns.isin(['tran_id','tran_timestamp' ,'is_sar','prior_sar_count','bank_id'])])
merge_dummy = pd.get_dummies(merge)

corrMatrix = merge_dummy.corr().round(2)
c = corrMatrix.abs()
upper_tri = c.where(np.triu(np.ones(c.shape),k=1).astype(np.bool))
print(upper_tri)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
to_drop.append('bank_id_bank_c')
print(); print(to_drop)

merge_dummy.drop(to_drop, axis=1, inplace =True)
import modAL
from functools import partial
from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
from sklearn.neighbors import KNeighborsClassifier
f1_results = {}
gmean_results = {}
auc_results = {}
BATCH_SIZE = 20
N_SAMPLES = 320
N_QUERIES = N_SAMPLES // BATCH_SIZE
preset_batch = partial(uncertainty_batch_sampling, n_instances=BATCH_SIZE)
knn = KNeighborsClassifier(n_neighbors= 10)
sv = svm.SVC(probability = True, class_weight= 'balanced')
df_split = np.array_split(merge_dummy, 10)

for i in range(len(df_split)):
  print("Processing iteration no.", i)
  #split dataset: 20% train, 10% test, 70% unlabel
  X_raw = df_split[i].loc[:,~df_split[i].columns.isin(['tran_id','is_sar','tran_timestamp','orig_acct', 'bene_acct'])]
  y_raw = df_split[i][['is_sar']]
  X_train, X_test, y_train, y_test = train_test_split(X_raw,y_raw,
                                                      train_size = 0.2,
                                                      test_size=0.1,
                                                      random_state=10,
                                                      shuffle = False)
  pool_index_X = list(X_train.index)+list(X_test.index)
  pool_index_y = list(y_train.index)+list(y_test.index)
  X_pool = X_raw[~X_raw.index.isin(pool_index_X)]
  y_pool = y_raw[~y_raw.index.isin(pool_index_y)]

  print(Counter(y_raw['is_sar']))
    # =============================================================================
  # Active learning
  # ============================================================================

  # creating the active learner
  learner = ActiveLearner(
      estimator=sv,
      X_training=np.array(X_train),
      y_training=np.ravel(np.array(y_train)),
      query_strategy=preset_batch
  )

  # pool where learner queries from
  X_pool = X_pool.to_numpy()
  y_pool = np.ravel(np.array(y_pool).astype(float))
  X_test = X_test.to_numpy()
  y_test = np.ravel(np.array(y_test).astype(float))

  raw_preds = learner.predict(X_raw)
  print(classification_report(y_raw, raw_preds))
  # Record our learner's score on the raw data.
  unqueried_f1_score = f1_score(y_raw, raw_preds)
  unqueried_auc_score = roc_auc_score(y_raw, raw_preds)
  unqueried_gmean = geometric_mean_score(y_raw, raw_preds)

  f1_history = [unqueried_f1_score]
  auc_history = [unqueried_auc_score]
  gmean_history = [unqueried_gmean]
  for index in range(N_QUERIES):
      query_index, query_instance = learner.query(X_pool)

      # Teach our ActiveLearner model the record it has requested.
      X, y = X_pool[query_index], y_pool[query_index]
      learner.teach(X=X, y=y)
      y_pred = learner.predict(X_test)
      # Remove the queried instance from the unlabeled pool.
      X_pool = np.delete(X_pool, query_index, axis=0)
      y_pool = np.delete(y_pool, query_index)

      # Calculate and report our model's accuracy.
      model_f1_score = f1_score(y_test, y_pred)
      model_auc = roc_auc_score(y_test, y_pred)
      model_gmean = geometric_mean_score(y_test, y_pred)
      print('F1 after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_f1_score))

      # Save our model's performance for plotting.
      f1_history.append(model_f1_score)
      auc_history.append(model_auc)
      gmean_history.append(model_gmean)
  f1_results['iteration_{0}'.format(i)]=f1_history
  auc_results['iteration_{0}'.format(i)]=auc_history
  gmean_results['iteration_{0}'.format(i)]=gmean_history

print ("Finished")

print(f1_results)
print(auc_results)
print(gmean_results)


import pickle
with open('rbm_loop_svc.pkl', 'wb') as file:
      
    # A new file will be created
    pickle.dump(f1_results, file)
    pickle.dump(auc_results, file)
    pickle.dump(gmean_results, file)
    
with open('rbm_loop_svc.pkl', 'rb') as file:
      
    # Call load method to deserialze
    var1= pickle.load(file)
    var2= pickle.load(file)
    var3= pickle.load(file)

f1_df = pd.DataFrame(f1_results).apply(pd.to_numeric)
auc_df = pd.DataFrame(auc_results).astype(int).apply(pd.to_numeric)
gm_df = pd.DataFrame(gmean_results).astype(int).apply(pd.to_numeric)

f1_df['average'] = f1_df.mean(axis=1)
auc_df['average'] = auc_df.mean(axis=1)
gm_df['average'] = gm_df.mean(axis=1)

f1_df.plot(y='average', kind='line', title = 'Average F1 scores among subsets',
           xlabel='Query number', ylim=(0,1), figsize=(10,7))
           
