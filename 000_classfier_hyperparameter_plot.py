import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import matplotlib.pyplot as plt

import mne
import numpy as np
from ipdb import set_trace
import pandas as pd
import argparse
import pickle
import time
import os.path as op
import os
import importlib
from glob import glob
from natsort import natsorted
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import sem, spearmanr
from prettytable import PrettyTable
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import warnings
# warnings.filterwarnings('ignore', '.*Provided stat_fun.*', )
warnings.filterwarnings('ignore', '.*No clusters found.*', )

from utils.decod import *
from utils.params import *

matplotlib.rcParams.update({'font.size': 19})
matplotlib.rcParams.update({'lines.linewidth': 2})
plt.rcParams['figure.figsize'] = [12., 8.]
plt.rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser(description='Plotting of decoding hyperparameter optimization results')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='all',help='subject name')
parser.add_argument('-o', '--out-dir', default='agg', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('-v', '--verbose', action='store_true',  default=False, help='Print more stuff')
args = parser.parse_args()

config = importlib.import_module(f"configs.{args.config}", "Config").Config() # import config parameters
for arg in vars(config): setattr(args, arg, getattr(config, arg)) # update argparse with arguments from the config
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

print('This script lists all the patterns.npy files in all the subjects decoding output directories, plots the patterns and compute the correlations between them')
v = args.version
decoding_dir = f"Decoding_opti_svc_v{v}" 
if args.subject in ["all", "v1", "v2",  "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"
print('\noutput files will be in: ' + out_dir)
create_folder(out_dir, args.overwrite)

# list all preds.npy files in the directory
all_perf_fns = natsorted(glob(in_dir + f'/*sub_perfs*.pkl'))
all_param_fns = natsorted(glob(in_dir + f'/*sub_params*.pkl'))
if not all_perf_fns:
    raise RuntimeError(f"Did not find any files in {in_dir}/*perf*.pkl ... Did you pass the right config?")

all_perfs, all_params = [], []
for perf_fn, param_fn in zip(all_perf_fns, all_param_fns): 
    
    perf = pickle.load(open(perf_fn, 'rb'))
    param = pickle.load(open(param_fn, 'rb'))

    all_perfs.append(perf)
    # all_params.append(param) # all the same! 
    
## AVERGAE OF SUBJECTS
ave_perfs = np.mean(all_perfs, 0) # average over subs
print(ave_perfs)
best_idx = np.argmax(ave_perfs)
print(f"Averaging first: Best set of params: {param[best_idx]}")
# param[i] = [0.1, 'l1', 'liblinear', 'balanced']

best_order = np.argsort(ave_perfs)[::-1]
for i, idx in enumerate(best_order[0:10]):
    print(f"{i+1}th best set of params: {param[idx]}: {ave_perfs[idx]}")
    ## For Logistic Regression:
# 1th best set of params: [0.1, 'l1', 'liblinear', 'balanced']: 0.5931095585264299
# 2th best set of params: [0.01, 'l2', 'liblinear', 'balanced']: 0.5925858331350647
# 3th best set of params: [0.01, 'l2', 'liblinear', None]: 0.5919073625262229
# 4th best set of params: [0.1, 'l1', 'saga', 'balanced']: 0.591786947328576
# 5th best set of params: [0.1, 'l1', 'liblinear', None]: 0.5912432240936122
# 6th best set of params: [0.01, 'l2', 'saga', 'balanced']: 0.5911589319398382
# 7th best set of params: [0.1, 'l2', 'liblinear', 'balanced']: 0.5904030581926304
# 8th best set of params: [0.01, 'l2', 'saga', None]: 0.5901056892744987
# 9th best set of params: [1, 'l1', 'liblinear', 'balanced']: 0.5900257662879338
# 10th best set of params: [0.1, 'l1', 'saga', None]: 0.5899671415891747

    ## For SVC - rbf    
# 1th best set of params: [1, 0.001, 'balanced']: 0.6067604331366996
# 2th best set of params: [1, 'auto', 'balanced']: 0.5997136904518987
# 3th best set of params: [0.1, 'auto', 'balanced']: 0.5969175750052083
# 4th best set of params: [10, 0.001, 'balanced']: 0.5919030989541929
# 5th best set of params: [0.1, 0.01, 'balanced']: 0.588482749765073
# 6th best set of params: [1, 'scale', 'balanced']: 0.5881632031879797
# 7th best set of params: [0.1, 0.001, 'balanced']: 0.5878032341272227
# 8th best set of params: [0.1, 'scale', 'balanced']: 0.5874950712171936
# 9th best set of params: [1, 0.01, 'balanced']: 0.5820310892356083
# 10th best set of params: [0.1, 0.01, None]: 0.578337220112099



print(f"Now getting the best set of params per subject and looking at the counts.")
best_idx_per_sub = [np.argmax(perf) for perf in all_perfs]
print(f"Getting best set per subject: Best set of params per subject: {[param[idx] for idx in best_idx_per_sub]}")
# [0.1, 'l1', 'liblinear', 'balanced']
# svc: 

# get the count of each param for everytime it is among the best param for a subject. 
each_best_subparam = [ [] for _ in range(len(param[0])) ]
for idx in best_idx_per_sub:
    for i, p in enumerate(param[idx]):
        each_best_subparam[i].append(p)

for i, subparams in enumerate(each_best_subparam):
    counts_each_subparam = [[p,subparams.count(p)] for p in set(subparams)]
    print(f"Param {i}:  was the best {counts_each_subparam}")
# Param 0:  was the best [[0.1, 12], [1, 4], [1000, 3], [0.01, 6], [0.001, 4]]
# Param 1:  was the best [['l2', 13], ['l1', 16]]
# Param 2:  was the best [['saga', 17], ['liblinear', 12]]
# Param 3:  was the best [[None, 11], ['balanced', 18]]
# -> [0.1, 'l1', 'saga', 'balanced']

# SVC
# Param 0:  was the best [[0.1, 4], [1, 21], [10, 3]]
# Param 1:  was the best [['scale', 2], [0.001, 20], ['auto', 6]]
# Param 2:  was the best [['balanced', 28]]
# -> [1, 0.001, 'balanced']



print(f"Now looking for subjects at chance performance for each set of params")
ave_perfs_per_sub = np.max(all_perfs, 1) # average over params

sub_order = np.argsort(ave_perfs_per_sub)
subs_sorted_by_perf = np.array(args.all_subjects)[sub_order]
sorted_perf = ave_perfs_per_sub[sub_order]

print(f"All subjects and their performance on this parameters selection localizer: ")
for sub, perf in zip(subs_sorted_by_perf, sorted_perf):
    print(f"{sub}: {perf}")
    
# 23: 0.5173213249930946
# 17: 0.5497465078956679
# 25: 0.5503156465087956
# 11_rb210035: 0.5509079095827337
# 27: 0.5562060680740726
# 19: 0.56196611800342
# 20: 0.5663579127803929
# 12_mb160165: 0.5727074541152787
# 08_ch180036: 0.5766474980707629
# 13_lg170436: 0.5848334315762261
# 10_ma200371: 0.5873206358154449
# 21: 0.5883026423218264
# 09_jl190711: 0.5903322318633565
# 22: 0.5951218971276956
# 15_ar160084: 0.5971313421772938
# 26: 0.5991257224502734
# 14_eb180237: 0.6023129980042915
# 24: 0.6039710162792119
# 16_er123987: 0.6125305686617339
# 29: 0.6137212558193843
# 03_cr170417: 0.6196655242791819
# 05_mb140004: 0.6244986531791007
# 01_js180232: 0.6262803714699136
# 02_jm100042: 0.6288747541239933
# 30: 0.6372399098303597
# 28: 0.6401424683440643
# 04_ag170045: 0.6403019046406014
# 07_jv200206: 0.6649413411831594
# 06_ll180197: 0.6861455107535671



## SVC
# 14_eb180237: 0.5652047174480727
# 26: 0.5694229714727501
# 10_ma200371: 0.5713200042839595
# 04_ag170045: 0.5719310425396158
# 20: 0.5731253931542809
# 15_ar160084: 0.5741904040046204
# 11_rb210035: 0.5757335869664915

# 19: 0.5839949652424169
# 02_jm100042: 0.5891524788205312
# 24: 0.5903088802192302
# 17: 0.5915093998610275
# 09_jl190711: 0.5950518540882688
# 03_cr170417: 0.5958303961499744
# 08_ch180036: 0.5999421838441332
# 22: 0.60773760844269
# 06_ll180197: 0.6080917150351764
# 23: 0.6094215850998284
# 25: 0.6197547715827887
# 01_js180232: 0.6320419555639881
# 28: 0.6325854643743123
# 16_er123987: 0.6329611809628811
# 13_lg170436: 0.6422326487995949
# 05_mb140004: 0.6426721291800059
# 12_mb160165: 0.6475168361775835
# 07_jv200206: 0.6535398257032942
# 27: 0.6537490043986997
# 29: 0.6563724380534477
# 21: 0.6685097436394463


print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")
