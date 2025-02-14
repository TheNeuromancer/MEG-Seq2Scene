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
decoding_dir = f"Decoding_opti_v{v}" 
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




print(f"Now getting the best set of params per subject and looking at the counts.")
best_idx_per_sub = [np.argmax(perf) for perf in all_perfs]
print(f"Getting best set per subject: Best set of params per subject: {[param[idx] for idx in best_idx_per_sub]}")
# [0.1, 'l1', 'liblinear', 'balanced']

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




print(f"Now looking for subjects at chance performance for each set of params")
ave_perfs_per_sub = np.mean(all_perfs, 1) # average over params

sub_order = np.argsort(ave_perfs_per_sub)
subs_sorted_by_perf = np.array(args.all_subjects)[sub_order]
sorted_perf = ave_perfs_per_sub[sub_order]

print(f"All subjects and their performance on this parameters selection localizer: ")
for sub, perf in zip(subs_sorted_by_perf, sorted_perf):
    print(f"{sub}: {perf}")

# 23: 0.5033565790831263
# 27: 0.5280319012834223
# 25: 0.5343305785730027
# 17: 0.5399329372520107
# 11_rb210035: 0.5400056092571954
# 19: 0.5484838659729154
# 13_lg170436: 0.555663993059934
# 20: 0.55624901219134
# 12_mb160165: 0.5591907541483082
# 08_ch180036: 0.5634584422821743
# 09_jl190711: 0.5650072195652127
# 10_ma200371: 0.5680052509565987
# 21: 0.5747929484606267
# 22: 0.5752464996085099
# 14_eb180237: 0.5825786749074876
# 26: 0.5827757230896715
# 15_ar160084: 0.5843682377107433
# 24: 0.5870446976104883
# 03_cr170417: 0.5929521447654755
# 29: 0.5933942189363027
# 16_er123987: 0.593992864152255
# 05_mb140004: 0.5943475789683631
# 01_js180232: 0.6001821088924023
# 02_jm100042: 0.6083286773644885
# 28: 0.6155559343765409
# 30: 0.6201727257593787
# 04_ag170045: 0.6233743541387133
# 07_jv200206: 0.6318877633288904
# 06_ll180197: 0.6424396325867822

print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")
