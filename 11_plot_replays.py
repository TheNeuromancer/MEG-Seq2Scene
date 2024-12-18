import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import matplotlib.pyplot as plt

import mne
import numpy as np
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
from scipy.stats import sem
# from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import warnings
# warnings.filterwarnings('ignore', '.*Provided stat_fun.*', )
warnings.filterwarnings('ignore', '.*No clusters found.*', )

from utils.decod import *
from utils.params import *

matplotlib.rcParams.update({'font.size': 19})
matplotlib.rcParams.update({'lines.linewidth': 2})
plt.rcParams['figure.figsize'] = [12., 8.]
plt.rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser(description='MEG plotting of replay decoding results')
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


print('This script lists all the .npy files in all the subjects decoding output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')
v = args.version
decoding_dir = f"Decoding_ovr_v{v}" # if args.ovr else f"Decoding_v{v}"
if args.subject in ["all", "v1", "v2",  "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"
print('\noutput files will be in: ' + out_dir)
create_folder(out_dir, args.overwrite)

df_fn = 
all_fns = natsorted(glob(in_dir + f'/*preds.npy')) # list all preds.npy files in the directory
if not all_fns:
    raise RuntimeError(f"Did not find any preds files in {in_dir}/*preds.npy ... Did you pass the right config?")


## All possible training time (depends on the property that is decoded).
train_times = [".17", ".2", ".3", ".4", ".5", ".6", ".8"] + ["0.77", "0.8", "0.9", "1.0", "1.1", "1.2", "1.4"] + ["1.37", "1.4", "1.5", "1.6", "1.7", "1.8", "2.0"]
train_times = train_times + ["1.97", "2.0", "2.1", "2.2", "2.3", "2.4", "2.6"] + ["2.57", "2.6", "2.7", "2.8", "2.9", "3.0", "3.2"]
## Generalization window for objects and scenes
gen_windows = [(3, 5)] # (1.5, 2.2), 



maxLag = 35
theoretical_peak = 11 # in the paper the peak lag is at 110ms
n_states = 6
n_permutations = 1000
pval_th = 0.05 / maxLag
null = False
