import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import matplotlib.pyplot as plt

import mne
import numpy as np
from ipdb import set_trace
import argparse
import pickle
import time
import os.path as op
import os
import importlib
from glob import glob
from natsort import natsorted
from itertools import permutations, combinations
import pandas as pd
from copy import deepcopy

from utils.decod import *
from utils.params import *

matplotlib.rcParams.update({'font.size': 19})
matplotlib.rcParams.update({'lines.linewidth': 2})
plt.rcParams['figure.figsize'] = [12., 8.]
plt.rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser(description='MEG plotting of dimensionality analysis results')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='all',help='subject name')
parser.add_argument('-o', '--out-dir', default='agg', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--plot_all_subs', action='store_true',  default=False, help='Whether to plot all subjects separately, or just averages')
parser.add_argument('--smooth_plot', default=0, type=int, help='Smoothing data before plotting')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

feat2feats = {"Shape": ['carre', 'cercle', 'triangle', 'ca', 'cl', 'tr'], "Colour": ['rouge', 'bleu', 'vert', 'vr', 'bl', 'rg']}

print('This script lists all the decoding window results files in all the subjects decoding output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')
_dir = f"Dimensionality_v{args.version}"
if args.subject in ["all", "v1", "v2",  "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/{_dir}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/{_dir}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/{_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"

print('\noutput files will be in: ' + out_dir)

if op.exists(out_dir): # warn and stop if args.overwrite is set to False
    print('output file already exists...')
    if args.overwrite:
        print('overwrite is set to True ... overwriting')
    else:
        print('overwrite is set to False ... exiting')
        exit()
else:
    print('Constructing output dirtectory: ', out_dir)
    os.makedirs(out_dir)

# list all .npy files in the directory
all_fns = natsorted(glob(in_dir + '/*.npy'))
print(f"Found {len(all_fns)} files")

# keep the first 8 subjects for the 1st version, all the remaining for v2
if args.subject == "v1":
    all_fns = [fn for fn in all_fns if int(op.basename(op.dirname(fn))[0:2]) < 9]
    version = "v1"
elif args.subject == "v2":
    all_fns = [fn for fn in all_fns if int(op.basename(op.dirname(fn))[0:2]) > 8]
    version = "v2"
elif args.subject == "all":
    version = "v2"
elif args.subject == "goods":
    all_fns = [fn for fn in all_fns if not op.basename(op.dirname(fn))[0:2] in bad_subs_dim]
    version = "v2"
elif int(args.subject[0:2]) < 9:
    version = "v1"
elif int(args.subject[0:2]) > 8:
    version = "v2"
else:
    qwe

## PR plots
ths = [.3, 0.5, .52, .54, .525, .55, .575, .6]
ylabel = "Participation Ratio"
all_means, all_stds = {}, {}
for cond in ["obj", "scenes"]:
    for complexity in (0, 1, 2, None): # PR over time plots
        for th in ths:
            print(f"_{th}th_")
            data, all_subs = [], []
            for fn in all_fns:
                if "reconstruction_L2" in fn: continue # reject reconstruction accuracy files
                if complexity is not None and not (f"Complexity={complexity}" in fn): continue
                if complexity is None and "Complexity" in fn: continue
                if f"cond-{cond}-" not in fn: continue
                if f"_{th}th_" not in fn: continue
                print(f"Loading {fn}")
                data.append(np.load(fn))
                all_subs.append(op.basename(op.dirname(fn))[0:2])

            if not data:
                print(f"did not find any data for complexity = {complexity}") # for mirror={mirror} and window={window}, and reducdim={reducdim}, moving on")
                continue
            
            print(f"Doing  complexity={complexity} with {len(all_subs)} files")
            data = np.array(data)
            complexity_str = f"_complexity={complexity}"
            n_subs = len(all_subs)
            tmin, tmax = tmin_tmax_dict[cond]
            n_times = data.shape[1]
            times = np.linspace(tmin, tmax, data.shape[1])

            # all_subjects
            out_fn = f"{out_dir}/{cond}_all_{len(data)}{complexity_str}_{th}th_PR"
            # , data_std=None
            plot_multi_diag(data, out_fn, cond, tmin, tmax, ylabel="PR", contrast=False, version=version, 
                            cmap_name='hsv', cmap_groups=[], labels=[])

            out_fn = f"{out_dir}/{cond}_{len(data)}ave{complexity_str}_{th}th_PR"
            dat_mean, dat_std = np.mean(data, 0), np.std(data, 0)
            plot_diag(data_mean=dat_mean, data_std=dat_std, out_fn=out_fn, train_cond=cond, ybar=None, resplock=False, contrast=False, 
                      train_tmin=tmin, train_tmax=tmax, ylabel=ylabel, version=version, window=None, smooth_plot=args.smooth_plot)
            all_means[f"Complexity={complexity}_ave_{th}th"] = dat_mean
            all_stds[f"Complexity={complexity}_ave_{th}th"] = dat_std

            out_fn = f"{out_dir}/{cond}_{len(data)}med{complexity_str}_{th}th_PR"
            dat_median = np.median(data, 0)
            plot_diag(data_mean=dat_median, data_std=dat_std, out_fn=out_fn, train_cond=cond, ybar=None, resplock=False, contrast=False, 
                      train_tmin=tmin, train_tmax=tmax, ylabel=ylabel, version=version, window=None, smooth_plot=args.smooth_plot)
            all_means[f"Complexity={complexity}_med_{th}th"] = dat_median
            all_stds[f"Complexity={complexity}_med_{th}th"] = dat_std

            out_fn = f"{out_dir}/{cond}_{len(data)}ave{complexity_str}_{th}th_PR_baselined"
            baselined_data = mne.baseline.rescale(data, times, baseline=(tmin, None))
            baselined_data_ave = np.mean(baselined_data, 0)
            plot_diag(data_mean=baselined_data_ave, data_std=dat_std, out_fn=out_fn, train_cond=cond, ybar=None, resplock=False, contrast=False, 
                      train_tmin=tmin, train_tmax=tmax, ylabel=ylabel, version=version, window=None, smooth_plot=args.smooth_plot)
            all_means[f"Complexity={complexity}_baselined_{th}th"] = baselined_data_ave
            all_stds[f"Complexity={complexity}_baselined_{th}th"] = dat_std

            
            if args.plot_all_subs:
                for sub, dat in zip(all_subs, data):
                    out_fn = f"{out_dir}/{cond}_{sub}{complexity_str}_{th}th_PR"
                    plot_diag(data_mean=dat, data_std=None, out_fn=out_fn, train_cond=cond, ybar=None, resplock=False, contrast=False, 
                          train_tmin=tmin, train_tmax=tmax, ylabel=ylabel, version=version, window=None, smooth_plot=args.smooth_plot)


## Aggregate PR plots
# n_times = dat_mean.shape[0]
# times = np.linspace(tmin, tmax, n_times)
labels = ["Complexity=0", "Complexity=1", "Complexity=2"]
for th in ths:
    try:
        for cond in ["ave", "med", "baselined"]:
            fig, ax = plt.subplots()
            for label in labels:
                lab = f"{label}_{cond}_{th}th"
                plt.plot(times, all_means[lab], label=label)
                ax.fill_between(times, all_means[lab]-all_stds[lab], all_means[lab]+all_stds[lab], alpha=0.2)
            plt.legend()
            plt.savefig(f"{out_dir}/{len(data)}_{cond}_{th}th_all_complexity_PR.png")
            plt.close("all")
    except:
        pass


# ## Reconstruction L2 plots
# ylabel = "Mean reconstruction error (L2)"
# data, all_subs = [], []
# for fn in all_fns:
#     if not "reconstruction_L2" in fn: continue # reject reconstruction accuracy files
#     data.append(np.load(fn))
#     all_subs.append(op.basename(op.dirname(fn))[0:2])
# if not data:
#     print(f"did not find any data for L2") # for mirror={mirror} and window={window}, and reducdim={reducdim}, moving on")
#     qwe
# data = np.array(data)
# print(data.shape)
# n_subs = len(all_subs)

# dat_mean, dat_std = np.mean(data, 0), np.std(data, 0)
# labels = ["Complexity=0", "Complexity=1", "Complexity=2"]
# for i in range(dat_mean.shape[1]):
#     out_fn = f"{out_dir}/{len(data)}ave_{labels[i]}_reconstruction_L2"
#     plot_diag(data_mean=dat_mean[:,i], data_std=dat_std[:,i], out_fn=out_fn, train_cond=cond, ybar=None, resplock=False, contrast=False, 
#               train_tmin=tmin, train_tmax=tmax, ylabel=ylabel, version=version, window=None, smooth_plot=args.smooth_plot)

# n_times = dat_mean.shape[0]
# times = np.linspace(tmin, tmax, n_times)
# fig, ax = plt.subplots()
# labels = ["Complexity=0", "Complexity=1", "Complexity=2"]
# for i in range(dat_mean.shape[1]):
#     plt.plot(times, dat_mean[:,i], label=labels[i])
#     ax.fill_between(times, dat_mean[:,i]-dat_std[:,i], dat_mean[:,i]+dat_std[:,i], alpha=0.2)
# plt.legend()
# plt.savefig(f"{out_dir}/{len(data)}_all_complexity_reconstruction_l2.png")


# set_trace()
print("ALL DONE")
