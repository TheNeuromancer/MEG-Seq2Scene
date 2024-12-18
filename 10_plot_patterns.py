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
from scipy.stats import sem
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

parser = argparse.ArgumentParser(description='MEG ans SEEG plotting of decoding results')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='all',help='subject name')
parser.add_argument('-o', '--out-dir', default='agg', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
# parser.add_argument('--ovr', action='store_true',  default=False, help='Whether to get the one versus rest directory or classic decoding')
parser.add_argument('-v', '--verbose', action='store_true',  default=False, help='Print more stuff')
parser.add_argument('--smooth_plot', default=0, type=int, help='Smoothing preds before plotting')
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

# list all preds.npy files in the directory
all_fns = natsorted(glob(in_dir + f'/*patterns.npy'))
if not all_fns:
    raise RuntimeError(f"Did not find any patterns files in {in_dir}/*patterns.npy ... Did you pass the right config?")
print(all_fns)

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
    all_fns = [fn for fn in all_fns if not op.basename(op.dirname(fn))[0:2] in bad_subjects]
    version = "v2"
elif int(args.subject[0:2]) < 9:
    version = "v1"
elif int(args.subject[0:2]) > 8:
    version = "v2"
else:
    qwe

mag_idx, grad_idx, all_idx = [pickle.load(open(f"{args.root_path}/Data/{s}_indices.p", "rb")) for s in ['mag', 'grad', 'all']]
mag_info, grad_info, all_info = [pickle.load(open(f"{args.root_path}/Data/{s}_info.p", "rb")) for s in ['mag', 'grad', 'all']]


def plot_patterns(pattern, out_fn, mag_info, mag_idx, grad_info, grad_idx):
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(pattern[mag_idx], mag_info, axes=ax, contours=0)
    plt.savefig(f'{out_fn}_pattern_mag.png')
    plt.close()
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(pattern[grad_idx], grad_info, axes=ax, contours=0)
    plt.savefig(f'{out_fn}_pattern_grad.png')
    plt.close()

    # vmin, vcenter, vmax = np.min(pattern), 0, np.max(pattern)
    # plot_single_ch_perf(pattern[mag_idx], mag_info, f"{out_fn}_pattern_mag_.png", cmap_name='bwr', vmin=vmin, vcenter=vcenter, vmax=vmax, score_label='Weight', title=None, ticksize=14)
    # plot_single_ch_perf(pattern[grad_idx], grad_info, f"{out_fn}_pattern_grad_.png", cmap_name='bwr', vmin=vmin, vcenter=vcenter, vmax=vmax, score_label='Weight', title=None, ticksize=14)


## All possible training time (depends on the property that is decoded).
train_times = [".17", ".2", ".3", ".4", ".5", ".6", ".8"] + ["0.77", "0.8", "0.9", "1.0", "1.1", "1.2", "1.4"] + ["1.37", "1.4", "1.5", "1.6", "1.7", "1.8", "2.0"]
train_times = train_times + ["1.97", "2.0", "2.1", "2.2", "2.3", "2.4", "2.6"] + ["2.57", "2.6", "2.7", "2.8", "2.9", "3.0", "3.2"]
## Generalization window for objects and scenes

patterns_fn = f"{op.dirname(op.dirname(out_dir))}/all_patterns.p"
# confusions_fn = f"{op.dirname(op.dirname(out_dir))}/all_confusions.p"
all_labels = np.unique([op.basename(fn).split('-')[0] for fn in all_fns])
all_df = []
for label in all_labels:
    if args.verbose: print(f"Doing {label}")
    if "Obj" in label:
        print("Skipping Objects for now")
        continue
    for train_cond in ["localizer", "obj", "scenes"]:
        gen_cond = None
        for train_time in train_times:
            if args.verbose: print(train_time)
            all_patterns = []
            future_df = {}
            for fn in all_fns:
                if op.basename(fn)[0:len(label)+1] != f"{label}-": continue 
                if f"cond-{train_cond}-" not in fn: continue
                if "tested_on" in fn: continue # ensure we don't have generalization results (shouldn't be usefull after the following line)
                if train_time not in fn:
                    continue
                if f"#{train_time},{train_time}#{train_time},{train_time}#" not in fn:
                    continue

                if args.verbose: print('loading file ', fn)

                # load corresponding pattern and confusion matrix (for direct training only)
                # pattern_fn = f"{op.dirname(fn)}/{'_'.join(op.basename(fn).split('_')[0:2])}_best_pattern_t*.npy"
                # pattern_fn = f"{op.dirname(fn)}/{op.basename(fn).replace('_preds.npy', '')}_best_pattern_t*.npy"
                # pattern_fn = glob(pattern_fn)
                # if len(pattern_fn) == 0:
                #     if args.verbose: print(f"!!!No pattern found!!! passing for now but look it up")
                # elif len(pattern_fn) > 1:
                #     if args.verbose: print(f"!!!Multiple patterns found!!! passing for now but look it up")
                # else:
                pattern = np.load(fn) # .squeeze()
                all_patterns.append(pattern)
                
                future_df['pattern'] = [pattern]
                future_df['subject'] = [op.basename(op.dirname(fn))[0:2]]
                future_df['train_cond'] = [train_cond]
                future_df['train_time'] = [train_time]
                future_df['label'] = [label]
                all_df.append(pd.DataFrame(future_df))


            n_subs = len(all_patterns)
            if not n_subs: 
                if args.verbose: print(f"Not a single pattern found for {label} trained on {train_cond} at {train_time} ...") 
                continue
            else:
                median_pattern = np.median(all_patterns, 0) # median over subjects
            out_fn = f"{out_dir}/{label}_trained_on_{train_cond}_at_{train_time}_{n_subs}ave"

            if median_pattern.ndim == 2: # OVR, one additional dimension n_classes  * n_sensors
                for patt in median_pattern:
                    plot_patterns(patt, out_fn, mag_info, mag_idx, grad_info, grad_idx)
            else:
                plot_patterns(median_pattern, out_fn, mag_info, mag_idx, grad_info, grad_idx)

            if args.verbose: print(f"Finished {label} trained on {train_cond} at {train_time}\n")
            plt.close('all')

                # pattern_all_labels[f"{label}_{train_cond}"] = pattern # store values for all labels for multi plot

    # print(f"saving all data to {preds_fn} and {diags_fn}")
    # pickle.dump(preds_all_labels, open(preds_fn, "wb"))
    # diag_preds_all_labels = {k: np.array([np.diag(x) for x in v]) for k, v in preds_all_labels.items()}
    # pickle.dump(diag_preds_all_labels, open(diags_fn, "wb"))
    # # pickle.dump(pattern_all_labels, open(patterns_fn, "wb"))

from ipdb import set_trace; set_trace()
df = pd.concat(all_df)
df.to_csv(f"{out_dir}/all_patterns.csv", index=False)

print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")
