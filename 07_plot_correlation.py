import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
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
parser.add_argument('--ovr', action='store_true',  default=False, help='Whether to get the one versus rest directory or classic decoding')
parser.add_argument('--regression', action='store_true',  default=False, help='Whether to get the regression decoding or classic decoding')
parser.add_argument('--slices', action='store_true',  default=False, help='Whether to make horizontal slice plots of single decoder')
parser.add_argument('--only_agg', action='store_true',  default=False, help='Do plot for each available condition, or just the aggregates plots with multiple conditions')
parser.add_argument('-r', '--remake', action='store_true',  default=False, help='recompute average again, even if plotting only aggregates')
parser.add_argument('-v', '--verbose', action='store_true',  default=False, help='Print more stuff')
parser.add_argument('--freq_bands', action='store_true',  default=False, help='whether to load frequency bands separately, or everything all at once (if you did not run multiple freq bands)')
parser.add_argument('--smooth_plot', default=0, type=int, help='Smoothing R before plotting')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

feat2feats = {"Shape": ['carre', 'cercle', 'triangle', 'ca', 'cl', 'tr'], "Colour": ['rouge', 'bleu', 'vert', 'vr', 'bl', 'rg']}

ylabel = "R"

print('This script lists all the .npy files in all the subjects decoding output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')
v = args.version
in_dir_name = f"Correlation_v{v}"
if args.subject in ["all", "v1", "v2",  "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/{in_dir_name}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/{in_dir_name}/{args.epochs_dir}/{args.subject}/"
# out_dir = f"{args.root_path}/Results/{in_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"
out_dir = f"{args.root_path}/Results/{in_dir_name}/{args.epochs_dir}/{args.subject}/"
print('\noutput files will be in: ' + out_dir)
if op.exists(out_dir): # warn and stop if args.overwrite is set to False
    print('output file already exists...')
    if args.overwrite:
        print('overwrite is set to True ... overwriting')
    else:
        print('overwrite is set to False ... exiting')
        exit()
else:
    print('Constructing output dirtectory: ', out_dir)
    os.makedirs(out_dir)

# list all .npy files in the directory
all_fns = natsorted(glob(in_dir + f'/*{ylabel}.npy'))
if not all_fns:
    raise RuntimeError(f"Did not find any R files in {in_dir}/*{ylabel}*.npy ... Did you pass the right config?")

mag_idx, grad_idx = [pickle.load(open(f"{args.root_path}/Data/{s}_indices.p", "rb")) for s in ['mag', 'grad']]
mag_info, grad_info = [pickle.load(open(f"{args.root_path}/Data/{s}_info.p", "rb")) for s in ['mag', 'grad']]

mirror_str = "mirror"
for cond in ["scenes"]: # only scenes should be computed "localizer", "obj", 
    for mirror in [True, False]: # mirror image or only same sentence.          
        for window in [True, False]: # windowed or whole epochs
            all_Rs_matched, all_Rs_mismatched = [], []
            for fn in all_fns:
                if not "winlength100_overlap10" in fn: continue
                if "Perf=1" in fn: continue
                if f"-{cond}-" not in fn: continue
                if mirror:
                    if "mirror" not in fn: continue
                else:
                    if "mirror" in fn: continue
                if window:
                    if "#" not in fn: continue
                    train_tmin, train_tmax = [float(s) for s in fn.split("#")[1].split(",")]
                else:
                    if "#" in fn: continue

                if args.verbose: print('loading file ', fn)
                r = np.load(fn)
                all_Rs_matched.append(r[:, 0])
                all_Rs_mismatched.append(r[:, 1])
                 
            if not all_Rs_matched: 
                if args.verbose: print(f"found no file for mirror {mirror}, cond: {cond}")
                continue
            if args.verbose: print(f"\nDoing {ylabel} trained on {cond}")
            n_subs = len(all_Rs_matched)
            if n_subs < 2: 
                print(f"Single subject found, moving on to next conditon")
                continue
            if n_subs > 30: 
                print(f"More than 30 subjects found, investigate")
                set_trace()

            if window:                
                win_str = f"{train_tmin}-{train_tmax}s"

            out_fn = f"{out_dir}/{cond}_{n_subs}ave"
            if mirror: out_fn += "_mirror"
            if window: out_fn += f"_{win_str}"

            R_mean_matched = np.nanmean(all_Rs_matched, 0)
            overall_R_mean_matched = np.mean(all_Rs_matched)
            R_std_matched = sem(all_Rs_matched, 0, nan_policy='omit')

            R_mean_mismatched = np.nanmean(all_Rs_mismatched, 0)
            overall_R_mean_mismatched = np.mean(all_Rs_mismatched)
            R_std_mismatched = sem(all_Rs_mismatched, 0, nan_policy='omit')

            overall_R_max_matched = np.max(all_Rs_matched)
            overall_R_max_mismatched = np.max(all_Rs_mismatched)

            overall_R_sum_matched = np.sum(all_Rs_matched)
            overall_R_sum_mismatched = np.sum(all_Rs_mismatched)

            print(f"for cond={cond} with mirror={mirror} and window={window} the overall mean R is {overall_R_mean_matched:.4f}, mismatched: {overall_R_mean_mismatched:.4f}")
            print(f"for cond={cond} with mirror={mirror} and window={window} the overall max R is {overall_R_max_matched:.4f}, mismatched: {overall_R_max_mismatched:.4f}")
            print(f"for cond={cond} with mirror={mirror} and window={window} the overall sum R is {overall_R_sum_matched:.4f}, mismatched: {overall_R_sum_mismatched:.4f}")
            print("\n")

            # plot_diag(data_mean, out_fn, train_cond, train_tmin, train_tmax, data_std=None, ylabel="R", ybar=0)
            data = np.array([R_mean_matched, R_mean_mismatched])
            data_std = np.array([R_std_matched, R_std_mismatched])
            tmin, tmax = tmin_tmax_dict["two_objects"]
            plot_multi_diag(data, out_fn, "scenes", tmin, tmax, data_std=data_std, ylabel="R", cmap_name='spring', labels=["matched", "mismatched"], plot_mean=False)
            plt.close('all')


print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.4f}min")
