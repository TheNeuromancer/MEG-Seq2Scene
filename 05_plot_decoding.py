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
from mne.stats import permutation_cluster_1samp_test

from utils.decod import *
from utils.params import *

matplotlib.rcParams.update({'font.size': 19})
matplotlib.rcParams.update({'lines.linewidth': 2})
plt.rcParams['figure.figsize'] = [12., 8.]
plt.rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser(description='MEG ans SEEG plotting of decoding results')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='01_js180232',help='subject name')
parser.add_argument('-o', '--out-dir', default='agg', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--slices', action='store_true',  default=False, help='Whether to make horizontal slice plots of single decoder')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

feat2feats = {"Shape": ['carre', 'cercle', 'triangle', 'ca', 'cl', 'tr'], "Colour": ['rouge', 'bleu', 'vert', 'vr', 'bl', 'rg']}

if args.slices:
    slices_loc = [0.17, 0.3, 0.43, 0.5, 0.72, 0.85]
    slices_one_obj = [0.2, 0.85, 1.1, 2.5, 2.75]
else:
    slices = []


print('This script lists all the .npy files in all the subjects decoding output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')

if args.subject == "all":
    in_dir = f"{args.root_path}/Results/Decoding_v{args.version}/{args.epochs_dir}/0*/"
else:
    in_dir = f"{args.root_path}/Results/Decoding_v{args.version}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/Decoding_v{args.version}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"

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
all_filenames = glob(in_dir + '/*AUC*.npy')

report = mne.report.Report()


all_labels = np.unique([op.basename(fn).split('-')[0] for fn in all_filenames])

for label in all_labels:
    for train_cond in ["localizer", "one_object", "two_objects"]:
        if args.slices:
            if train_cond in ["localizer"]: slices = slices_loc
            if train_cond == "one_object": slices = slices_one_obj
        for match in [True, False]:
            for gen_cond in [None, "localizer", "one_object", "two_objects"]:
                # if train_cond == gen_cond: continue # skip when we both have one object, it is not generalization
                all_AUC = []
                for fn in all_filenames:
                    if op.basename(fn)[0:len(label)+1] != f"{label}-": continue 
                    if f"-{train_cond}-" not in fn: continue
                    if match: # keep only the files where we tested on matching trials
                        if "match" not in fn: continue
                        match_str = "_match"
                    else: # do not keep these trials
                        if "match" in fn: continue
                        match_str = ""
                    if gen_cond is not None:
                        if f"tested_on_{gen_cond}" not in fn: continue
                    else: # ensure we don't have generalization results
                        if "tested_on" in fn: continue

                    print('loading file ', fn)
                    all_AUC.append(np.load(fn))

                if not all_AUC: 
                    # print(f"found no file for {label} trained on {train_cond} with generalization to {gen_cond} {'matching trials only' if match else ''}")
                    continue
                print(f"\nDoing {label} trained on {train_cond} with generalization {gen_cond} {'matching trials only' if match else ''}")
                gen_str = f"_tested_on_{gen_cond}" if gen_cond is not None else ""
                out_fn = f"{out_dir}/{label}_trained_on_{train_cond}{gen_str}{match_str}"
                all_AUC = np.array(all_AUC)
                AUC_mean = np.mean(all_AUC, 0)
                AUC_std = np.std(all_AUC, 0)

                
                train_tmin, train_tmax = tmin_tmax_dict[train_cond]
                if gen_cond is not None:
                    test_tmin, test_tmax = tmin_tmax_dict[gen_cond]
                else:
                    test_tmin, test_tmax = train_tmin, train_tmax
                
                n_times_train = AUC_mean.shape[0]
                n_times_test = AUC_mean.shape[1]
                times_train = np.linspace(train_tmin, train_tmax, n_times_train)
                times_test = np.linspace(test_tmin, test_tmax, n_times_test)
                
                ylabel = get_ylabel_from_fn(fn)
                is_contrast = True if (np.min(np.array(all_AUC)) < 0) or (np.max(np.array(all_AUC)) < .4) else False

                if gen_cond is None:
                    plot_diag(data_mean=AUC_mean, data_std=AUC_std, out_fn=out_fn, train_cond=train_cond, 
                        train_tmin=train_tmin, train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast)


                plot_GAT(data_mean=AUC_mean, out_fn=out_fn, train_cond=train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, 
                         test_tmax=test_tmax, ylabel=ylabel, contrast=is_contrast, gen_cond=gen_cond, slices=slices)

                print(f"Finished {label} trained on {train_cond} with generalization {gen_cond} {'matching trials only' if match else ''}\n")
                plt.close('all')
