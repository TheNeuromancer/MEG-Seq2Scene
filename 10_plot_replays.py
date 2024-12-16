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
parser.add_argument('--ovr', action='store_true',  default=False, help='Whether to get the one versus rest directory or classic decoding')
parser.add_argument('-v', '--verbose', action='store_true',  default=False, help='Print more stuff')
parser.add_argument('--smooth_plot', default=0, type=int, help='Smoothing preds before plotting')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

feat2feats = {"Shape": ['carre', 'cercle', 'triangle', 'ca', 'cl', 'tr'], "Colour": ['rouge', 'bleu', 'vert', 'vr', 'bl', 'rg']}

# ylabel = "preds"
# ybar = 0.5 # vertical line
# print(ylabel)

print('This script lists all the .npy files in all the subjects decoding output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')
v = args.version
decoding_dir = f"Decoding_ovr_v{v}" if args.ovr else f"Decoding_v{v}"
if args.subject in ["all", "v1", "v2",  "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"

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

# list all preds.npy files in the directory
all_fns = natsorted(glob(in_dir + f'/*preds.npy'))
if not all_fns:
    raise RuntimeError(f"Did not find any preds files in {in_dir}/*preds.npy ... Did you pass the right config?")

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

dummy_class_enc = LabelEncoder()
dummy_labbin = LabelBinarizer()
# mag_idx, grad_idx = [pickle.load(open(f"{args.root_path}/Data/{s}_indices.p", "rb")) for s in ['mag', 'grad']]
# mag_info, grad_info = [pickle.load(open(f"{args.root_path}/Data/{s}_info.p", "rb")) for s in ['mag', 'grad']]

## All possible training time (depends on the property that is decoded).
train_times = [".17", ".2", ".3", ".4", ".5", ".6", ".8"] + ["0.77", "0.8", "0.9", "1.0", "1.1", "1.2", "1.4"] + ["1.37", "1.4", "1.5", "1.6", "1.7", "1.8", "2.0"]
train_times = train_times + ["1.97", "2.0", "2.1", "2.2", "2.3", "2.4", "2.6"] + ["2.57", "2.6", "2.7", "2.8", "2.9", "3.0", "3.2"]
# train_times = ["1.37", "1.5", "1.6", "1.7", "1.8"]
## Generalization window for objects and scenes
gen_windows = [(3, 5)] # (1.5, 2.2), 

preds_fn = f"{op.dirname(op.dirname(out_dir))}/all_preds.p"
diags_fn = f"{op.dirname(op.dirname(out_dir))}/all_diags.p"
patterns_fn = f"{op.dirname(op.dirname(out_dir))}/all_patterns.p"
confusions_fn = f"{op.dirname(op.dirname(out_dir))}/all_confusions.p"
all_labels = np.unique([op.basename(fn).split('-')[0] for fn in all_fns])
preds_all_labels, pattern_all_labels, confusion_all_labels = {}, {}, {}
all_df = []
for label in all_labels:
    if "Obj" in label:
        print("Skipping Objects for now")
        continue
    for train_cond in ["localizer", "obj", "scenes"]:
        for split_query in [False]: # no split query in replay decoding so far (but migh wanna include it later)
        # for split_query in ["match", "nonmatch", "flash", "noflash", "match_or_Error_type=l0", "match_or_Error_type=l1", \
        #                     "match_or_Error_type=l2", "Complexity=0", "Complexity=1", "Complexity=2", \
        #                     "Change.str.containsshape", "Change.str.containscolour", False]:
            for gen_cond in [None, "localizer", "obj", "scenes"]:
                for train_time in train_times:
                    for gen_window in gen_windows:
                        if args.verbose: print(train_time)
                        all_patterns, all_confusions, all_preds, all_subs, all_items = [], [], [], [], []
                        future_df = {}
                        for fn in all_fns:
                            # print("go ", end="")
                            if op.basename(fn)[0:len(label)+1] != f"{label}-": continue 
                            # print(f"label: {label}")
                            if f"cond-{train_cond}-" not in fn: continue
                            # print(f"train_cond: {train_cond}")
                            if not split_query: # if not split_query or nonmatch markers, keep all non-splitqueries
                                split_query_str = ""
                                if "_for_" in fn: continue
                            else:
                                if f"for_{split_query}" not in fn: continue
                                split_query_str = f"_for_{split_query}"
                            # print(f"split_query: {split_query}")
                            if gen_cond is not None:
                                if f"#{train_time},{train_time}#{gen_window[0]},{gen_window[1]}#" not in fn: 
                                    continue
                                if f"tested_on_{gen_cond}" not in fn: continue # only generalization results
                            else: # gen_cond is None 
                                if f"#{train_time},{train_time}#{train_time},{train_time}#" not in fn:
                                    continue
                                if "tested_on" in fn: continue # ensure we don't have generalization results (shouldn't be usefull after the preceeding line)
                            # do not load the full mnius splits
                            # print(f"gen_cond: {gen_cond}")
                            if "full_minus_split" in fn: continue

                            if args.verbose: print('loading file ', fn)
                            preds = np.load(fn) # times * trials * classes 
                            preds = preds.squeeze() # trials * classes (single time point for replay decoding)
                            # print(preds.shape)
                            all_preds.append(preds) # len(n_subs) of array of inhomogeneous shape n_trials * n_classes
                            all_subs.append(op.basename(op.dirname(fn))[0:2])
                            all_items.append(op.basename(fn))

                            future_df["train_time"] = [train_time] * len(preds)
                            future_df["gen_window"] = [gen_window] * len(preds)
                            future_df["train_cond"] = [train_cond] * len(preds)
                            future_df["gen_cond"] = [gen_cond] * len(preds)
                            future_df["split_query"] = [split_query] * len(preds)
                            future_df["label"] = [label] * len(preds)
                            future_df["preds"] = preds.tolist()
                            future_df["sub"] = [op.basename(op.dirname(fn))[0:2]] * len(preds)

                            # load corresponding pattern and confusion matrix (for direct training only)
                            if not split_query and (gen_cond is None): 
                                # pattern_fn = f"{op.dirname(fn)}/{'_'.join(op.basename(fn).split('_')[0:2])}_best_pattern_t*.npy"
                                pattern_fn = f"{op.dirname(fn)}/{op.basename(fn).replace('_preds.npy', '')}_best_pattern_t*.npy"
                                pattern_fn = glob(pattern_fn)
                                if len(pattern_fn) == 0:
                                    if args.verbose: print(f"!!!No pattern found!!! passing for now but look it up")
                                elif len(pattern_fn) > 1:
                                    if args.verbose: print(f"!!!Multiple patterns found!!! passing for now but look it up")
                                else:
                                    all_patterns.append(np.load(pattern_fn[0]).squeeze())

                                confusion_fn = f"{op.dirname(fn)}/{op.basename(fn).replace('_preds.npy', '')}_confusions.npy"
                                confusion_fn = glob(confusion_fn)
                                if len(confusion_fn) == 0:
                                    if args.verbose: print(f"!!!No confusion found!!! passing for now but look it up")
                                elif len(confusion_fn) > 1:
                                    if args.verbose: print(f"!!!Multiple confusions found!!! passing for now but look it up")
                                else:
                                    all_confusions.append(np.load(confusion_fn[0]).squeeze())

                        if not all_preds: 
                            if args.verbose: print(f"found no file for {label} trained on {train_cond} with generalization to {gen_cond} for  split query {split_query}, train time {train_time}, gen window {gen_window}, continuing")
                            continue
                        if args.verbose: print(f"\nDoing {label} trained on {train_cond} with generalization {gen_cond} for  split query {split_query}")
                        n_subs = len(all_preds)
                        if n_subs < 2: 
                            print(f"Single subject found, moving on to next conditon")
                            continue
                        if n_subs > 30: 
                            set_trace()
                        gen_str = f"_tested_on_{gen_cond}" if gen_cond is not None else ""
                        out_fn = f"{out_dir}/{label}_trained_on_{train_cond}{gen_str}{split_query_str}_{n_subs}ave"

                        if not len(all_preds): 
                            print(f"did find any pred for {label} trained on {train_cond} with generalization {gen_cond} for  split query {split_query}, continuing")
                            continue

                        # pattern and confusions
                        if not split_query and (gen_cond is None):
                            if len(all_patterns) == 0: 
                                print(f"Not a single pattern found ...") 
                            else:
                                pattern = np.median(all_patterns, 0)
                                if pattern.ndim == 2: # OVR, one additional dimension
                                    pattern = np.median(pattern, 0)
                                    all_patterns = np.concatenate(all_patterns) # first dim = subjs*classes
                                # pattern_all_labels[f"{label}_{train_cond}"] = pattern # store values for all labels for multi plot

                            if len(all_confusions) == 0: 
                                print(f"Not a single confusion found ...") 
                            else:
                                all_confusions = np.array(all_confusions)
                                mean_confusion = np.nanmean(all_confusions, 0)
                                median_confusion = np.median(all_confusions, 0)
                                # if confusion.ndim == 2: # OVR, one additional dimension
                                #     confusion = np.median(confusion, 0)
                                #     all_confusions = np.concatenate(all_confusions) # first dim = subjs*classes
                                # confusion_all_labels[f"{label}_{train_cond}"] = mean_confusion # store values for all labels for multi plot

                              ## plotting confusion matrices
                                # fig, axes = plt.subplots(len(all_confusions), figsize=(6, len(all_confusions)/2))
                                # for i_p, patt in enumerate(all_confusions):
                                #     try:
                                #         mne.viz.plot_topomap(np.squeeze(patt)[mag_idx], mag_info, axes=axes[i_p])
                                #     except:
                                #         set_trace()
                                # plt.savefig(f'{out_fn}_{label}_{train_cond}_confusionS_mag.png')
                                ## ave confusion
                                # mne.viz.plot_topomap(np.squeeze(np.mean(all_confusions, 0))[mag_idx], mag_info, axes=ax)
                                # from ipdb import set_trace; set_trace()
                                labels = shapes if "S" in label else colors if "C" in label else ['w1', 'w2', 'w3']
                                fig, ax = plt.subplots()
                                plt.imshow(mean_confusion, cmap='viridis', origin='lower', vmin=0.1, vmax=.15)
                                ax.set(xticks=[0,1,2], xticklabels=labels, yticks=[0,1,2], yticklabels=labels)
                                plt.colorbar()
                                plt.savefig(f'{out_fn}_mean_confusion_{train_time}s.png')
                                plt.close()
                                fig, ax = plt.subplots()
                                plt.imshow(median_confusion, cmap='viridis', origin='lower', vmin=0.1, vmax=.15)
                                ax.set(xticks=[0,1,2], xticklabels=labels, yticks=[0,1,2], yticklabels=labels)
                                plt.colorbar()
                                plt.savefig(f'{out_fn}_median_confusion_{train_time}s.png')
                                plt.close()


                        # store values for all labels for multi plot
                        all_df.append(pd.DataFrame(future_df))
                        preds_all_labels[f"{label}_{train_cond}_{gen_cond}_{split_query_str}"] = all_preds

                        # ## plotting all patterns
                        # if len(all_patterns):
                        #     # fig, axes = plt.subplots(len(all_patterns), figsize=(6, len(all_patterns)/2))
                        #     # for i_p, patt in enumerate(all_patterns):
                        #     #     try:
                        #     #         mne.viz.plot_topomap(np.squeeze(patt)[mag_idx], mag_info, axes=axes[i_p])
                        #     #     except:
                        #     #         set_trace()
                        #     # plt.savefig(f'{out_fn}_{label}_{train_cond}_patternS_mag.png')
                        #     ## ave pattern
                        #     fig, ax = plt.subplots()
                        #     mne.viz.plot_topomap(np.squeeze(np.mean(all_patterns, 0))[mag_idx], mag_info, axes=ax)
                        #     plt.savefig(f'{out_fn}_{label}_{train_cond}_ave_pattern_mag.png')
                        #     plt.close()

                        if args.verbose: print(f"Finished {label} trained on {train_cond} with generalization {gen_cond}  for  split query {split_query}\n")
                        plt.close('all')


    # print(f"saving all data to {preds_fn} and {diags_fn}")
    # pickle.dump(preds_all_labels, open(preds_fn, "wb"))
    # diag_preds_all_labels = {k: np.array([np.diag(x) for x in v]) for k, v in preds_all_labels.items()}
    # pickle.dump(diag_preds_all_labels, open(diags_fn, "wb"))
    # # pickle.dump(pattern_all_labels, open(patterns_fn, "wb"))

from ipdb import set_trace; set_trace()
df = pd.concat(all_df)
df.to_csv(f"{out_dir}/all_preds_data.csv", index=False)

print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")
