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
from prettytable import PrettyTable
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

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
parser.add_argument('-v', '--verbose', action='store_true',  default=False, help='Print more stuff')
parser.add_argument('--save_loc', action='store_true',  default=False, help='Save results as localizer')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

feat2feats = {"Shape": ['carre', 'cercle', 'triangle', 'ca', 'cl', 'tr'], "Colour": ['rouge', 'bleu', 'vert', 'vr', 'bl', 'rg']}


print('This script lists all the .npy files in all the subjects decoding output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')
decoding_dir = f"Decoding_single_ch_v{args.version}"
if args.subject in ["all", "v1", "v2",  "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"

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
all_fns = natsorted(glob(in_dir + '/*AUC*.npy'))
if not all_fns:
    raise RuntimeError(f"Did not find any AUC files in {in_dir}/*AUC*.npy ... Did you pass the right config?")

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

all_labels = np.unique([op.basename(fn).split('-')[0] for fn in all_fns])
max_auc_all_facconds = []
all_faccond_names = []
ave_auc_all_labels = {}
med_auc_all_labels = {}
std_auc_all_labels = {}
for label in all_labels:
    for train_cond in ["localizer", "obj", "scenes"]: # , "one_object", "two_objects", 
        for split_query in ["match", "nonmatch", "flash", "noflash", False]:
            for gen_cond in [None, "localizer", "obj", "scenes"]: # , "one_object", "two_objects", 
                # if train_cond == gen_cond: continue # skip when we both have one object, it is not generalization
                all_AUC = []
                all_subs = []
                all_items = []
                for fn in all_fns:
                    if op.basename(fn)[0:len(label)+1] != f"{label}-": continue 
                    if f"-{train_cond}-" not in fn: continue
                    if not split_query: # if not split_query or nonmatch markers, keep all non-splitqueries
                            split_query_str = ""
                            if "_for_" in fn: continue
                    else:
                        if f"for_{split_query}" not in fn: continue
                        split_query_str = f"_for_{split_query}"
                        # raise RuntimeError("Matching issue")
                    if gen_cond is not None:
                        if f"tested_on_{gen_cond}" not in fn: continue
                    else: # ensure we don't have generalization results
                        if "tested_on" in fn: continue
                    # do not load the full mnius splits
                    if "full_minus_split" in fn: continue

                    if args.verbose: print('loading file ', fn)
                    all_AUC.append(np.load(fn))
                    all_subs.append(op.basename(op.dirname(fn))[0:2])
                    all_items.append(op.basename(fn))

                if not all_AUC: 
                    print(f"found no file for {label} trained on {train_cond} with generalization to {gen_cond} {f'for split query {split_query}' if split_query else ''}")
                    continue
                if args.verbose: print(f"\nDoing {label} trained on {train_cond} with generalization {gen_cond} {f'for split query {split_query}' if split_query else ''}")
                gen_str = f"_tested_on_{gen_cond}" if gen_cond is not None else ""
                out_fn = f"{out_dir}/{label}_trained_on_{train_cond}{gen_str}{split_query_str}_{len(all_AUC)}ave"

                if args.save_loc:
                    AUC_dict = {s: auc for s, auc in zip(all_subs, all_AUC)}
                    localizer_out_dir = op.dirname(op.dirname(op.dirname(op.dirname(op.dirname(out_dir)))))
                    localizer_out_fn = f"{localizer_out_dir}/Localizer/{label}_trained_on_{train_cond}{gen_str}{split_query_str}.p"
                    pickle.dump(AUC_dict, open(localizer_out_fn, "wb"))

                all_subs_labels = np.unique(all_subs)
                all_subs = dummy_class_enc.fit_transform(all_subs)
                all_items = dummy_class_enc.fit_transform(all_items)

                # average per subject, then accross subjects
                all_AUC = np.array(all_AUC)
                AUC_mean = np.mean(all_AUC, 0)
                AUC_med = np.median(all_AUC, 0)
                AUC_std = np.std(all_AUC, 0)

                # store values for all labels for multi plot
                if train_cond=="scenes" and gen_cond is None and "win" not in label:
                    ave_auc_all_labels[label] = np.diag(AUC_mean)
                    med_auc_all_labels[label] = np.diag(AUC_med)
                    std_auc_all_labels[label] = np.diag(AUC_std)

                # save max values to save to csv
                max_auc_all_facconds.append(np.max(AUC_mean))
                all_faccond_names.append(f"{label} trained on {train_cond}{' tested on ' if gen_cond is not None else ''}{gen_cond if gen_cond is not None else ''}")

                ## get epochs info
                # print("CHECK THAT ALL EPOCHS INFO ARE THE SAME, and that the fb is correct")
                epo_info = pickle.load(open(fn.replace('AUC_diag.npy', 'epo_info.p'), "rb"))
                
                 
                ylabel = get_ylabel_from_fn(fn)
                is_contrast = True if (np.min(np.array(all_AUC)) < 0) or (np.max(np.array(all_AUC)) < .4) else False

                # get vmin and vmax
                # vmin, vmax = np.min(AUC_mean), np.max(AUC_mean)
                vmin = 0.45
                vmax = .55 if AUC_mean.max()<.55 else .6 if AUC_mean.max()<.6 else .7 if AUC_mean.max()<.7 else .8 if AUC_mean.max()<.8 else 1
                title = f"{op.basename(out_fn).split(str(len(all_AUC))+'ave')[0]} - Max AUC = {AUC_mean.max():.3f}".replace("_", " ")
                plot_single_ch_perf(AUC_mean, epo_info, f"{out_fn}_mean.png", title=title, vmin=vmin, vmax=vmax)
                
                # vmin, vmax = np.min(AUC_med), np.max(AUC_med)
                vmax = .55 if AUC_mean.max()<.55 else .6 if AUC_mean.max()<.6 else .7 if AUC_mean.max()<.7 else .8 if AUC_mean.max()<.8 else 1
                title = f"{op.basename(out_fn).split(str(len(all_AUC))+'ave')[0]} - Max AUC = {AUC_med.max():.3f}".replace("_", " ")
                plot_single_ch_perf(AUC_med, epo_info, f"{out_fn}_median.png", title=title, vmin=vmin, vmax=vmax)

                # if gen_cond is None or "win" in label:
                #     # try:
                #         plot_diag(data_mean=AUC_mean, data_std=AUC_std, out_fn=out_fn, train_cond=train_cond, 
                #             train_tmin=train_tmin, train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast, version=version, window=window)

                # plot_GAT(data_mean=AUC_mean, out_fn=out_fn, train_cond=train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, 
                #          test_tmax=test_tmax, ylabel=ylabel, contrast=is_contrast, gen_cond=gen_cond, slices=slices, version=version, window=window)

                print(f"Finished {label} trained on {train_cond} with generalization {gen_cond} {f'for split query {split_query}' if split_query else ''}\n")
                plt.close('all')
