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
parser.add_argument('-s', '--subject', default='01_js180232',help='subject name')
parser.add_argument('-o', '--out-dir', default='agg', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--ovr', action='store_true',  default=False, help='Whether to get the one versus rest directory or classic decoding')
parser.add_argument('--only_agg', action='store_true',  default=False, help='Do plot for each available condition, or just the aggregates plots with multiple conditions')
parser.add_argument('-r', '--remake', action='store_true',  default=False, help='recompute average again, even if plotting only aggregates')
parser.add_argument('-v', '--verbose', action='store_true',  default=False, help='Print more stuff')
parser.add_argument('--smooth_plot', default=0, type=int, help='Smoothing AUC before plotting')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

feat2feats = {"Shape": ['carre', 'cercle', 'triangle', 'ca', 'cl', 'tr'], "Colour": ['rouge', 'bleu', 'vert', 'vr', 'bl', 'rg']}

ylabel = "AUC"
ybar = 0.5 # vertical line
print(ylabel)

print('This script lists all the .npy files in all the subjects decoding output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')
v = args.version
decoding_dir = f"Decoding_ovr_v{v}" if args.ovr else f"Decoding_v{v}"
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
all_fns = natsorted(glob(in_dir + f'/*{ylabel}.npy'))
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
# mag_idx, grad_idx = [pickle.load(open(f"{args.root_path}/Data/{s}_indices.p", "rb")) for s in ['mag', 'grad']]
# mag_info, grad_info = [pickle.load(open(f"{args.root_path}/Data/{s}_info.p", "rb")) for s in ['mag', 'grad']]

## All possible training time (depends on the property that is decoded).
train_times = [.17, .2, .3, .4, .5, .6, .8] + [0.77, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4] + [1.37, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
train_times = train_times + [1.97, 2.0, 2.1, 2.2, 2.3, 2.4, 2.6] + [2.57, 2.6, 2.7, 2.8, 2.9, 3.0, 3.2]
## Generalization window for objects and scenes
gen_windows = [(1.5, 2.2), (3, 5)]

data_fn = f"{op.dirname(op.dirname(out_dir))}/all_data.p"
diags_fn = f"{op.dirname(op.dirname(out_dir))}/all_diags.p"
patterns_fn = f"{op.dirname(op.dirname(out_dir))}/all_patterns.p"
confusions_fn = f"{op.dirname(op.dirname(out_dir))}/all_confusions.p"
if args.only_agg and op.exists(data_fn) and not args.remake:
    print(f"skipping loading each results and directly loading diag aggregates from {diags_fn}")
    diag_auc_all_labels = pickle.load(open(diags_fn, "rb"))
    pattern_all_labels = pickle.load(open(patterns_fn, "rb"))
    # auc_all_labels = pickle.load(open(data_fn, "rb"))
else:
    all_labels = np.unique([op.basename(fn).split('-')[0] for fn in all_fns])
    if args.verbose: print(f"all labels: {all_labels}")
    max_auc_all_facconds = []
    all_faccond_names = []
    auc_all_labels = {}
    pattern_all_labels = {}
    confusion_all_labels = {}
    for label in all_labels:
        if "Obj" in label:
            print("Skipping Objects for now")
            continue
        for train_cond in ["localizer", "obj", "scenes"]:
            for split_query in ["match", "nonmatch", "flash", "noflash", "match_or_Error_type=l0", "match_or_Error_type=l1", \
                                "match_or_Error_type=l2", "Complexity=0", "Complexity=1", "Complexity=2", \
                                "Change.str.containsshape", "Change.str.containscolour", False]:
                for gen_cond in [None, "localizer", "obj", "scenes"]:
                    for train_time in train_times:
                        for gen_window in gen_windows:
                            if args.verbose: print(train_time)
                            all_patterns = []
                            all_confusions = []
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
                                if gen_cond is not None:
                                    if f"#{train_time},{train_time}#{gen_window[0]},{gen_window[1]}#" not in fn: continue
                                    if f"tested_on_{gen_cond}" not in fn: continue # only generalization results
                                    # if 
                                else:
                                    if f"#{train_time},{train_time}#{train_time},{train_time}#" not in fn: continue
                                    print(fn)
                                    print(f"#{train_time},{train_time}#{train_time},{train_time}#")
                                    from ipdb import set_trace; set_trace()
                                    if "tested_on" in fn: continue # ensure we don't have generalization results (shouldn't be usefull after the preceeding line)
                                # do not load the full mnius splits
                                if "full_minus_split" in fn: continue

                                if args.verbose: print('loading file ', fn)
                                AUC = np.load(fn)
                                all_AUC.append(AUC)
                                all_subs.append(op.basename(op.dirname(fn))[0:2])
                                all_items.append(op.basename(fn))

                                # load corresponding pattern and confusion matrix (for direct training only)
                                if not split_query and (gen_cond is None): 
                                    # pattern_fn = f"{op.dirname(fn)}/{'_'.join(op.basename(fn).split('_')[0:2])}_best_pattern_t*.npy"
                                    pattern_fn = f"{op.dirname(fn)}/{op.basename(fn).replace('_AUC.npy', '')}_best_pattern_t*.npy"
                                    pattern_fn = glob(pattern_fn)
                                    if len(pattern_fn) == 0:
                                        print(f"!!!No pattern found!!! passing for now but look it up")
                                    elif len(pattern_fn) > 1:
                                        print(f"!!!Multiple patterns found!!! passing for now but look it up")
                                    else:
                                        all_patterns.append(np.load(pattern_fn[0]))

                                    confusion_fn = f"{op.dirname(fn)}/{op.basename(fn).replace('_AUC.npy', '')}_confusions.npy"
                                    confusion_fn = glob(confusion_fn)
                                    if len(confusion_fn) == 0:
                                        print(f"!!!No confusion found!!! passing for now but look it up")
                                    elif len(confusion_fn) > 1:
                                        print(f"!!!Multiple confusions found!!! passing for now but look it up")
                                    else:
                                        all_confusions.append(np.load(confusion_fn[0]))


                            if not all_AUC: 
                                if args.verbose: print(f"found no file for {label} trained on {train_cond} with generalization to {gen_cond} for  split query {split_query}, train time {train_time}, gen window {gen_window}, continuing")
                                continue
                            if args.verbose: print(f"\nDoing {label} trained on {train_cond} with generalization {gen_cond} for  split query {split_query}")
                            n_subs = len(all_AUC)
                            if n_subs < 2: 
                                print(f"Single subject found, moving on to next conditon")
                                continue
                            if n_subs > 30: 
                                set_trace()
                            gen_str = f"_tested_on_{gen_cond}" if gen_cond is not None else ""
                            out_fn = f"{out_dir}/{label}_trained_on_{train_cond}{gen_str}{split_query_str}_{n_subs}ave"

                            all_subs_labels = np.unique(all_subs)
                            all_subs = dummy_class_enc.fit_transform(all_subs)
                            all_items = dummy_class_enc.fit_transform(all_items)

                            # average per subject, then accross subjects
                            # if not np.all([all_AUC[0].shape == auc.shape for auc in all_AUC]):
                            #     print(f"Shape mismatch ... rejecting one but look it up")
                            #     # set_trace()
                            #     shape = [226, 226] if train_cond=='obj' else [426, 426]
                            #     all_AUC = [auc for auc in all_AUC if auc.shape == shape]
                            if not len(all_AUC): 
                                print(f"did find any auc for {label} trained on {train_cond} with generalization {gen_cond} for  split query {split_query}, continuing")
                                continue
                            all_AUC = np.array(all_AUC)
                            try:
                                # AUC_mean = np.nanmean(all_AUC, 0)
                                AUC_mean = np.nanmedian(all_AUC, 0)
                            except:
                                print(f"Shape mismatch ... continuing for now but look it up")
                                continue
                                set_trace()
                            AUC_std = sem(all_AUC, 0, nan_policy='omit')

                            # pattern and confusions
                            if not split_query and (gen_cond is None):
                                if len(all_patterns) == 0: 
                                    print(f"Not a single pattern found ...") 
                                else:
                                    pattern = np.median(all_patterns, 0)
                                    if pattern.ndim == 2: # OVR, one additional dimension
                                        pattern = np.median(pattern, 0)
                                        all_patterns = np.concatenate(all_patterns) # first dim = subjs*classes
                                    pattern_all_labels[f"{label}_{train_cond}"] = pattern # store values for all labels for multi plot

                                if len(all_confusions) == 0: 
                                    print(f"Not a single confusion found ...") 
                                else:
                                    all_confusions = np.array(all_confusions)
                                    confusion = np.nanmean(all_confusions, 0)
                                    # from ipdb import set_trace; set_trace()
                                    # if confusion.ndim == 2: # OVR, one additional dimension
                                    #     confusion = np.median(confusion, 0)
                                    #     all_confusions = np.concatenate(all_confusions) # first dim = subjs*classes
                                    confusion_all_labels[f"{label}_{train_cond}"] = confusion # store values for all labels for multi plot

                            # store values for all labels for multi plot
                            # if train_cond=="scenes": # and gen_cond is None and "win" not in label:
                            auc_all_labels[f"{label}_{train_cond}_{gen_cond}_{split_query_str}"] = np.array(all_AUC)

                            # save max values to save to csv
                            max_auc_all_facconds.append(np.max(AUC_mean))
                            all_faccond_names.append(f"{label} trained on {train_cond}{' tested on ' if gen_cond is not None else ''}{gen_cond if gen_cond is not None else ''}")

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

                            if args.only_agg: continue # skip per condition plots, just save results for aggregate plots


                            ## Plots per condition
                            if not "win" in label and not "delay" in label: # remove mismatch side when you tun wth win in the label
                                window = False
                                train_tmin, train_tmax = tmin_tmax_dict[train_cond]
                                if gen_cond is not None:
                                    test_tmin, test_tmax = tmin_tmax_dict[gen_cond]
                                else:
                                    test_tmin, test_tmax = train_tmin, train_tmax
                            else: # windowed
                                print(fn)
                                train_tmin, train_tmax = [float(s) for s in fn.split("#")[1].split(",")]
                                test_tmin, test_tmax = [float(s) for s in fn.split("#")[2].split(",")]
                                window = True

                            is_resplock = False
                            times = np.arange(train_tmin, train_tmax+1e-10, 1./args.sfreq)

                            # ## plotting confusion matrices
                            if len(all_confusions) and gen_cond is None:
                                # fig, axes = plt.subplots(len(all_confusions), figsize=(6, len(all_confusions)/2))
                                # for i_p, patt in enumerate(all_confusions):
                                #     try:
                                #         mne.viz.plot_topomap(np.squeeze(patt)[mag_idx], mag_info, axes=axes[i_p])
                                #     except:
                                #         set_trace()
                                # plt.savefig(f'{out_fn}_{label}_{train_cond}_confusionS_mag.png')
                                ## ave confusion
                                fig, ax = plt.subplots()
                                # mne.viz.plot_topomap(np.squeeze(np.mean(all_confusions, 0))[mag_idx], mag_info, axes=ax)
                                from ipdb import set_trace; set_trace()
                                plt.imshow(confusion, cmap='viridis', origin='lower')
                                plt.colorbar()
                                plt.savefig(f'{out_fn}_{label}_{train_cond}_ave_confusion_{t}s.png')
                                plt.close()

                            is_contrast = True if (np.min(np.array(all_AUC)) < 0) or (np.max(np.array(all_AUC)) < .4) else False

                            if gen_cond is not None: # or "win" in label:
                            # if train_tmax==test_tmax or "win" in label: # square time window, necessary for getting a diagonal
                                plot_diag(data_mean=AUC_mean, data_std=AUC_std, out_fn=out_fn, train_cond=train_cond, ybar=ybar, resplock=is_resplock,
                                    train_tmin=train_tmin, train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast, version=version, window=window, smooth_plot=args.smooth_plot)
                                
                                ## plot diags for all subjects
                                # plot_multi_diag(data=all_AUC, data_std=None, out_fn=out_fn, train_cond=train_cond, train_tmin=train_tmin, 
                                #                 train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast, version=version)
                                ## each sub + average of each subjects
                                all_subs_bin = dummy_labbin.fit_transform(all_subs).T.astype(bool)
                                sub_ave_AUC = np.array([np.mean(all_AUC[indices], 0) for indices in all_subs_bin])
                                # sub_std_AUC = np.array([np.std(all_AUC[indices], 0) for indices in all_subs_bin])
                                plot_multi_diag(data=sub_ave_AUC, data_std=None, out_fn=f"{out_fn}_subave", train_cond=train_cond, train_tmin=train_tmin, 
                                                train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast, version=version) #, labels=all_subs_labels)

                                # # same     for each subject
                                # plot_multi_diag(data=all_AUC, data_std=None, out_fn=f"{out_fn}_colsub", train_cond=train_cond, train_tmin=train_tmin, 
                                #                 train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast, version=version, cmap_groups=all_subs)
                                # # same color for each training condition
                                # plot_multi_diag(data=all_AUC, data_std=None, out_fn=f"{out_fn}_colitem", train_cond=train_cond, train_tmin=train_tmin, 
                                #                 train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast, version=version, cmap_groups=all_items)
                                # ## plots diags for the average of training condition - implement get label for this?
                                # all_items_bin = dummy_labbin.fit_transform(all_items).T.astype(bool)
                                # item_ave_AUC = np.array([np.mean(all_AUC[indices], 0) for indices in all_items_bin])
                                # # item_std_AUC = np.array([np.std(all_AUC[indices], 0) for indices in all_items_bin])
                                # plot_multi_diag(data=item_ave_AUC, data_std=None, out_fn=f"{out_fn}_condave", train_cond=train_cond, train_tmin=train_tmin, 
                                #                 train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast, version=version)

                            if args.verbose: print(f"Finished {label} trained on {train_cond} with generalization {gen_cond}  for  split query {split_query}\n")
                            plt.close('all')


    print(f"saving all data to {data_fn} and {diags_fn}")
    pickle.dump(auc_all_labels, open(data_fn, "wb"))
    diag_auc_all_labels = {k: np.array([np.diag(x) for x in v]) for k, v in auc_all_labels.items()}
    pickle.dump(diag_auc_all_labels, open(diags_fn, "wb"))
    # pickle.dump(pattern_all_labels, open(patterns_fn, "wb"))

## all basic decoders diagonals on the same plot
tmin, tmax = tmin_tmax_dict["scenes"]
times = np.arange(tmin, tmax+1e-10, 1./args.sfreq)
word_onsets, image_onset = get_onsets("scenes", version=version) # , resplock=resplock

# for label, pattern in pattern_all_labels.items():
#     fig = mne.viz.plot_topomap(pattern[mag_idx], mag_info, contours=0)
#     plt.savefig(f'{out_dir}/{label}_pattern_mag.png')
#     plt.close('all')
#     fig = mne.viz.plot_topomap(pattern[grad_idx], grad_info, contours=0)
#     plt.savefig(f'{out_dir}/{label}_pattern_grad.png')
#     plt.close('all')
# # from ipdb import set_trace; set_trace()   

print(diag_auc_all_labels.keys())

# only basics
labels = ['S1_scenes_None_', 'C1_scenes_None_', 'R_scenes_None_', 'S2_scenes_None_', 'C2_scenes_None_']
joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_basic.png', word_onsets=word_onsets, image_onset=image_onset)
labels = ['Flash_scenes_None_', 'Matching_scenes_None_', 'Button_scenes_None_', 'Perf_scenes_None_'] # 'SameObj_scenes_None_', 
joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, y_inc=.1, out_fn=f'{out_dir}/scenes_joyplot_basic2.png', word_onsets=word_onsets, image_onset=image_onset)

# # Shape gen
# labels = ['S1_scenes_None_', 'S2_scenes_None_', 'S1_scenes_scenes_', 'S2_scenes_scenes_']
# joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_shape_gen.png', word_onsets=word_onsets, image_onset=image_onset)

# extended
labels = ['S1_scenes_None_', 'C1_scenes_None_', 'R_scenes_None_', 'S2_scenes_None_', 'C2_scenes_None_', 'Flash_scenes_None_', 'Matching_scenes_None_', 'Button_scenes_None_', 'Perf_scenes_None_']
joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_extended.png', word_onsets=word_onsets, image_onset=image_onset)



labels = ['PropMismatch_scenes_None_', 'BindMismatch_scenes_None_', 'RelMismatch_scenes_None_']
joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, tmin=4.5, times=times, out_fn=f'{out_dir}/scenes_joyplot_mismatches.png', word_onsets=word_onsets, image_onset=image_onset)



# object properties
tmin, tmax = tmin_tmax_dict["obj"]
times = np.arange(tmin, tmax+1e-10, 1./args.sfreq)
word_onsets, image_onset = get_onsets("obj", version=version)
labels = ['S_obj_None_', 'C_obj_None_'] #, 'CMismatch_obj_None_', 'SMismatch_obj_None_'] # ] #, 'AllObj_obj_None_',
joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, tmax=4, times=times, out_fn=f'{out_dir}/scenes_joyplot_obj_props.png', word_onsets=word_onsets, image_onset=image_onset)
labels = ['SMismatch_obj_None_', 'CMismatch_obj_None_']
joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, tmin=2, tmax=4, times=times, out_fn=f'{out_dir}/scenes_joyplot_obj_mismatches.png', word_onsets=word_onsets, image_onset=image_onset)

# object properties gen to scenes
labels = ['S_0_obj_scenes_', 'S_1_obj_scenes_', 'C_0_obj_scenes_', 'C_1_obj_scenes_']
joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_obj_gen.png', word_onsets=word_onsets, image_onset=image_onset)


print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")
