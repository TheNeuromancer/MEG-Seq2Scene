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
parser.add_argument('--slices', action='store_true',  default=False, help='Whether to make horizontal slice plots of single decoder')
parser.add_argument('--only_agg', action='store_true',  default=False, help='Do plot for each available condition, or just the aggregates plots with multiple conditions')
parser.add_argument('-r', '--remake', action='store_true',  default=False, help='recompute average again, even if plotting only aggregates')
parser.add_argument('-v', '--verbose', action='store_true',  default=False, help='Print more stuff')
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
decoding_dir = f"Decoding_ovr_v{args.version}" if args.ovr else f"Decoding_v{args.version}"
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
mag_idx, grad_idx = [pickle.load(open(f"{args.root_path}/Data/{s}_indices.p", "rb")) for s in ['mag', 'grad']]
mag_info, grad_info = [pickle.load(open(f"{args.root_path}/Data/{s}_info.p", "rb")) for s in ['mag', 'grad']]


data_fn = f"{op.dirname(op.dirname(out_dir))}/all_data.p"
diags_fn = f"{op.dirname(op.dirname(out_dir))}/all_diags.p"
patterns_fn = f"{op.dirname(op.dirname(out_dir))}/all_patterns.p"
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
    for label in all_labels:
        for train_cond in ["localizer", "obj", "scenes"]: # , "one_object", "two_objects", 
            if args.slices:
                if train_cond in ["localizer"]: slices = slices_loc
                if train_cond == "one_object": slices = slices_one_obj
            for split_query in ["match", "nonmatch", "flash", "noflash", False]:
                for gen_cond in [None, "localizer", "obj", "scenes"]: # , "one_object", "two_objects", 
                    # if train_cond == gen_cond: continue # skip when we both have one object, it is not generalization
                    all_patterns = []
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
                        # else:
                        #     raise RuntimeError("split queries issue")
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

                        # load corresponding pattern (for direct training only)
                        if not split_query and (gen_cond is None): 
                            # pattern_fn = f"{op.dirname(fn)}/{'_'.join(op.basename(fn).split('_')[0:2])}_best_pattern_t*.npy"
                            pattern_fn = f"{op.dirname(fn)}/{op.basename(fn).replace('_AUC.npy', '')}_best_pattern_t*.npy"
                            pattern_fn = glob(pattern_fn)
                            if len(pattern_fn) != 1:
                                print(f"!!!Multiple patterns found!!! passing for now but look it up")
                                # set_trace()
                            all_patterns.append(np.load(pattern_fn[0]))
                            

                    if not all_AUC: 
                        if args.verbose: print(f"found no file for {label} trained on {train_cond} with generalization to {gen_cond} for  split query {split_query}")
                        continue
                    if args.verbose: print(f"\nDoing {label} trained on {train_cond} with generalization {gen_cond} for  split query {split_query}")
                    gen_str = f"_tested_on_{gen_cond}" if gen_cond is not None else ""
                    out_fn = f"{out_dir}/{label}_trained_on_{train_cond}{gen_str}{split_query_str}_{len(all_AUC)}ave"

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
                        AUC_mean = np.nanmean(all_AUC, 0)
                    except:
                        print(f"Shape mismatch ... continuing for now but look it up")
                        continue
                        set_trace()
                    AUC_std = sem(all_AUC, 0, nan_policy='omit')
                    if not split_query and (gen_cond is None): 
                        pattern = np.median(all_patterns, 0)
                        if pattern.ndim == 2: # OVR, one additional dimension
                            pattern = np.median(pattern, 0)
                            all_patterns = np.concatenate(all_patterns) # first dim = subjs*classes

                    # store values for all labels for multi plot
                    # if train_cond=="scenes": # and gen_cond is None and "win" not in label:
                    auc_all_labels[f"{label}_{train_cond}_{gen_cond}_{split_query_str}"] = np.array(all_AUC)
                    if not split_query and (gen_cond is None): 
                        pattern_all_labels[f"{label}_{train_cond}"] = pattern

                    # save max values to save to csv
                    max_auc_all_facconds.append(np.max(AUC_mean))
                    all_faccond_names.append(f"{label} trained on {train_cond}{' tested on ' if gen_cond is not None else ''}{gen_cond if gen_cond is not None else ''}")

                    # ## plotting all patterns
                    # if len(all_patterns):
                    #     fig, axes = plt.subplots(len(all_patterns), figsize=(6, len(all_patterns)/2))
                    #     for i_p, patt in enumerate(all_patterns):
                    #         try:
                    #             mne.viz.plot_topomap(np.squeeze(patt)[mag_idx], mag_info, axes=axes[i_p])
                    #         except:
                    #             set_trace()
                    #     plt.savefig(f'{out_fn}_{label}_{train_cond}_patternS_mag.png')
                    #     plt.close()

                    if args.only_agg: continue # skip per condition plots, just save results for aggregate plots


                    ## Plots per condition
                    if not "win" in label and not "delay" in label:
                        window = False
                        train_tmin, train_tmax = tmin_tmax_dict[train_cond]
                        if gen_cond is not None:
                            test_tmin, test_tmax = tmin_tmax_dict[gen_cond]
                        else:
                            test_tmin, test_tmax = train_tmin, train_tmax
                    else: # windowed
                        train_tmin, train_tmax = [float(s) for s in fn.split("#")[1].split(",")]
                        test_tmin, test_tmax = [float(s) for s in fn.split("#")[2].split(",")]
                        window = True
                    
                    ylabel = get_ylabel_from_fn(fn)
                    is_contrast = True if (np.min(np.array(all_AUC)) < 0) or (np.max(np.array(all_AUC)) < .4) else False

                    if gen_cond is None or "win" in label:
                        plot_diag(data_mean=AUC_mean, data_std=AUC_std, out_fn=out_fn, train_cond=train_cond, 
                            train_tmin=train_tmin, train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast, version=version, window=window)
                        
                        # ## plot diags for all subjects
                        # # plot_multi_diag(data=all_AUC, data_std=None, out_fn=out_fn, train_cond=train_cond, train_tmin=train_tmin, 
                        # #                 train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast, version=version)
                        # ## each sub + average of each subjects
                        # all_subs_bin = dummy_labbin.fit_transform(all_subs).T.astype(bool)
                        # sub_ave_AUC = np.array([np.mean(all_AUC[indices], 0) for indices in all_subs_bin])
                        # # sub_std_AUC = np.array([np.std(all_AUC[indices], 0) for indices in all_subs_bin])
                        # plot_multi_diag(data=sub_ave_AUC, data_std=None, out_fn=f"{out_fn}_subave", train_cond=train_cond, train_tmin=train_tmin, 
                        #                 train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast, version=version) #, labels=all_subs_labels)

                        # # same color for each subject
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

                    if not window:
                        plot_GAT(data_mean=AUC_mean, out_fn=out_fn, train_cond=train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, 
                                 test_tmax=test_tmax, ylabel=ylabel, contrast=is_contrast, gen_cond=gen_cond, slices=slices, version=version, window=window)

                    if args.verbose: print(f"Finished {label} trained on {train_cond} with generalization {gen_cond}  for  split query {split_query}\n")
                    plt.close('all')


    print(f"saving all data to {data_fn} and {diags_fn}")
    pickle.dump(auc_all_labels, open(data_fn, "wb"))
    diag_auc_all_labels = {k: np.array([np.diag(x) for x in v]) for k, v in auc_all_labels.items()}
    pickle.dump(diag_auc_all_labels, open(diags_fn, "wb"))
    pickle.dump(pattern_all_labels, open(patterns_fn, "wb"))

## all basic decoders diagonals on the same plot
tmin, tmax = tmin_tmax_dict["scenes"]
times = np.arange(tmin, tmax+1e-10, 1./args.sfreq)
word_onsets, image_onset = get_onsets("scenes", version=version)

# for label, pattern in pattern_all_labels.items():
#     fig = mne.viz.plot_topomap(pattern[mag_idx], mag_info, contours=0)
#     plt.savefig(f'{out_dir}/{label}_pattern_mag.png')
#     plt.close('all')
#     fig = mne.viz.plot_topomap(pattern[grad_idx], grad_info, contours=0)
#     plt.savefig(f'{out_dir}/{label}_pattern_grad.png')
#     plt.close('all')
# # from ipdb import set_trace; set_trace()   

print(diag_auc_all_labels.keys())

if args.ovr:
    # mismatch type
    labels = ['Matching_scenes_None_', 'PropMismatch_scenes_None_', 'BindMismatch_scenes_None_', 'RelMismatch_scenes_None_'] #, 'Button_scenes_None_', 'Perf_scenes_None_']
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_mismatches.png', tmin=4.5, word_onsets=word_onsets, image_onset=image_onset)

    # only basics
    labels = ['S1_scenes_None_', 'C1_scenes_None_', 'R_scenes_None_', 'S2_scenes_None_', 'C2_scenes_None_']
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_basic.png', word_onsets=word_onsets, image_onset=image_onset)
    labels = ['Flash_scenes_None_', 'Matching_scenes_None_', 'Button_scenes_None_', 'Perf_scenes_None_'] # 'SameObj_scenes_None_', 
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_basic2.png', word_onsets=word_onsets, image_onset=image_onset)

    # # Shape gen
    # labels = ['S1_scenes_None_', 'S2_scenes_None_', 'S1_scenes_scenes_', 'S2_scenes_scenes_']
    # joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_shape_gen.png', word_onsets=word_onsets, image_onset=image_onset)

    # extended
    labels = ['S1_scenes_None_', 'C1_scenes_None_', 'R_scenes_None_', 'S2_scenes_None_', 'C2_scenes_None_', 'Flash_scenes_None_', 'Matching_scenes_None_', 'Button_scenes_None_', 'Perf_scenes_None_']
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_extended.png', word_onsets=word_onsets, image_onset=image_onset)



    # object properties and gen
    tmin, tmax = tmin_tmax_dict["obj"]
    times = np.arange(tmin, tmax+1e-10, 1./args.sfreq)
    word_onsets, image_onset = get_onsets("obj", version=version)
    labels = ['S_obj_None_', 'C_obj_None_'] #, 'AllObj_obj_None_', 'CMismatch_obj_None_', 'SMismatch_obj_None_']
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, tmax=4, times=times, out_fn=f'{out_dir}/scenes_joyplot_obj_props.png', word_onsets=word_onsets, image_onset=image_onset)
    # labels = ['S_obj_None_', 'S_0_obj_scenes_', 'S_1_obj_scenes_', 'C_obj_None_', 'C_0_obj_scenes_', 'C_1_obj_scenes_']
    # joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, tmax=4, times=times, out_fn=f'{out_dir}/scenes_joyplot_obj_prop_and_gen.png', word_onsets=word_onsets, image_onset=image_onset)

    # object properties gen to scenes
    labels = ['S_0_obj_scenes_', 'S_1_obj_scenes_', 'C_0_obj_scenes_', 'C_1_obj_scenes_']
    # set_trace()
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_obj_gen.png', word_onsets=word_onsets, image_onset=image_onset)

else:
    labels = ['SameC_scenes_None_', 'SameS_scenes_None_', 'SameC_0_scenes_scenes_', 'SameS_0_scenes_scenes_', 'SameObj_scenes_None_']
    # labels = ['SameC_scenes_None_', 'SameS_scenes_None_', 'SameObj_scenes_None_']
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_complexity.png', word_onsets=word_onsets, image_onset=image_onset)

    labels = ['PropMismatch_scenes_None_', 'BindMismatch_scenes_None_', 'RelMismatch_scenes_None_']
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, tmin=4.5, times=times, out_fn=f'{out_dir}/scenes_joyplot_mismatches.png', word_onsets=word_onsets, image_onset=image_onset)





# joyplot_with_stats(data_dict=diag_auc_all_labels, labels=diag_auc_all_labels.keys(), times=times, out_fn=f'{out_dir}/scenes_joyplot_all.png', word_onsets=word_onsets, image_onset=image_onset) # ['S1','C1','R','S2','C2','All1stObj','All2ndObj']

# # all properties
# plot_all_props_multi(ave_dict=ave_auc_all_labels, std_dict=std_auc_all_labels, labels=ave_auc_all_labels.keys(), times=times, out_fn=f'{out_dir}/scenes_props_multi_all.png', word_onsets=word_onsets, image_onset=image_onset) # ['S1','C1','R','S2','C2','All1stObj','All2ndObj']
# plot_all_props(ave_dict=ave_auc_all_labels, std_dict=std_auc_all_labels, labels=ave_auc_all_labels.keys(), times=times, out_fn=f'{out_dir}/scenes_props_all.png', word_onsets=word_onsets, image_onset=image_onset) # ['S1','C1','R','S2','C2','All1stObj','All2ndObj']

# # default ones
# plot_all_props_multi(ave_dict=ave_auc_all_labels, std_dict=std_auc_all_labels, times=times, out_fn=f'{out_dir}/scenes_props_multi.png', word_onsets=word_onsets, image_onset=image_onset)
# plot_all_props(ave_dict=ave_auc_all_labels, std_dict=std_auc_all_labels, times=times, out_fn=f'{out_dir}/scenes_props.png', word_onsets=word_onsets, image_onset=image_onset)

# # extended ones
# plot_all_props_multi(ave_dict=ave_auc_all_labels, std_dict=std_auc_all_labels, labels=['S1','C1','R','S2','C2','All1stObj','All2ndObj'], times=times, out_fn=f'{out_dir}/scenes_props_multi_all.png', word_onsets=word_onsets, image_onset=image_onset)
# plot_all_props(ave_dict=ave_auc_all_labels, std_dict=std_auc_all_labels, times=times, out_fn=f'{out_dir}/scenes_props_all.png', word_onsets=word_onsets, image_onset=image_onset)


# x = PrettyTable()
# x.field_names = ["Factor", "max AUC"]
# order = np.argsort(max_auc_all_facconds)
# with open(f'{out_fn}_max_AUC.csv', 'w') as f:
#     for i in order:
#         x.add_row([all_faccond_names[i], f"{max_auc_all_facconds[i]:.3f}"])
#         f.write(f"{all_faccond_names[i]}, {max_auc_all_facconds[i]:.3f}")
# # print(x)
# # set_trace()

print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")
