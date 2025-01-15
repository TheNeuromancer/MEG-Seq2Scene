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
from scipy.stats import sem, ttest_ind
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler
import warnings
# warnings.filterwarnings('ignore', '.*Provided stat_fun.*', )
# warnings.filterwarnings('ignore', '.*No clusters found.*', )

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
parser.add_argument('-d', '--dont_recompute', action='store_true',  default=False, help='Whether to skip the aggregation phase, only works if we already saved the preds')
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
minmaxScaler = MinMaxScaler()

## All possible training time (depends on the property that is decoded).
train_times = ['0.8', '2.6', '0.2', '1.4', '2.0']
# train_times = ["0.17", "0.2", "0.3", "0.4", "0.5", "0.6", "0.8"] + ["0.77", "0.9", "1.0", "1.1", "1.2", "1.4"] + ["1.37", "1.5", "1.6", "1.7", "1.8", "2.0"]
# train_times = train_times + ["1.97", "2.1", "2.2", "2.3", "2.4", "2.6"] + ["2.57", "2.7", "2.8", "2.9", "3.0", "3.2"]
## Generalization window for objects and scenes
gen_windows = [(3, 5), (1.5, 2.2)]

if args.dont_recompute is False:
    preds_fn = f"{op.dirname(op.dirname(out_dir))}/all_preds.p"
    metadata_fn = f"{op.dirname(op.dirname(out_dir))}/all_metadata.p"
    all_labels = np.unique([op.basename(fn).split('-')[0] for fn in all_fns])
    # preds_all_labels, pattern_all_labels, confusion_all_labels = {}, {}, {}
    all_df = []
    for label in all_labels:
        if args.verbose: print(f"Doing {label}")
        if "Obj" in label:
            print("Skipping Objects for now")
            continue
        for train_cond in ["localizer", "obj", "scenes"]:
            for split_query in [False]: # no split query in replay decoding so far (but migh wanna include it later)
            # for split_query in ["match", "nonmatch", "flash", "noflash", "match_or_Error_type=l0", "match_or_Error_type=l1", \
            #                     "match_or_Error_type=l2", "Complexity=0", "Complexity=1", "Complexity=2", \
            #                     "Change.str.containsshape", "Change.str.containscolour", False]:
                for gen_cond in ["obj", "scenes"]: # "localizer", 
                    for train_time in train_times:
                        for gen_window in gen_windows:
                            if args.verbose: print(train_time)
                            all_patterns, all_confusions, all_preds, all_subs, all_items = [], [], [], [], []
                            mds_this_cond = []
                            for fn in all_fns:
                                if op.basename(fn)[0:len(label)+1] != f"{label}-": continue 
                                if f"cond-{train_cond}-" not in fn: continue
                                # print(fn)
                                # if not split_query: # if not split_query or nonmatch markers, keep all non-splitqueries
                                #     split_query_str = ""
                                #     if "_for_" in fn: continue
                                # else:
                                #     if f"for_{split_query}" not in fn: continue
                                #     split_query_str = f"_for_{split_query}"
                                if gen_cond is not None:
                                    if f"#{train_time},{train_time}#{gen_window[0]},{gen_window[1]}#" not in fn: 
                                        continue
                                    if f"tested_on_{gen_cond}" not in fn: continue # only generalization results
                                else: # gen_cond is None 
                                    if "tested_on" in fn: continue # ensure we don't have generalization results (shouldn't be usefull after the preceeding line)
                                    print(train_time, fn)
                                    if f"#{train_time},{train_time}#{train_time},{train_time}#" not in fn:
                                        continue
                                # do not load the full mnius splits
                                if "full_minus_split" in fn: continue

                                if args.verbose: print('loading file ', fn)
                                preds = np.load(fn) # times * trials * classes 
                                preds = preds.squeeze() # trials * classes (single time point for replay decoding)
                                all_preds.append(preds) # len(n_subs) of array of inhomogeneous shape n_trials * n_classes
                                # all_subs.append(op.basename(op.dirname(fn))[0:2])
                                # all_items.append(op.basename(fn))

                                metadata_fn = fn.replace('preds.npy', 'metadata.csv')
                                md = pd.read_csv(metadata_fn) # times * trials * classes 
                                n_times, n_trials, n_classes = preds.shape

                                md["train_time"] = [train_time] * n_trials
                                md["gen_window"] = [gen_window] * n_trials
                                md["train_cond"] = [train_cond] * n_trials
                                md["gen_cond"] = [gen_cond] * n_trials
                                md["split_query"] = [split_query] * n_trials
                                md["label"] = [label] * n_trials
                                md["preds"] = preds.transpose(1,0,2).tolist()
                                sub = op.basename(op.dirname(fn))[0:2]
                                md["sub"] = [sub] * n_trials
                                # md["trial"] = [f"{sub}_{i}" for i in range(n_trials)] # not unique for eqch trial

                                # mds_this_cond.append(md)
                                all_df.append(md)


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
                            out_fn = f"{out_dir}/{label}_trained_on_{train_cond}{gen_str}_{n_subs}ave"

                            if not len(all_preds): 
                                print(f"did find any pred for {label} trained on {train_cond} with generalization {gen_cond} for  split query {split_query}, continuing")
                                continue

                            # store values for all labels for multi plot
                            # all_df.append(pd.DataFrame(future_df))
                            # preds_all_labels[f"{label}_{train_cond}_{gen_cond}_"] = all_preds

                            if args.verbose: print(f"Finished {label} trained on {train_cond} with generalization {gen_cond}  for  split query {split_query}\n")
                            plt.close('all')


    df = pd.concat(all_df)
    df.drop(columns=["Unnamed: 0", "Loc_word", "Word_position", "Mapping", "Matching", "Error_type", "Violated_position", "split_query", "Mismatch_side"], inplace=True)
    df.reset_index(inplace=True)
    df['trial_id'] = df ["sub"] + df["run_nb"].apply(str) + df["RT"].apply(str) + df["Difficulty"] + df["Shape1"] + df["Colour1"] + df["Shape2"] + df["Colour2"]
    # + df["Img_position"]
     # + df["Fontsize"].apply(str) + df["Change"]
    df.to_csv(f"{out_dir}/all_preds_data.csv", index=False)

else:
    df = pd.read_csv(f"{out_dir}/all_preds_data.csv")


from utils.replays import *

maxLag = 50
times = np.arange(maxLag)*10
# theoretical_peak = 11 # in the paper the peak lag is at 110ms
# pval_th = 0.05 / maxLag
subs = df['sub'].unique()
n_subs = len(subs)

def get_TF_2words(shape, color):
    """ get the forward transition matrix
    for 2-word blocks. 
    Positions in the matrix are shape, then colors """
    color_offset = 3
    T = np.zeros((6, 6))
    T[shapes.index(shape), colors.index(color) + color_offset] = 1
    return T
    # single transition ... is that ok for sequenceness? 


def get_TF_5words(s1, c1, rel, s2, c2):
    """ get the forward transition matrix
    for 5-word blocks. 
    Positions in the matrix are shape, color, then relation """
    T = np.zeros((8, 8))
    color_offset = 3
    relation_offset = 6
    s1_idx, c1_idx, r_idx, s2_idx, c2_idx = shapes.index(s1), colors.index(c1), relations.index(rel), shapes.index(s2), colors.index(c2)
    T[s1_idx, c1_idx+color_offset] = 1
    T[c1_idx+color_offset, r_idx+relation_offset] = 1
    T[r_idx+relation_offset, s2_idx] = 1
    T[s2_idx, c2_idx+color_offset] = 1
    return T

def splits(data, num_splits=30):
    """
    Splits the data into `num_splits` equal-length segments,
    and then calculates the overall mean and SEM across splits.
    """
    split_data = np.array_split(data, num_splits, axis=0)  # Split data into `num_splits` parts
    split_means = [np.mean(split, axis=0) for split in split_data]  # Mean of each split
    split_means = np.array(split_means)
    overall_mean = np.mean(split_means, axis=0)  # Overall mean across splits
    overall_sem = np.std(split_means, axis=0) / np.sqrt(num_splits)  # SEM across splits
    return overall_mean, overall_sem


def plot_average_preds(present, absent, kind):
    """ bar plot of average predictions during the delay
    presents: list of np.array of len(n_trials), grouped for all subjects 
    absents: list of np.array of len(n_trials), grouped for all subjects
    kind: str to add to the out_fn, where the decoders were trained on (ImgLoc, scenes, ...)
    """
    # from ipdb import set_trace; set_trace()
    present_ave, present_sem = [], []
    for i in range(len(present)):
        qwe, asd = splits(present[i])
        present_ave.append(qwe)
        present_sem.append(asd)
    # present_ave, present_sem = splits(present)
    # present_ave = [np.mean(np.array_split(preds, 30, axis=0), 0) for preds in present]
    # present_sem = [sem(np.array_split(preds, 30, axis=0)) for preds in present]  # Split data into `num_splits` parts
    # present_sem = [sem(preds, 0, nan_policy='omit') for preds in present]
    # present_sem = [np.std(preds, 0) for preds in present]
    # absent_ave = [np.mean(preds, 0) for preds in absent]
    # absent_sem = [sem(preds, 0, nan_policy='omit') for preds in absent]
    # absent_sem = [np.std(preds, 0) for preds in absent]
    # absent_ave = [np.mean(np.array_split(preds, 30, axis=0), 0) for preds in absent]
    # absent_sem = [sem(np.array_split(preds, 30, axis=0)) for preds in absent]  # Split data into `num_splits` parts
    # absent_ave, absent_sem = splits(absent)
    absent_ave, absent_sem = [], []
    for i in range(len(absent)):
        qwe, asd = splits(absent[i])
        absent_ave.append(qwe)
        absent_sem.append(asd)
    
    # Bar plot
    labels = ['Shape', 'Color', 'Relation']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, present_ave, width, yerr=present_sem, label='Present', color='skyblue')
    rects2 = ax.bar(x + width/2, absent_ave, width, yerr=absent_sem, label='Absent', color='orange')

    # Add labels, title, and legend
    ax.set_ylabel('Average Predictions')
    # ax.set_title('Average Predictions by Presence')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # Add value labels
    # ax.bar_label(rects1, fmt='%.2f', padding=3)
    # ax.bar_label(rects2, fmt='%.2f', padding=3)


    # Perform t-tests for each category
    preds_shape_present, preds_color_present, preds_rel_present = present
    preds_shape_absent, preds_color_absent, preds_rel_absent = absent
    shape_ttest = ttest_ind(preds_shape_present, preds_shape_absent)
    color_ttest = ttest_ind(preds_color_present, preds_color_absent)
    relation_ttest = ttest_ind(preds_rel_present, preds_rel_absent)
    p_values = [shape_ttest.pvalue, color_ttest.pvalue, relation_ttest.pvalue]

    alpha = 0.05
    print("\nSignificance Testing Results:")
    print(f"Shape: {'Significant' if shape_ttest.pvalue < alpha else 'Not Significant'} (p = {shape_ttest.pvalue:.4f})")
    print(f"Color: {'Significant' if color_ttest.pvalue < alpha else 'Not Significant'} (p = {color_ttest.pvalue:.4f})")
    print(f"Relation: {'Significant' if relation_ttest.pvalue < alpha else 'Not Significant'} (p = {relation_ttest.pvalue:.4f})")

    # Add significance stars
    for i, p_val in enumerate(p_values):
        if p_val < alpha:
            y_max = max(present_ave[i] + present_sem[i], absent_ave[i] + absent_sem[i])
            ax.text(i, y_max + 0.05, '*', ha='center', va='bottom', fontsize=16, color='k')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{out_dir}/average_preds_{kind}_tested.png", dpi=400)


    plt.close()


# ### 5-words blocks ###
# n_states = 8
# T_auto = np.eye(n_states)  # Autotransitions
# T_const = np.ones((n_states, n_states))  # Uniform transitions
# # df_5words = df.query("label in ['ImgS_1', 'ImgS_2', 'R_0', 'ImgC_1', 'ImgC_2']") # only the 5-words blocks
# df_5words = df.query("label in ['ImgS_1', 'ImgS_2', 'R_0', 'ImgC_1', 'ImgC_2']") # only the 5-words blocks
# sf = np.full((n_subs, maxLag), np.nan) # to store the average of all trials for each subject and lag
# sb, srand = np.copy(sf), np.copy(sf) # also a random matrix, for checking purpose
# preds_shape_present, preds_color_present, preds_rel_present = [], [], []
# preds_shape_absent, preds_color_absent, preds_rel_absent = [], [], []
# for iSub, sub in tqdm(enumerate(subs)):
#     df_sub = df_5words.query(f"sub=='{sub}'")
#     trial_ids = df_sub.trial_id.unique()
#     n_trials = len(trial_ids)

#     for iLag in range(maxLag): # for each lag
#         if iLag > 0: continue

#         sf_all_trials, sb_all_trials, srand_all_trials = [], [], []

#         for iTrial in range(n_trials):
#             # s1, c1, rel, s2, c2 = df_5words.iloc[iTrial][["Shape1", "Colour1", "Relation", "Shape2", "Colour2"]]
#             df_trial = df_sub.query(f"trial_id=='{trial_ids[iTrial]}'")
#             if len(df_trial) != 5: 
#                 print(f"Had to skip trial {iTrial} for lag {iLag*10} ms")
#                 continue
#                 from ipdb import set_trace; set_trace()
#             if df_trial.Shape1.nunique() > 1: from ipdb import set_trace; set_trace()
#             if df_trial.Colour1.nunique() > 1: from ipdb import set_trace; set_trace()
#             if df_trial.Relation.nunique() > 1: from ipdb import set_trace; set_trace()
#             if df_trial.Shape2.nunique() > 1: from ipdb import set_trace; set_trace()
#             if df_trial.Colour2.nunique() > 1: from ipdb import set_trace; set_trace()

#             s1, c1, rel, s2, c2 = df_trial.iloc[0][["Shape1", "Colour1", "Relation", "Shape2", "Colour2"]].values
#             try:
#                 TF = get_TF_5words(s1, c1, rel, s2, c2)
#             except:
#                 from ipdb import set_trace; set_trace()
#             TR = TF.T
#             rand_inds = np.random.permutation(8)
#             Trand = TF[rand_inds]
#             templates = [TF, TR, Trand, T_auto, T_const]

#             preds_shape = df_trial.query("label=='ImgS_1'")['preds'].values[0] # ImgS_1 and ImgS_2 are equal because they are based on the same decoder
#             preds_color = df_trial.query("label=='ImgC_2'")['preds'].values[0]
#             preds_rel = df_trial.query("label=='R_0'")['preds'].values[0]

#             # save preds for barplot of average predictions
#             if iLag == 0:
#                 preds_shape, preds_color, preds_rel = np.array(preds_shape), np.array(preds_color), np.array(preds_rel)
#                 # present 
#                 preds_shape_present.append(preds_shape[:, shapes.index(s1)].mean())
#                 preds_shape_present.append(preds_shape[:, shapes.index(s2)].mean())
#                 preds_rel_present.append(preds_rel[:, relations.index(rel)].mean())
#                 preds_color_present.append(preds_color[:, colors.index(c1)].mean())
#                 preds_color_present.append(preds_color[:, colors.index(c2)].mean())
#                 # absent
#                 shapes_absent = [s for s in shapes if s not in [s1, s2]]
#                 colors_absent = [c for c in colors if c not in [c1, c2]]
#                 relation_absent = [r for r in relations if r != rel][0]
#                 for absent_shape in shapes_absent:
#                     preds_shape_absent.append(preds_shape[:, shapes.index(absent_shape)].mean())
#                 for absent_color in colors_absent:
#                     preds_color_absent.append(preds_color[:, colors.index(absent_color)].mean())
#                 preds_rel_absent.append(preds_rel[:, relations.index(relation_absent)].mean())

# #             trial_preds = np.concatenate([preds_shape, preds_color, preds_rel], axis=1)
# #             trm = compute_TRM_single_trial(trial_preds, iLag)
# #             trm = minmaxScaler.fit_transform(trm) # a priori no used in wimmer
# #             Z = second_level_analysis(trm, templates)

# #             sf_all_trials.append(Z[0])
# #             sb_all_trials.append(Z[1])
# #             srand_all_trials.append(Z[2])

# #         # mean over trials for this subject, lag and condition
# #         sf[iSub, iLag] = np.nanmean(np.array(sf_all_trials), axis=0)
# #         sb[iSub, iLag] = np.nanmean(np.array(sb_all_trials), axis=0)
# #         srand[iSub, iLag] = np.nanmean(np.array(srand_all_trials), axis=0)

# #     sf[iSub] -= np.nanmean(sf[iSub]) # mean correct
# #     sb[iSub] -= np.nanmean(sb[iSub]) # mean correct
# #     srand[iSub] -= np.nanmean(srand[iSub]) # mean correct
  
# # mean_SF = np.nanmean(sf, 0) # average over subjects
# # std_SF = sem(sf, 0, nan_policy='omit') # std over subjects
# # mean_SB = np.nanmean(sb, 0) # average over subjects
# # std_SB = sem(sb, 0, nan_policy='omit') # std over subjects
# # mean_SRAND = np.nanmean(srand, 0) # average over subjects
# # std_SRAND = sem(srand, 0, nan_policy='omit') # std over subjects
# # plot_f = plt.plot(times, mean_SF, label='forward')[0]
# # plt.fill_between(times, mean_SF-std_SF, mean_SF+std_SF, alpha=0.2, color=plot_f.get_color(), lw=0)
# # plot_b = plt.plot(times, mean_SB, label='backward')[0]
# # plt.fill_between(times, mean_SB-std_SB, mean_SB+std_SB, alpha=0.2, color=plot_b.get_color(), lw=0)
# # plot_rand = plt.plot(times, mean_SRAND, label='random')[0]
# # plt.fill_between(times, mean_SRAND-std_SRAND, mean_SRAND+std_SRAND, alpha=0.2, color=plot_rand.get_color(), lw=0)
# # plt.xlabel("Lag (ms)")
# # plt.ylabel("Sequenceness")
# # plt.legend()
# # plt.savefig(f"{out_dir}/mean_sequenceness_avetrm_scenes_trained_on_ImgLoc.png", dpi=400)
# # plt.close()

# present = [preds_shape_present, preds_color_present, preds_rel_present]
# absent = [preds_shape_absent, preds_color_absent, preds_rel_absent]
# plot_average_preds(present, absent, "scenes_trained_on_ImgLoc")

# ### 5-words blocks WORD ###
# n_states = 8
# T_auto = np.eye(n_states)  # Autotransitions
# T_const = np.ones((n_states, n_states))  # Uniform transitions
# df_5words = df.query("label in ['WordS_1', 'WordS_2', 'R_0', 'WordC_1', 'WordC_2']") # only the 5-words blocks
# sf = np.full((n_subs, maxLag), np.nan) # to store the average of all trials for each subject and lag
# sb, srand = np.copy(sf), np.copy(sf) # also a random matrix, for checking purpose
# preds_shape_present, preds_color_present, preds_rel_present = [], [], []
# preds_shape_absent, preds_color_absent, preds_rel_absent = [], [], []
# for iSub, sub in tqdm(enumerate(subs)):
#     df_sub = df_5words.query(f"sub=='{sub}'")
#     trial_ids = df_sub.trial_id.unique()
#     n_trials = len(trial_ids)

#     for iLag in range(maxLag): # for each lag
#         if iLag > 0: continue



#         sf_all_trials, sb_all_trials, srand_all_trials = [], [], []

#         for iTrial in range(n_trials):
#             df_trial = df_sub.query(f"trial_id=='{trial_ids[iTrial]}'")
#             if len(df_trial) != 5: 
#                 print(f"Had to skip trial {iTrial} for lag {iLag*10} ms")
#                 continue
#             s1, c1, rel, s2, c2 = df_trial.iloc[0][["Shape1", "Colour1", "Relation", "Shape2", "Colour2"]].values
#             TF = get_TF_5words(s1, c1, rel, s2, c2)    
#             TR = TF.T
#             rand_inds = np.random.permutation(8)
#             Trand = TF[rand_inds]
#             templates = [TF, TR, Trand, T_auto, T_const]
#             # trial_preds = preds[:, i_trial, :]
#             preds_shape = df_trial.query("label=='WordS_1'")['preds'].values[0] # WordS_1 and WordS_2 are equal because they are based on the same decoder
#             preds_color = df_trial.query("label=='WordC_2'")['preds'].values[0]
#             preds_rel = df_trial.query("label=='R_0'")['preds'].values[0]

#             # save preds for barplot of average predictions
#             if iLag == 0:
#                 preds_shape, preds_color, preds_rel = np.array(preds_shape), np.array(preds_color), np.array(preds_rel)
#                 # present 
#                 preds_shape_present.append(preds_shape[:, shapes.index(s1)].mean())
#                 preds_shape_present.append(preds_shape[:, shapes.index(s2)].mean())
#                 preds_rel_present.append(preds_rel[:, relations.index(rel)].mean())
#                 preds_color_present.append(preds_color[:, colors.index(c1)].mean())
#                 preds_color_present.append(preds_color[:, colors.index(c2)].mean())
#                 # absent
#                 shapes_absent = [s for s in shapes if s not in [s1, s2]]
#                 colors_absent = [c for c in colors if c not in [c1, c2]]
#                 relation_absent = [r for r in relations if r != rel][0]
#                 for absent_shape in shapes_absent:
#                     preds_shape_absent.append(preds_shape[:, shapes.index(absent_shape)].mean())
#                 for absent_color in colors_absent:
#                     preds_color_absent.append(preds_color[:, colors.index(absent_color)].mean())
#                 preds_rel_absent.append(preds_rel[:, relations.index(relation_absent)].mean())

# #             trial_preds = np.concatenate([preds_shape, preds_color, preds_rel], axis=1)
# #             trm = compute_TRM_single_trial(trial_preds, iLag)
# #             trm = minmaxScaler.fit_transform(trm) # a priori no used in wimmer
# #             Z = second_level_analysis(trm, templates)

# #             sf_all_trials.append(Z[0])
# #             sb_all_trials.append(Z[1])
# #             srand_all_trials.append(Z[2])

# #         # mean over trials for this subject, lag and condition
# #         sf[iSub, iLag] = np.nanmean(np.array(sf_all_trials), axis=0)
# #         sb[iSub, iLag] = np.nanmean(np.array(sb_all_trials), axis=0)
# #         srand[iSub, iLag] = np.nanmean(np.array(srand_all_trials), axis=0)

# #     sf[iSub] -= np.nanmean(sf[iSub]) # mean correct
# #     sb[iSub] -= np.nanmean(sb[iSub]) # mean correct
# #     srand[iSub] -= np.nanmean(srand[iSub]) # mean correct
  
# # mean_SF = np.nanmean(sf, 0) # average over subjects
# # std_SF = sem(sf, 0, nan_policy='omit') # std over subjects
# # mean_SB = np.nanmean(sb, 0) # average over subjects
# # std_SB = sem(sb, 0, nan_policy='omit') # std over subjects
# # mean_SRAND = np.nanmean(srand, 0) # average over subjects
# # std_SRAND = sem(srand, 0, nan_policy='omit') # std over subjects
# # plot_f = plt.plot(times, mean_SF, label='forward')[0]
# # plt.fill_between(times, mean_SF-std_SF, mean_SF+std_SF, alpha=0.2, color=plot_f.get_color(), lw=0)
# # plot_b = plt.plot(times, mean_SB, label='backward')[0]
# # plt.fill_between(times, mean_SB-std_SB, mean_SB+std_SB, alpha=0.2, color=plot_b.get_color(), lw=0)
# # plot_rand = plt.plot(times, mean_SRAND, label='random')[0]
# # plt.fill_between(times, mean_SRAND-std_SRAND, mean_SRAND+std_SRAND, alpha=0.2, color=plot_rand.get_color(), lw=0)
# # plt.xlabel("Lag (ms)")
# # plt.ylabel("Sequenceness")
# # plt.legend()
# # plt.savefig(f"{out_dir}/mean_sequenceness_avetrm_scenes_trained_on_WordLoc.png", dpi=400)
# # plt.close()

# present = [preds_shape_present, preds_color_present, preds_rel_present]
# absent = [preds_shape_absent, preds_color_absent, preds_rel_absent]
# plot_average_preds(present, absent, "scenes_trained_on_WordLoc")


# ### trained on 2-words blocks ###
# n_states = 8
# T_auto = np.eye(n_states)  # Autotransitions
# T_const = np.ones((n_states, n_states))  # Uniform transitions
# df_5words = df.query("label in ['S_0', 'S_1', 'R_0', 'C_0', 'C_1']") # only the 5-words blocks
# sf = np.full((n_subs, maxLag), np.nan) # to store the average of all trials for each subject and lag
# sb, srand = np.copy(sf), np.copy(sf) # also a random matrix, for checking purpose
# preds_shape_present, preds_color_present, preds_rel_present = [], [], []
# preds_shape_absent, preds_color_absent, preds_rel_absent = [], [], []
# for iSub, sub in tqdm(enumerate(subs)):
#     df_sub = df_5words.query(f"sub=='{sub}'")
#     trial_ids = df_sub.trial_id.unique()
#     n_trials = len(trial_ids)

#     for iLag in range(maxLag): # for each lag
#         if iLag > 0: continue



#         sf_all_trials, sb_all_trials, srand_all_trials = [], [], []

#         for iTrial in range(n_trials):
#             df_trial = df_sub.query(f"trial_id=='{trial_ids[iTrial]}'")
#             if len(df_trial) != 5: 
#                 print(f"Had to skip trial {iTrial} for lag {iLag*10} ms")
#                 continue
#             s1, c1, rel, s2, c2 = df_trial.iloc[0][["Shape1", "Colour1", "Relation", "Shape2", "Colour2"]].values
#             TF = get_TF_5words(s1, c1, rel, s2, c2)
#             TR = TF.T
#             rand_inds = np.random.permutation(8)
#             Trand = TF[rand_inds]
#             templates = [TF, TR, Trand, T_auto, T_const]
#             preds_shape = df_trial.query("label=='S_0'")['preds'].values[0]
#             preds_color = df_trial.query("label=='C_0'")['preds'].values[0]
#             preds_rel = df_trial.query("label=='R_0'")['preds'].values[0]

#             # save preds for barplot of average predictions
#             if iLag == 0:
#                 preds_shape, preds_color, preds_rel = np.array(preds_shape), np.array(preds_color), np.array(preds_rel)
#                 # present 
#                 preds_shape_present.append(preds_shape[:, shapes.index(s1)].mean())
#                 preds_shape_present.append(preds_shape[:, shapes.index(s2)].mean())
#                 preds_rel_present.append(preds_rel[:, relations.index(rel)].mean())
#                 preds_color_present.append(preds_color[:, colors.index(c1)].mean())
#                 preds_color_present.append(preds_color[:, colors.index(c2)].mean())
#                 # absent
#                 shapes_absent = [s for s in shapes if s not in [s1, s2]]
#                 colors_absent = [c for c in colors if c not in [c1, c2]]
#                 relation_absent = [r for r in relations if r != rel][0]
#                 for absent_shape in shapes_absent:
#                     preds_shape_absent.append(preds_shape[:, shapes.index(absent_shape)].mean())
#                 for absent_color in colors_absent:
#                     preds_color_absent.append(preds_color[:, colors.index(absent_color)].mean())
#                 preds_rel_absent.append(preds_rel[:, relations.index(relation_absent)].mean())


# #             trial_preds = np.concatenate([preds_shape, preds_color, preds_rel], axis=1)
# #             trm = compute_TRM_single_trial(trial_preds, iLag)
# #             trm = minmaxScaler.fit_transform(trm) # a priori no used in wimmer
# #             Z = second_level_analysis(trm, templates)

# #             sf_all_trials.append(Z[0])
# #             sb_all_trials.append(Z[1])
# #             srand_all_trials.append(Z[2])

# #         # mean over trials for this subject, lag and condition
# #         sf[iSub, iLag] = np.nanmean(np.array(sf_all_trials), axis=0)
# #         sb[iSub, iLag] = np.nanmean(np.array(sb_all_trials), axis=0)
# #         srand[iSub, iLag] = np.nanmean(np.array(srand_all_trials), axis=0)

# #     sf[iSub] -= np.nanmean(sf[iSub]) # mean correct
# #     sb[iSub] -= np.nanmean(sb[iSub]) # mean correct
# #     srand[iSub] -= np.nanmean(srand[iSub]) # mean correct
  
# # mean_SF = np.nanmean(sf, 0) # average over subjects
# # std_SF = sem(sf, 0, nan_policy='omit') # std over subjects
# # mean_SB = np.nanmean(sb, 0) # average over subjects
# # std_SB = sem(sb, 0, nan_policy='omit') # std over subjects
# # mean_SRAND = np.nanmean(srand, 0) # average over subjects
# # std_SRAND = sem(srand, 0, nan_policy='omit') # std over subjects
# # plot_f = plt.plot(times, mean_SF, label='forward')[0]
# # plt.fill_between(times, mean_SF-std_SF, mean_SF+std_SF, alpha=0.2, color=plot_f.get_color(), lw=0)
# # plot_b = plt.plot(times, mean_SB, label='backward')[0]
# # plt.fill_between(times, mean_SB-std_SB, mean_SB+std_SB, alpha=0.2, color=plot_b.get_color(), lw=0)
# # plot_rand = plt.plot(times, mean_SRAND, label='random')[0]
# # plt.fill_between(times, mean_SRAND-std_SRAND, mean_SRAND+std_SRAND, alpha=0.2, color=plot_rand.get_color(), lw=0)
# # plt.xlabel("Lag (ms)")
# # plt.ylabel("Sequenceness")
# # plt.legend()
# # plt.savefig(f"{out_dir}/mean_sequenceness_avetrm_scenes_trained_on_Obj.png", dpi=400)
# # plt.close()

# present = [preds_shape_present, preds_color_present, preds_rel_present]
# absent = [preds_shape_absent, preds_color_absent, preds_rel_absent]
# plot_average_preds(present, absent, "scenes_trained_on_Obj")


### trained on 5-words blocks ###
n_states = 8
T_auto = np.eye(n_states)  # Autotransitions
T_const = np.ones((n_states, n_states))  # Uniform transitions
df_5words = df.query("label in ['S1_0', 'S2_1', 'R_0', 'C1_0', 'C2_1']") # only the 5-words blocks
sf = np.full((n_subs, maxLag), np.nan) # to store the average of all trials for each subject and lag
sb, srand = np.copy(sf), np.copy(sf) # also a random matrix, for checking purpose
preds_shape_present, preds_color_present, preds_rel_present = [], [], []
preds_shape_absent, preds_color_absent, preds_rel_absent = [], [], []
for iSub, sub in tqdm(enumerate(subs)):
    df_sub = df_5words.query(f"sub=='{sub}'")
    trial_ids = df_sub.trial_id.unique()
    n_trials = len(trial_ids)

    for iLag in range(maxLag): # for each lag
        if iLag > 0: continue



        sf_all_trials, sb_all_trials, srand_all_trials = [], [], []

        for iTrial in range(n_trials):
            df_trial = df_sub.query(f"trial_id=='{trial_ids[iTrial]}'")
            if len(df_trial) != 5: 
                print(f"Had to skip trial {iTrial} for lag {iLag*10} ms")
                continue
            s1, c1, rel, s2, c2 = df_trial.iloc[0][["Shape1", "Colour1", "Relation", "Shape2", "Colour2"]].values
            TF = get_TF_5words(s1, c1, rel, s2, c2)
            TR = TF.T
            rand_inds = np.random.permutation(8)
            Trand = TF[rand_inds]
            templates = [TF, TR, Trand, T_auto, T_const]
            if iTrial==0: print(f"TODO: Check the trained on scenes / tested in scenes. You nw have 2 different decoders for the shape and for the colour.")
            # preds_shape = df_trial.query("label=='S1_0'")['preds'].values[0] # now 'S1_0', 'S2_1' are different because based on different decoders.
            # preds_color = df_trial.query("label=='C1_0'")['preds'].values[0]
            preds_shape = df_trial.query("label=='S2_1'")['preds'].values[0] # now 'S1_0', 'S2_1' are different because based on different decoders.
            preds_color = df_trial.query("label=='C2_1'")['preds'].values[0]
            preds_rel = df_trial.query("label=='R_0'")['preds'].values[0]

            # save preds for barplot of average predictions
            if iLag == 0:
                preds_shape, preds_color, preds_rel = np.array(preds_shape), np.array(preds_color), np.array(preds_rel)
                # present 
                preds_shape_present.append(preds_shape[:, shapes.index(s1)].mean())
                preds_shape_present.append(preds_shape[:, shapes.index(s2)].mean())
                preds_rel_present.append(preds_rel[:, relations.index(rel)].mean())
                preds_color_present.append(preds_color[:, colors.index(c1)].mean())
                preds_color_present.append(preds_color[:, colors.index(c2)].mean())
                # absent
                shapes_absent = [s for s in shapes if s not in [s1, s2]]
                colors_absent = [c for c in colors if c not in [c1, c2]]
                relation_absent = [r for r in relations if r != rel][0]
                for absent_shape in shapes_absent:
                    preds_shape_absent.append(preds_shape[:, shapes.index(absent_shape)].mean())
                for absent_color in colors_absent:
                    preds_color_absent.append(preds_color[:, colors.index(absent_color)].mean())
                preds_rel_absent.append(preds_rel[:, relations.index(relation_absent)].mean())


#             trial_preds = np.concatenate([preds_shape, preds_color, preds_rel], axis=1)
#             trm = compute_TRM_single_trial(trial_preds, iLag)
#             trm = minmaxScaler.fit_transform(trm) # a priori no used in wimmer
#             Z = second_level_analysis(trm, templates)

#             sf_all_trials.append(Z[0])
#             sb_all_trials.append(Z[1])
#             srand_all_trials.append(Z[2])

#         # mean over trials for this subject, lag and condition
#         sf[iSub, iLag] = np.nanmean(np.array(sf_all_trials), axis=0)
#         sb[iSub, iLag] = np.nanmean(np.array(sb_all_trials), axis=0)
#         srand[iSub, iLag] = np.nanmean(np.array(srand_all_trials), axis=0)

#     sf[iSub] -= np.nanmean(sf[iSub]) # mean correct
#     sb[iSub] -= np.nanmean(sb[iSub]) # mean correct
#     srand[iSub] -= np.nanmean(srand[iSub]) # mean correct
  
# mean_SF = np.nanmean(sf, 0) # average over subjects
# std_SF = sem(sf, 0, nan_policy='omit') # std over subjects
# mean_SB = np.nanmean(sb, 0) # average over subjects
# std_SB = sem(sb, 0, nan_policy='omit') # std over subjects
# mean_SRAND = np.nanmean(srand, 0) # average over subjects
# std_SRAND = sem(srand, 0, nan_policy='omit') # std over subjects
# plot_f = plt.plot(times, mean_SF, label='forward')[0]
# plt.fill_between(times, mean_SF-std_SF, mean_SF+std_SF, alpha=0.2, color=plot_f.get_color(), lw=0)
# plot_b = plt.plot(times, mean_SB, label='backward')[0]
# plt.fill_between(times, mean_SB-std_SB, mean_SB+std_SB, alpha=0.2, color=plot_b.get_color(), lw=0)
# plot_rand = plt.plot(times, mean_SRAND, label='random')[0]
# plt.fill_between(times, mean_SRAND-std_SRAND, mean_SRAND+std_SRAND, alpha=0.2, color=plot_rand.get_color(), lw=0)
# plt.xlabel("Lag (ms)")
# plt.ylabel("Sequenceness")
# plt.legend()
# plt.savefig(f"{out_dir}/mean_sequenceness_avetrm_scenes_trained_on_Scenes.png", dpi=400)
# plt.close()

present = [preds_shape_present, preds_color_present, preds_rel_present]
absent = [preds_shape_absent, preds_color_absent, preds_rel_absent]
plot_average_preds(present, absent, "scenes_trained_on_scenes")


# ### 2-words blocks ###
# n_states = 6
# T_auto = np.eye(n_states)  # Autotransitions
# T_const = np.ones((n_states, n_states))  # Uniform transitions
# conds = [f'{s} {c}' for s in shapes for c in colors] # all conditions for the 2-words blocks
# sf = np.full((n_subs, maxLag), np.nan) # to store the average of all trials for each subject and lag
# sb, srand = np.copy(sf), np.copy(sf) # also a random matrix, for checking purpose
# for iSub, sub in tqdm(enumerate(subs)):

#     for iLag in range(maxLag): # for each lag
#         sf_all_trials, sb_all_trials, srand_all_trials = [], [], []

#         for cond in conds: # for each single set of words
#             shape, color = cond.split()
#             TF = get_TF_2words(shape, color)
#             TR = TF.T
#             rand_inds = np.random.randint(0, 2, 2) # maybe make this better, choose indices that are not the one that are evaluated and that are valid transitions
#             # , random_state=iSub
#             Trand = np.ones((n_states, n_states))
#             Trand[rand_inds[0], rand_inds[1]] = 1
#             templates = [TF, TR, Trand, T_auto, T_const]

#             preds_shape = df.query(f"Shape1=='{shape}' and Colour1=='{color}' and sub=='{sub}' and label=='ImgS_0'")['preds'].values
#             preds_color = df.query(f"Shape1=='{shape}' and Colour1=='{color}' and sub=='{sub}' and label=='ImgC_0'")['preds'].values
#             preds = np.concatenate([np.stack(preds_shape), np.stack(preds_color)], axis=2)
#             n_times, n_trials, n_classes = preds.shape

#             for i_trial in range(n_trials):
#                 trial_preds = preds[:, i_trial, :]
#                 trm = compute_TRM_single_trial(trial_preds, iLag)
#                 trm = minmaxScaler.fit_transform(trm) # a priori no used in wimmer
#                 Z = second_level_analysis(trm, templates)

#                 sf_all_trials.append(Z[0])
#                 sb_all_trials.append(Z[1])
#                 srand_all_trials.append(Z[2])

#                 # if null: 
#                 #     null_distributions_f[iSub, epi-1, :, iLag], null_distributions_b[iSub, epi-1, :, iLag] = \
#                 #         compute_null_distribution(preds, templates, lag=iLag, n_permutations=n_permutations)

#         # mean over trials for this subject, lag and condition
#         sf[iSub, iLag] = np.nanmean(np.array(sf_all_trials), axis=0)
#         sb[iSub, iLag] = np.nanmean(np.array(sb_all_trials), axis=0)
#         srand[iSub, iLag] = np.nanmean(np.array(srand_all_trials), axis=0)

#     sf[iSub] -= np.nanmean(sf[iSub]) # mean correct
#     sb[iSub] -= np.nanmean(sb[iSub]) # mean correct
#     srand[iSub] -= np.nanmean(srand[iSub]) # mean correct


# mean_SF = np.nanmean(sf, 0) # average over subjects
# std_SF = sem(sf, 0, nan_policy='omit') # std over subjects
# mean_SB = np.nanmean(sb, 0) # average over subjects
# std_SB = sem(sb, 0, nan_policy='omit') # std over subjects
# mean_SRAND = np.nanmean(srand, 0) # average over subjects
# std_SRAND = sem(srand, 0, nan_policy='omit') # std over subjects

# plot_f = plt.plot(times, mean_SF, label='forward')[0]
# plt.fill_between(times, mean_SF-std_SF, mean_SF+std_SF, alpha=0.2, color=plot_f.get_color(), lw=0)
# plot_b = plt.plot(times, mean_SB, label='backward')[0]
# plt.fill_between(times, mean_SB-std_SB, mean_SB+std_SB, alpha=0.2, color=plot_b.get_color(), lw=0)
# plot_rand = plt.plot(times, mean_SRAND, label='random')[0]
# plt.fill_between(times, mean_SRAND-std_SRAND, mean_SRAND+std_SRAND, alpha=0.2, color=plot_rand.get_color(), lw=0)
# plt.xlabel("Lag (ms)")
# plt.ylabel("Sequenceness")
# plt.legend()
# plt.savefig(f"{out_dir}/mean_sequenceness_avetrm_obj_trained_on_ImgLoc.png", dpi=400)
# plt.close()





# from my wimmer script. Not sure about it.   
# if null:
#     # Compute z-scores and p-values
#     z_scores_f, p_values_f = combine_null_distributions_and_test(null_distributions_f, sf)
#     print("Z-scores for forward:", z_scores_f)
#     print("P-values for forward:", p_values_f)

#     z_scores_b, p_values_b = combine_null_distributions_and_test(null_distributions_b, sb)
#     print("Z-scores for backward:", z_scores_b)
#     print("P-values back forward:", p_values_b)
### PLOT ### 
# if null:
#     yval_f = np.max(mean_SF) + (np.max(mean_SF) * 0.1)
#     sig_times_f = times[np.where(p_values_f < pval_th)]
#     plt.plot(sig_times_f, np.full(len(sig_times_f), yval_f), 'D', markersize=3, color=plot_f.get_color())
#     yval_b = np.max(mean_SF) + (np.max(mean_SF) * 0.2)
#     sig_times_b = times[np.where(p_values_b < pval_th)]
#     plt.plot(sig_times_b, np.full(len(sig_times_b), yval_b), 'D', markersize=3, color=plot_b.get_color())


# plot = plt.plot(times, mean_SF - mean_SB, label='forward - backward')[0]
# # not sure how to go about this ... 
# # plt.fill_between(times, (mean_SF-mean_SB)-(std_SF, mean_SF+std_SF, alpha=0.2, color=plot.get_color(), lw=0)
# plt.xlabel("Lag (ms)")
# plt.ylabel("Sequenceness")
# plt.legend()
# plt.savefig(f"{out_dir}/mean_difference_sequenceness_obj_difference_last.png", dpi=400)
# plt.close()


# sdiff = sf - sb
# mean_SDIFF = np.nanmean(sdiff, 0) # average over subjects
# std_SDIFF = sem(sdiff, 0, nan_policy='omit') # std over subjects
# plot = plt.plot(times, mean_SDIFF, label='forward - backward')[0]
# plt.fill_between(times, mean_SDIFF-std_SDIFF, mean_SDIFF+std_SDIFF, alpha=0.2, color=plot.get_color(), lw=0)
# plt.xlabel("Lag (ms)")
# plt.ylabel("Sequenceness")
# plt.legend()
# plt.savefig(f"{out_dir}/mean_difference_sequenceness_difference_first.png", dpi=400)
# plt.close()

print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")

from ipdb import set_trace; set_trace()
