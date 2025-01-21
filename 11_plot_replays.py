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
import importlib
from glob import glob
from natsort import natsorted
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import sem
# import warnings
# warnings.filterwarnings('ignore', '.*Provided stat_fun.*', )
# warnings.filterwarnings('ignore', '.*No clusters found.*', )

from utils.decod import *
from utils.params import *
from utils.replays import *

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
res_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"
print('\noutput files will be in: ' + res_dir)
create_folder(res_dir, args.overwrite)

df_fn = f"{res_dir}/all_preds_data.csv"
df = pd.read_csv(df_fn)
# df['preds'].apply(lambda x: np.array(x))

dat_fn = f"{res_dir}/all_preds_data.csv"
all_preds_data = pickle.load(open(f"{res_dir}/all_preds_data.pkl", 'rb'))


maxLag = 50
# times = np.arange(maxLag)*10
# theoretical_peak = 11 # in the paper the peak lag is at 110ms
# pval_th = 0.05 / maxLag
# subs = df['sub'].unique()
# n_subs = len(subs)


def get_preds_and_sequenceness_for_cond(df, preds, train_cond, test_cond, maxLag=50, n_states=8):
    T_auto = np.eye(n_states)  # Autotransitions
    T_const = np.ones((n_states, n_states))  # Uniform transitions
    times = np.arange(maxLag)*10
    subs = df['sub'].unique()
    n_subs = len(subs)

    from ipdb import set_trace; set_trace()
    df_cond = df.query(f"train_cond == {train_cond} and test_cond == {test_cond}")
    sf = np.full((n_subs, maxLag), np.nan) # to store the average of all trials for each subject and lag
    sb, srand = np.copy(sf), np.copy(sf) # also a random matrix, for comparison purpose
    preds_present, preds_absent = [], []
    for iSub, sub in enumerate(subs):
        df_sub = df_cond.query(f"sub=='{sub}'")
        trial_ids = df_sub.trial_id.unique()
        n_trials = len(trial_ids)

        for iLag in range(maxLag): # for each lag
            sf_all_trials, sb_all_trials, srand_all_trials = [], [], []
            # preds_present_all_trials, preds_absent_all_trials = [], []
            preds_all = {f"{prez}_{prop}": [] for prez in ['present', 'absent'] for prop in properties}
            perfs = []
            for iTrial in range(n_trials):
                df_trial = df_sub.query(f"trial_id=='{trial_ids[iTrial]}'")
                if len(df_trial) != 5: 
                    print(f"Had to skip trial {iTrial} for lag {iLag*10} ms")
                    continue
                s1, c1, rel, s2, c2 = df_trial.iloc[0][Properties].values
                TF = get_TF_5words(s1, c1, rel, s2, c2)
                TR = TF.T
                rand_inds = np.random.permutation(8)
                Trand = TF[rand_inds]
                templates = [TF, TR, Trand, T_auto, T_const]

                preds_shape = df_trial.query(f"label=='{train_cond[0]}'")['preds'].values[0] # now 'S1_0', 'S2_1' are different because based on different decoders.
                preds_color = df_trial.query(f"label=='{train_cond[1]}'")['preds'].values[0]
                preds_rel = df_trial.query(f"label=='{train_cond[2]}'")['preds'].values[0]

                def add_present_or_absent_preds_one_trial(preds, props, preds_all):
                    from ipdb import set_trace; set_trace()
                    for prez, prop in zip(['present', 'absent'], props):
                        preds_all[f"{prez}_{prop}"].append(preds[:, properties.index(prop)].mean())
                    return preds_all

                if iLag == 0: # save preds of present vs absent words for barplot of average predictions

                    from ipdb import set_trace; set_trace()
                    preds_shape, preds_color, preds_rel = np.array(preds_shape), np.array(preds_color), np.array(preds_rel)
                    preds_all = add_present_or_absent_preds_one_trial([preds_shape, preds_color, preds_rel], [s1, c1, rel, s2, c2], preds_all)


                    preds_shape_present.append(preds_shape[:, shapes.index(s1)].mean())
                    preds_shape_present.append(preds_shape[:, shapes.index(s2)].mean())
                    preds_rel_present.append(preds_rel[:, relations.index(rel)].mean())
                    preds_color_present.append(preds_color[:, colors.index(c1)].mean())
                    preds_color_present.append(preds_color[:, colors.index(c2)].mean())

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






maxLag = 60
# theoretical_peak = 11 # in the paper the peak lag is at 110ms
n_states = 6
pval_th = 0.05 / maxLag

T_auto = np.eye(n_states)  # Autotransitions
T_const = np.ones((n_states, n_states))  # Uniform transitions
subs = df['sub'].unique()
n_subs = len(subs)

conds = [f'{s} {c}' for s in shapes for c in colors] # all conditions for the 2-words blocks
sf = np.full((n_subs, maxLag), np.nan) # to store the average of all trials for each subject and lag
sb, srand = np.copy(sf), np.copy(sf) # also a random matrix, for checking purpose
for iSub, sub in tqdm(enumerate(subs)):

    for iLag in range(maxLag): # for each lag
        sf_all_trials, sb_all_trials, srand_all_trials = [], [], []

        for cond in conds: # for each single set of words
            shape, color = cond.split()
            TF = get_TF_2words(shape, color)
            TR = TF.T
            rand_inds = np.random.randint(0, 2, 2) # maybe make this better, choose indices that are not the one that are evaluated and that are valid transitions
            # , random_state=iSub
            Trand = np.ones((n_states, n_states))
            Trand[rand_inds[0], rand_inds[1]] = 1
            templates = [TF, TR, Trand, T_auto, T_const]

            preds_shape = df.query(f"Shape1=='{shape}' and Colour1=='{color}' and sub=={sub} and label=='ImgS_0'")['preds'].values
            preds_color = df.query(f"Shape1=='{shape}' and Colour1=='{color}' and sub=={sub} and label=='ImgC_0'")['preds'].values
            from ipdb import set_trace; set_trace()
            preds = np.concatenate([np.stack(preds_shape), np.stack(preds_color)], axis=2)
            n_times, n_trials, n_classes = preds.shape

            for i_trial in range(n_trials):
                trial_preds = preds[:, i_trial, :]
                trm = compute_TRM_single_trial(trial_preds, iLag)
                # trm = minmaxScaler.fit_transform(trm) # a priori no used in wimmer
                Z = second_level_analysis(trm, templates)

                sf_all_trials.append(Z[0])
                sb_all_trials.append(Z[1])
                srand_all_trials.append(Z[2])

                # if null: 
                #     null_distributions_f[iSub, epi-1, :, iLag], null_distributions_b[iSub, epi-1, :, iLag] = \
                #         compute_null_distribution(preds, templates, lag=iLag, n_permutations=n_permutations)

        # mean over trials for this subject, lag and condition
        sf[iSub, iLag] = np.nanmean(np.array(sf_all_trials), axis=0)
        sb[iSub, iLag] = np.nanmean(np.array(sb_all_trials), axis=0)
        srand[iSub, iLag] = np.nanmean(np.array(srand_all_trials), axis=0)

    sf[iSub] -= np.nanmean(sf[iSub]) # mean correct
    sb[iSub] -= np.nanmean(sb[iSub]) # mean correct
    srand[iSub] -= np.nanmean(srand[iSub]) # mean correct
  
# from my wimmer script. Not sure about it.   
# if null:
#     # Compute z-scores and p-values
#     z_scores_f, p_values_f = combine_null_distributions_and_test(null_distributions_f, sf)
#     print("Z-scores for forward:", z_scores_f)
#     print("P-values for forward:", p_values_f)

#     z_scores_b, p_values_b = combine_null_distributions_and_test(null_distributions_b, sb)
#     print("Z-scores for backward:", z_scores_b)
#     print("P-values back forward:", p_values_b)

mean_SF = np.nanmean(sf, 0) # average over subjects
std_SF = sem(sf, 0, nan_policy='omit') # std over subjects
mean_SB = np.nanmean(sb, 0) # average over subjects
std_SB = sem(sb, 0, nan_policy='omit') # std over subjects
mean_SRAND = np.nanmean(srand, 0) # average over subjects
std_SRAND = sem(srand, 0, nan_policy='omit') # std over subjects

from ipdb import set_trace; set_trace()


times = np.arange(maxLag)*10

plot_f = plt.plot(times, mean_SF, label='forward')[0]
plt.fill_between(times, mean_SF-std_SF, mean_SF+std_SF, alpha=0.2, color=plot_f.get_color(), lw=0)
plot_b = plt.plot(times, mean_SB, label='backward')[0]
plt.fill_between(times, mean_SB-std_SB, mean_SB+std_SB, alpha=0.2, color=plot_b.get_color(), lw=0)
plot_rand = plt.plot(times, mean_SRAND, label='random')[0]
plt.fill_between(times, mean_SRAND-std_SRAND, mean_SRAND+std_SRAND, alpha=0.2, color=plot_rand.get_color(), lw=0)
# if null:
#     yval_f = np.max(mean_SF) + (np.max(mean_SF) * 0.1)
#     sig_times_f = times[np.where(p_values_f < pval_th)]
#     plt.plot(sig_times_f, np.full(len(sig_times_f), yval_f), 'D', markersize=3, color=plot_f.get_color())
#     yval_b = np.max(mean_SF) + (np.max(mean_SF) * 0.2)
#     sig_times_b = times[np.where(p_values_b < pval_th)]
#     plt.plot(sig_times_b, np.full(len(sig_times_b), yval_b), 'D', markersize=3, color=plot_b.get_color())
plt.xlabel("Lag (ms)")
plt.ylabel("Sequenceness")
plt.legend()
plt.savefig(f"{res_dir}/mean_sequenceness_obj.png", dpi=400)
plt.close()

plot = plt.plot(times, mean_SF - mean_SB, label='forward - backward')[0]
# not sure how to go about this ... 
# plt.fill_between(times, (mean_SF-mean_SB)-(std_SF, mean_SF+std_SF, alpha=0.2, color=plot.get_color(), lw=0)
plt.xlabel("Lag (ms)")
plt.ylabel("Sequenceness")
plt.legend()
plt.savefig(f"{res_dir}/mean_difference_sequenceness_obj_difference_last.png", dpi=400)
plt.close()


sdiff = sf - sb
mean_SDIFF = np.nanmean(sdiff, 0) # average over subjects
std_SDIFF = sem(sdiff, 0, nan_policy='omit') # std over subjects
plot = plt.plot(times, mean_SDIFF, label='forward - backward')[0]
plt.fill_between(times, mean_SDIFF-std_SDIFF, mean_SDIFF+std_SDIFF, alpha=0.2, color=plot.get_color(), lw=0)
plt.xlabel("Lag (ms)")
plt.ylabel("Sequenceness")
plt.legend()
plt.savefig(f"{res_dir}/mean_difference_sequenceness_difference_first.png", dpi=400)
plt.close()


print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")