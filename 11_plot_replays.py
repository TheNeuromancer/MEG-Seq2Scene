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
# import os.path as op
# import os
import importlib
from glob import glob
from natsort import natsorted
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import sem
# from sklearn.preprocessing import LabelEncoder, LabelBinarizer
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
print(df)
df['preds'].apply(lambda x: np.array(x))

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