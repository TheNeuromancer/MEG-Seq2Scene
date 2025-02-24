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
from scipy.stats import sem, ttest_rel
import warnings
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
parser.add_argument('-i', '--in-dir', default='multi', help='input directory (ovr or multi)')
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
decoding_dir = f"Decoding_{args.in_dir}_v{v}" # if args.ovr else f"Decoding_v{v}"
res_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"
print('\noutput files will be in: ' + res_dir)
create_folder(res_dir, args.overwrite)

df_fn = f"{res_dir}/all_preds_data.csv"
df = pd.read_csv(df_fn)
all_preds_data = pickle.load(open(f"{res_dir}/all_preds_data.pkl", 'rb'))

print(f"\nOnly keeping the last second of the delay\n")
all_preds_data = [p[100::] for p in all_preds_data]

print(F"Only keeping Complexity==2 trials, because else the repeated states fucks up the replays analyses")
df = df.query(f"Complexity==2")

def get_replays_sophie_style(df, all_preds_data, train_cond, gen_cond, do_rel=True):
    """ Get sequential and synchronous reactivations
    for a prefiltered train and gen cond
    """
    subs = df['sub'].unique()
    n_subs = len(subs)
    if df['train_cond'].nunique() > 1 or df['gen_cond'].nunique() > 1:
        print(f"More than one train or gen condition in the dataframe; it should be pre-filtered before being fed to the func!")
        from ipdb import set_trace; set_trace()
    props = Properties if do_rel else Properties[0:2] + Properties[3:5]
    ave_preds_all_subs = {f"{prop}_{presence}": [] for presence in ['present', 'absent'] for prop in props}
    behav_df = {"Subject": [], "Condition": [], "Property": [], "Reactivation": [], "Performance": [], "RT": []}
    all_subjects_summary = []
    all_sync_coactivation_matrices, all_seq_coactivation_matrices = [], []

    for iSub, sub in tqdm(enumerate(subs)):
        df_sub = df.query(f"sub=={sub}")
        trial_ids = df_sub.trial_id.unique()
        n_trials = len(trial_ids)

        all_sync_react_this_subject = []
        all_seq_react_this_subject = []

        preds_sub_by_presence = {f"{prop}_{presence}": [] for presence in ['present', 'absent'] for prop in Properties} # for this subject and lag
        preds_sub_by_trial = [{f"{prop}_{presence}": [] for presence in ['present', 'absent'] for prop in Properties} for i in range(n_trials)] # for this subject and lag, each trial separately
        perfs = []
        RTs = []
        for iTrial in range(n_trials):
            df_trial = df_sub.query(f"trial_id=='{trial_ids[iTrial]}'")
            if len(df_trial) > 1:
                print(f"Found more than one entry for trial {iTrial}: {trial_ids[iTrial]}")

            s1, c1, rel, s2, c2 = df_trial.iloc[0][Properties].values

            preds_props = get_trial_preds_from_data(df_trial, all_preds_data) # shape n_times * n_classes
            
            # get predictions over the whole window, for each property, depending on whether it is present or absent
            # present: list of 5 arrays of shape (n_samples) for present properties
            # absent: list of lists of one or two arrays of shape (n_samples, n_states) for absent (depending on how many absent properties)
            present, absent = get_present_or_absent_preds_one_trial(preds_props, [s1, c1, rel, s2, c2], do_rel=do_rel)
            # keep only the first absent color and shape: if we that way we avoid repetitions that lead to high reactivations. 
            absent = absent[0:3] if do_rel else absent[0:2]
            ave_present = [p.mean() for p in present]
            ave_absent = [np.mean([a.mean() for a in sublist]) for sublist in absent] # average of averages, if there are multiple absent properties (ie, repetition in the original sentence)

            # update the dict of preds_sub_by_presence
            preds_sub_by_presence = update_present_or_absent_preds_one_trial(preds_sub_by_presence, ave_present, ave_absent, do_rel=do_rel)

            # behavioral results
            perf = df_trial["Perf"].values[0]
            RT = df_trial["RT"].values[0]
            behav_df = update_behav_df(behav_df, sub, perf, RT, ave_present, ave_absent, props, do_rel=do_rel)

            # get Sophie-style reactivations
            preds_this_trial, labels_this_trial = restructure_data(present, absent, do_rel=do_rel) # list of arrays of len n_times. The first five are the present items. The nexts are the absents. 
            # preds_this_trial, labels_this_trial = remove_label_duplicates(preds_this_trial, labels_this_trial) # randomly select one of the absent properties if there are multiple absents
            preds_this_trial, labels_this_trial = average_label_duplicates(preds_this_trial, labels_this_trial) # randomly select one of the absent properties if there are multiple absents
            signif_react = get_significant_reactivations(preds_this_trial)
            # reac_times = get_reactivation_times(signif_react) # not used
            consecutive_react = get_reactivation_episodes(signif_react) # list of tuples: [(start, end, state, duration), ...] for all reactivation episodes.

            # synchronous_episodes_pairs = get_synchronous_reactivations_pairs(consecutive_react) # (list of tuples): [(state1, state2, overlap_start, overlap_duration), ...]
            synchronous_episodes = get_synchronous_reactivations(consecutive_react, tolerance=-1) # list of tuples: [(states, overlap_start, overlap_duration)]
            all_sync_react_this_subject.extend(synchronous_episodes)
            
            sequential_episodes = get_sequential_reactivations(consecutive_react, iLag=10) # (list of lists): [[(state1, duration1, gap1), (state2, duration2, gap2), ...], ...]
            all_seq_react_this_subject.extend(sequential_episodes)



        ## For this subject, get the average of the predictions
        # update the dict of averages
        ave_preds_all_subs = get_subj_ave_preds(preds_sub_by_presence, ave_preds_all_subs, props=props) 


        # # synchronous coactivations
        np2_idx = 3 if do_rel else 4
        # NP1_counts, NP1_overlap = count_coactivations_pairs(all_sync_react_this_subject, 0, 1)
        # NP2_counts, NP2_overlap = count_coactivations_pairs(all_sync_react_this_subject, np2_idx, np2_idx+1)
        # print(f"NP1 states co-activated {NP1_counts} times; average overlap: {NP1_overlap}")
        # print(f"NP2 states co-activated {NP2_counts} times; average overlap: {NP2_overlap}")

        # n_states = max([state for episode in all_sync_react_this_subject for state in episode[0]]) + 1
        n_states = 8 if do_rel else 6
        all_states = set(range(n_states)) 
        
        # first_five_states = {0, 1, 2, 3, 4}
        subset_counts, ave_overlap_size, coactivation_matrix = count_synchronous_coactivations(all_sync_react_this_subject, all_states)
        all_sync_coactivation_matrices.append(coactivation_matrix)

        coactivation_counts, ave_durations, ave_gaps, coactivation_matrix = count_sequential_coactivations(all_seq_react_this_subject, all_states)
        all_seq_coactivation_matrices.append(coactivation_matrix)

        # print("Coactivation Matrix (State Transitions):")
        # for state1, transitions in coactivation_matrix.items():
        #     for state2, count in transitions.items():
        #         if count > 0:
        #             print(f"{state1} -> {state2}: {count}")

        # for i, episode in enumerate(all_seq_react_this_subject):
        #     filtered_episode = [(state, duration, gap) for state, duration, gap in episode if state in all_states]
        #     print(f"Episode {i}: {filtered_episode}")
        #     # I get transitions from and to the state ... that's not ok. 
        # from ipdb import set_trace; set_trace()
        
        # print("Coactivation frequencies for the first 5 states:")
        # for subset_size, count in subset_counts.items():
        #     print(f"{subset_size} states (tol=-1 = strict overlap)together: {count} times; average overlap: {ave_overlap_size[subset_size]}")


        NP1 = (0, 1)
        NP2 = (np2_idx, np2_idx+1)
        # summary = compute_np_significance(all_sync_react_this_subject, NP1, NP2, all_states)
        # all_subjects_summary.append(summary)
        
    coactivation_df = pd.DataFrame(all_subjects_summary)
    behav_df = pd.DataFrame(behav_df)
    return ave_preds_all_subs, behav_df, coactivation_df, all_sync_coactivation_matrices, all_seq_coactivation_matrices



def make_all_reactivation_plots(behav_df, ave_preds_all_subs, res_dir, add_str, do_rel):
    Props = Properties if do_rel else Properties[0:2] + Properties[3:5]
    D = ave_preds_all_subs
    if do_rel: 
        present = [D["Shape1_present"], D["Colour1_present"], D["Relation_present"], D["Shape2_present"], D["Colour2_present"]]
        absent = [D["Shape1_absent"], D["Colour1_absent"], D["Relation_absent"], D["Shape2_absent"], D["Colour2_absent"]]
    else:
        present = [D["Shape1_present"], D["Colour1_present"], D["Shape2_present"], D["Colour2_present"]]
        absent = [D["Shape1_absent"], D["Colour1_absent"], D["Shape2_absent"], D["Colour2_absent"]]
    plot_average_preds_seaborn(present, absent, labels=Props, out_fn=f"{res_dir}/average_preds_scenes_trained_scenes_tested_sns_t{add_str}.png")

    # ## Not averaged. Meaningless. Why?
    # # does_reactivations_predict_behavioral(behav_df.query("Condition=='Present'"), out_fn=f"{res_dir}/regplot_perf_react_present_t{add_str}.png")
    # does_reactivations_predict_behavioral(behav_df.query("Condition=='Difference'"), out_fn=f"{res_dir}/regplot_perf_react_diff_t{add_str}.png", scatter=False)
    # for prop in Props:
    #     local_df = behav_df.query(f"Property=='{prop}'")
    #     # does_reactivations_predict_behavioral(local_df.query("Condition=='Present'"), out_fn=f"{res_dir}/regplot_perf_react_present_{prop}_t{add_str}.png")
    #     does_reactivations_predict_behavioral(local_df.query("Condition=='Difference'"), out_fn=f"{res_dir}/regplot_perf_react_diff_{prop}_t{add_str}.png", scatter=False)
    # #     # does_reactivations_predict_behavioral(local_df.query("Condition=='Present'"), y="RT", out_fn=f"{res_dir}/regplot_perf_react_present_RT_{prop}_t{add_str}.png")
    # #     # does_reactivations_predict_behavioral(local_df.query("Condition=='Difference'"), y="RT", out_fn=f"{res_dir}/regplot_perf_react_diff_RT_{prop}_t{add_str}.png")

    # # average over trials ~ subjects with overall more reactivations are overall better... but no single trial prediction! 
    # ave_df = behav_df.groupby(["Subject", "Condition", "Property"]).mean("Performance").reset_index()
    # # doesn not work, pval=1 in all cases ... 
    # # # does_reactivations_predict_behavioral(ave_df.query("Condition=='Present'"), out_fn=f"{res_dir}/regplot_perf_react_present_aveTrials_t{add_str}.png")
    # # does_reactivations_predict_behavioral(ave_df.query("Condition=='Difference'"), out_fn=f"{res_dir}/regplot_perf_react_diff_aveTrials_t{add_str}.png")
    # # does_reactivations_predict_behavioral(ave_df.query("Condition=='Difference'"), y='RT', out_fn=f"{res_dir}/regplot_perf_react_diff_RT_aveTrials_t{add_str}.png")
    # for prop in Props:
    #     local_df = ave_df.query(f"Property=='{prop}'")
    #     # does_reactivations_predict_behavioral(local_df.query("Condition=='Present'"), out_fn=f"{res_dir}/regplot_perf_react_present_{prop}_aveTrials_t{add_str}.png")
    #     does_reactivations_predict_behavioral(local_df.query("Condition=='Difference'"), out_fn=f"{res_dir}/regplot_perf_react_diff_{prop}_aveTrials_t{add_str}.png")
    #     # does_reactivations_predict_behavioral(local_df.query("Condition=='Present'"), y="RT", out_fn=f"{res_dir}/regplot_perf_react_present_RT_{prop}_aveTrials_t{add_str}.png")
    #     does_reactivations_predict_behavioral(local_df.query("Condition=='Difference'"), y="RT", out_fn=f"{res_dir}/regplot_perf_react_diff_RT_{prop}_aveTrials_t{add_str}.png")


gen_cond = "scenes"
subs = df['sub'].unique()
for label in ["Prop0", "PropAll"]: # , 
    print(f"Doing label {label}")
    df_prop = df[df["label"].str.contains(label, na=False)]
    do_rel = True if label=="PropAll" else False

    for train_cond in ["localizer_one_object_two_objects", "localizer_two_objects", "two_objects"]:
        print(f"Doing train condition {train_cond}")
        df_train = df_prop.query(f"train_cond == '{train_cond}'")

        for t in ["0.2", "0.3", "0.4", "0.6", "0.8"]:
            add_str = f"{t}_{label}_{train_cond}"
            df_t = df_train[df_train["label"].str.contains(t, na=False)] # keep only the current training time
            if not len(df_t): 
                print(f"No data for {label}, traincond={train_cond} t={t} s")
                continue
            # mask = df_train["label"].str.contains(t, na=False).to_list()  # get the corresponding binary mask - Dont do that. We are using the indices from the df, they match the list. If you change the list then the indices do not match anymore! 
            # preds_data_t = [pred for pred, keep in zip(all_preds_data, mask) if keep] # we need to do this because all_preds_data is a list
            # labels = [f"{l}{t}_1" for l in ["S1", "C1", "R", "S2", "C2"]]
            # ave_preds_all_subs, behav_df, coactivation_df, sf, sb, sr = get_preds_and_sequenceness_for_cond(df_t, \
            #                                                             preds_data_t, train_cond, gen_cond, labels, do_rel=do_rel)
            ave_preds_all_subs, behav_df, coactivation_df, sync_coactivation_matrices, seq_coactivation_matrices = get_replays_sophie_style(df_t, all_preds_data, train_cond, gen_cond, do_rel=do_rel)

            # synchronous coactivations
            ave_coactivation_matrix = average_coactivation_matrices(sync_coactivation_matrices)
            coact_mat_out_fn = f"{res_dir}/synchronous_coactivation_matrix_{add_str}.png"
            half_labels = properties if do_rel else properties[0:2] + properties[3:5]
            # abs_idx = -3 if do_rel else -2
            abs_idx = -2
            full_labels = [f"{l}_p" for l in half_labels] + [f"{l}_a" for l in half_labels[0:abs_idx]]
            plot_coactivation_heatmap(ave_coactivation_matrix, full_labels, coact_mat_out_fn)
            # for sub, matrix in zip(subs, sync_coactivation_matrices):
            #     plot_coactivation_heatmap(matrix, full_labels, f"{res_dir}/synchronous_coactivation_matrix_{add_str}_{sub}.png")

            # sequential coactivations
            ave_coactivation_matrix = average_coactivation_matrices(seq_coactivation_matrices)
            coact_mat_out_fn = f"{res_dir}/sequential_coactivation_matrix_{add_str}.png"
            half_labels = properties if do_rel else properties[0:2] + properties[3:5]
            # from ipdb import set_trace; set_trace()
            full_labels = [f"{l}_p" for l in half_labels] + [f"{l}_a" for l in half_labels[0:abs_idx]]
            plot_coactivation_heatmap(ave_coactivation_matrix, full_labels, coact_mat_out_fn)
            # for sub, matrix in zip(subs, seq_coactivation_matrices):
            #     plot_coactivation_heatmap(matrix, full_labels, f"{res_dir}/seqential_coactivation_matrix_{add_str}_{sub}.png")


            # group_stats = coactivation_df.describe().T[['mean', 'std', '50%']]  # 50% is median
            # group_stats.rename(columns={'50%': 'median'}, inplace=True)
            # print(group_stats)
            # # Compare NP1 vs. mean of all pairs
            # try:
            #     t_stat, p_value = ttest_rel(coactivation_df["NP1_z_score"], coactivation_df["NP2_z_score"])
            # except:
            #     from ipdb import set_trace; set_trace()
            # print(f"Paired t-test between NP1 and NP2: t = {t_stat:.3f}, p = {p_value:.3f}")

            # sns.boxplot(data=coactivation_df[["NP1_z_score", "NP2_z_score"]])
            # plt.title("Z-Scores of NP1 vs. NP2 Across Subjects")
            # plt.ylabel("Z-Score")
            # # plt.show()
            # plt.savefig(f"{res_dir}/NP1_vs_NP2_zscore_boxplot_t{add_str}.png")
            # plt.close()
            
            make_all_reactivation_plots(behav_df, ave_preds_all_subs, res_dir, add_str, do_rel)


# from ipdb import set_trace; set_trace()
exit()

# does_reactivations_predict_behavioral(behav_df.query("Condition=='Present'"), out_fn=f"{res_dir}/regplot_perf_react_present.png")
# does_reactivations_predict_behavioral(behav_df.query("Condition=='Difference'"), out_fn=f"{res_dir}/regplot_perf_react_diff.png")

# does_reactivations_predict_behavioral(behav_df.query("Condition=='Present'"), y="RT", out_fn=f"{res_dir}/regplot_perf_react_present_RT.png")
# does_reactivations_predict_behavioral(behav_df.query("Condition=='Difference'"), y="RT", out_fn=f"{res_dir}/regplot_perf_react_diff_RT.png")

print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")