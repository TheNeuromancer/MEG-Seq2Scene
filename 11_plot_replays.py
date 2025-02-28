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
# all_preds_data = [p[int(args.sfreq * 1)::] for p in all_preds_data]
# all_preds_data = [p[100::] for p in all_preds_data]

# print(F"Only keeping Complexity==2 trials, because else the repeated states fucks up the replays analyses")
# df = df.query(f"Complexity==2")


# def get_replays_word_and_position(df, all_preds_data, train_cond, gen_cond, do_rel=True, min_len=10):
#     """ Get sequential and synchronous reactivations
#     for a prefiltered train and gen cond
#     aligns word reactivation with their word positions
#     min_len (int): minimum length of reactivation to be considered an event 
#     """
#     subs = df['sub'].unique()
#     n_subs = len(subs)
#     print(f"Found {n_subs} subjects")
#     if df['train_cond'].nunique() > 1 or df['gen_cond'].nunique() > 1:
#         print(f"More than one train or gen condition in the dataframe; it should be pre-filtered before being fed to the func!")
#         from ipdb import set_trace; set_trace()
#     props = Properties if do_rel else Properties[0:2] + Properties[3:5]
#     word_pos = [1,2,3,4,5] if do_rel else [1,2,4,5]
#     ave_preds_all_subs = {f"{prop}_present": [] for prop in props}
#     for prop_abs in props[0:-2]: ave_preds_all_subs[f"{prop_abs.rstrip('1')}_absent"] = [] # same as present but withtout second shape and colors. Also remove the "1" in Shape1
#     # for prop_abs in ["Shape", "Colour", "Relation"]: ave_preds_all_subs[f"{prop_abs}_absent"] = []
#     behav_df = {"Subject": [], "Condition": [], "Property": [], "Reactivation": [], "Performance": [], "RT": []}
#     all_subjects_summary = []
#     all_sync_coactivation_matrices, all_seq_coactivation_matrices = [], []
#     all_ind_seq_coactivation_matrices = []
#     all_sync_overlap_size = []

#     threshold_per_sub = {}
#     for iSub, sub in tqdm(enumerate(subs)):
#         df_sub = df.query(f"sub=={sub}")
#         trial_ids = df_sub.trial_id.unique()
#         n_trials = len(trial_ids)

#         # get subject threshold
#         indices = df_sub.index.values.astype(int)
#         preds = np.array([all_preds_data[i] for i in indices])
#         # for i in range(preds.shape[2]): preds[:,:,i] = zscore(preds[:,:,i]) # normalizing each prop
#         thresholds = [np.percentile(preds[:,:,i], 95) for i in range(preds.shape[2])] # property-specifc threshold
#         # thresholds = [2] * preds.shape[2] # property-specifc threshold ,but same because preds are normalized
#         threshold_per_sub[sub] = thresholds

#         all_sync_react_this_subject = []
#         all_seq_react_this_subject = []

#         # ave_preds_this_sub_by_presence = {f"{prop}_{presence}": [] for presence in ['present', 'absent'] for prop in Properties} # for this subject and lag
#         ave_preds_this_sub_by_presence = {f"{prop}_present": [] for prop in props}
#         for prop_abs in props[0:-2]: ave_preds_this_sub_by_presence[f"{prop_abs.rstrip('1')}_absent"] = [] # same as present but withtout second shape and colors. Also remove the "1" in Shape1
#         # storing the preds depending on wether the words was present once or repeated. TODO: actually use this
#         ave_preds_this_sub_by_presence_adv = {f"{prop}_{presence}": [] for presence in ['present_once', 'present_twice', 'absent'] for prop in Properties} # for this subject and lag
#         for iTrial in range(n_trials):
#             df_trial = df_sub.query(f"trial_id=='{trial_ids[iTrial]}'")
#             if len(df_trial) > 1:
#                 print(f"Found more than one entry for trial {iTrial}: {trial_ids[iTrial]}")

#             s1, c1, rel, s2, c2 = df_trial.iloc[0][Properties].values

#             # if you reset the index and give the data from this subject only, that works (I think)
#             # df_trial = df_trial.reset_index()
#             # preds_props = get_trial_preds_from_data_with_idx(iTrial, preds) # shape n_times * n_classes (ordered as shapes, colors, relations)
#             preds_props = get_trial_preds_from_data(df_trial, all_preds_data) # shape n_times * n_classes (ordered as shapes, colors, relations)
            
#             # get predictions over the whole window, for each property, depending on whether it is present or absent
#             # present: list arrays of shape (n_samples, n_states) for present, len=3 to 5 (depending on how many present properties)
#             # absent: same, len 3 to 5 (because if only 3 words are presented, then there are 5 absent words)
#             present, absent, present_words, absent_words = get_present_or_absent_preds_one_trial_v2(preds_props, [s1, c1, rel, s2, c2], do_rel=do_rel)
#             ave_present = [p.mean() for p in present]
#             ave_absent = [p.mean() for p in absent]

#             present_props = words2props(present_words)
#             absent_props = words2props(absent_words)

#             # update the dict of ave_preds_this_sub_by_presence
#             ave_preds_this_sub_by_presence = update_present_or_absent_preds_one_trial_v2(ave_preds_this_sub_by_presence, 
#                                                                     ave_present, ave_absent, present_props, absent_props)

#             # behavioral results
#             perf, RT = df_trial["Perf"].values[0], df_trial["RT"].values[0]
#             behav_df = update_behav_df_v2(behav_df, sub, perf, RT, ave_present, ave_absent, present_props, absent_props)

#             # get Sophie-style reactivations
#             preds_this_trial = present + absent
#             labels_this_trial = [f"{p}_present" for p in present_props] + [f"{p}_absent" for p in absent_props]
#             # signif_react = get_significant_reactivations(preds_this_trial, threshold=threshold)
#             signif_react = get_significant_reactivations_v2(preds_this_trial, thresholds)
#             consecutive_react = get_reactivation_episodes(signif_react) # list of tuples: [(start, end, state, duration), ...] for all reactivation episodes.

#             # filter consecutive reactivations of length min
#             consecutive_react = [react for react in consecutive_react if react[3] > min_len]

#             # synchronous_episodes_pairs = get_synchronous_reactivations_pairs(consecutive_react) # (list of tuples): [(state1, state2, overlap_start, overlap_duration), ...]
#             synchronous_episodes = get_synchronous_reactivations(consecutive_react, tolerance=0) # list of tuples: [(states, overlap_start, overlap_duration)]
#             # go from state number to labels, Synchronous episodes is list of tuples: [(states, overlap_start, overlap_duration)]
#             synchronous_episodes = [({labels_this_trial[s] for s in states}, start, duration) for states, start, duration in synchronous_episodes]
#             # print(synchronous_episodes)
#             all_sync_react_this_subject.extend(synchronous_episodes)
            
#             sequential_episodes = get_sequential_reactivations(consecutive_react, iLag=10) # (list of lists): [[(state1, duration1, gap1), (state2, duration2, gap2), ...], ...]
#             # go from state number to labels
#             sequential_episodes = [[(labels_this_trial[state], duration, gap) for state, duration, gap in ep] for ep in sequential_episodes]
#             # Rmove episodes with only one state (A->A)
#             sequential_episodes = [episode for episode in sequential_episodes if len(set(state for state, _, _ in episode)) > 1]
#             all_seq_react_this_subject.extend(sequential_episodes)

#         ## For this subject, get the average of the predictions
#         ave_preds_all_subs = get_subj_ave_preds(ave_preds_this_sub_by_presence, ave_preds_all_subs) # update the dict of averages

#         # n_states = 8 if do_rel else 6
#         # all_states = set(range(n_states))         
#         all_states = ave_preds_all_subs.keys()
#         subset_counts, ave_overlap_size, coactivation_matrix = count_synchronous_coactivations(all_sync_react_this_subject, all_states)
#         all_sync_coactivation_matrices.append(coactivation_matrix)
#         all_sync_overlap_size.append(ave_overlap_size)
#         # coactivation_matrix_, overlap_duration_matrix, unique_states = compute_coactivation_and_overlap(all_sync_react_this_subject)


#         coactivation_counts, ave_durations, ave_gaps, coactivation_matrix = count_sequential_coactivations(all_seq_react_this_subject, all_states)
#         all_seq_coactivation_matrices.append(coactivation_matrix)
#         # coactivation_matrix_, unique_states = compute_directional_coactivation_matrix(all_seq_react_this_subject)


#         ind_coactivation_counts, ind_ave_durations, ind_ave_gaps, ind_coactivation_matrix = count_indirect_sequential_coactivations(all_seq_react_this_subject, all_states)
#         all_ind_seq_coactivation_matrices.append(ind_coactivation_matrix)

#         # print("Coactivation Matrix (State Transitions):")
#         # for state1, transitions in coactivation_matrix.items():
#         #     for state2, count in transitions.items():
#         #         if count > 0:
#         #             print(f"{state1} -> {state2}: {count}")

#         # for i, episode in enumerate(all_seq_react_this_subject):
#         #     filtered_episode = [(state, duration, gap) for state, duration, gap in episode if state in all_states]
#         #     print(f"Episode {i}: {filtered_episode}")
#         #     # I get transitions from and to the state ... that's not ok. 
#         # from ipdb import set_trace; set_trace()
        
#         # print("Coactivation frequencies for the first 5 states:")
#         # for subset_size, count in subset_counts.items():
#         #     print(f"{subset_size} states (tol=-1 = strict overlap)together: {count} times; average overlap: {ave_overlap_size[subset_size]}")

#     # print(F"Thresholds per subject: {threshold_per_sub}")
 
#     coactivation_df = pd.DataFrame(all_subjects_summary)
#     behav_df = pd.DataFrame(behav_df)
#     return ave_preds_all_subs, behav_df, coactivation_df, all_sync_coactivation_matrices, all_seq_coactivation_matrices, all_ind_seq_coactivation_matrices



def get_replays_sophie_style(df, all_preds_data, train_cond, gen_cond, labels, min_len=5):
    """ Get sequential and synchronous reactivations
    for a prefiltered train and gen cond
    """
    subs = df['sub'].unique()
    n_subs = len(subs)
    print(f"Found {n_subs} subjects")
    if df['train_cond'].nunique() > 1 or df['gen_cond'].nunique() > 1:
        print(f"More than one train or gen condition in the dataframe; it should be pre-filtered before being fed to the func!")
        from ipdb import set_trace; set_trace()
    
    ave_preds_all_subs = {label: [] for label in labels}
    behav_df = {"Subject": [], "Condition": [], "Property": [], "Reactivation": [], "Performance": [], "RT": []}
    all_subjects_summary = []
    all_sync_coactivation_matrices, all_seq_coactivation_matrices = [], []
    all_ind_seq_coactivation_matrices = []
    all_sync_overlap_size = []
    
    threshold_per_sub = {}

    for iSub, sub in tqdm(enumerate(subs)):
        df_sub = df.query(f"sub=={sub}")
        trial_ids = df_sub.trial_id.unique()
        n_trials = len(trial_ids)

        indices = df_sub.index.values.astype(int)
        preds = np.array([all_preds_data[i] for i in indices])
        thresholds = [np.percentile(preds[:,:,i], 95) for i in range(preds.shape[2])]
        threshold_per_sub[sub] = thresholds

        all_sync_react_this_subject = []
        all_seq_react_this_subject = []

        ave_preds_this_sub_by_presence = {label: [] for label in labels}
        
        for iTrial in range(n_trials):
            df_trial = df_sub.query(f"trial_id=='{trial_ids[iTrial]}'")
            if len(df_trial) > 1:
                print(f"Found more than one entry for trial {iTrial}: {trial_ids[iTrial]}")
            
            preds_props = get_trial_preds_from_data(df_trial, all_preds_data)
            present, absent, present_words, absent_words = get_present_or_absent_preds_one_trial_v2(preds_props, df_trial.iloc[0][Properties].values, do_rel=do_rel)
            ave_present = [p.mean() for p in present]
            ave_absent = [p.mean() for p in absent]
            
            present_props = words2props(present_words)
            absent_props = words2props(absent_words)

            ave_preds_this_sub_by_presence = update_present_or_absent_preds_one_trial_v2(
                ave_preds_this_sub_by_presence, ave_present, ave_absent, present_props, absent_props)

            perf, RT = df_trial["Perf"].values[0], df_trial["RT"].values[0]
            behav_df = update_behav_df_v2(behav_df, sub, perf, RT, ave_present, ave_absent, present_props, absent_props)

            preds_this_trial = present + absent
            labels_this_trial = [f"{p}_present" for p in present_props] + [f"{p}_absent" for p in absent_props]
            signif_react = get_significant_reactivations_v2(preds_this_trial, thresholds)
            consecutive_react = get_reactivation_episodes(signif_react)
            consecutive_react = [react for react in consecutive_react if react[3] > min_len]
            
            synchronous_episodes = get_synchronous_reactivations(consecutive_react, tolerance=0)
            synchronous_episodes = [({labels_this_trial[s] for s in states}, start, duration) for states, start, duration in synchronous_episodes]
            all_sync_react_this_subject.extend(synchronous_episodes)
            
            sequential_episodes = get_sequential_reactivations(consecutive_react, iLag=10)
            sequential_episodes = [[(labels_this_trial[state], duration, gap) for state, duration, gap in ep] for ep in sequential_episodes]
            sequential_episodes = [episode for episode in sequential_episodes if len(set(state for state, _, _ in episode)) > 1]
            all_seq_react_this_subject.extend(sequential_episodes)

        ave_preds_all_subs = get_subj_ave_preds(ave_preds_this_sub_by_presence, ave_preds_all_subs)
        all_states = ave_preds_all_subs.keys()
        subset_counts, ave_overlap_size, coactivation_matrix = count_synchronous_coactivations(all_sync_react_this_subject, all_states)
        all_sync_coactivation_matrices.append(coactivation_matrix)
        all_sync_overlap_size.append(ave_overlap_size)
        
        coactivation_counts, ave_durations, ave_gaps, coactivation_matrix = count_sequential_coactivations(all_seq_react_this_subject, all_states)
        all_seq_coactivation_matrices.append(coactivation_matrix)
        
        ind_coactivation_counts, ind_ave_durations, ind_ave_gaps, ind_coactivation_matrix = count_indirect_sequential_coactivations(all_seq_react_this_subject, all_states)
        all_ind_seq_coactivation_matrices.append(ind_coactivation_matrix)
    
    coactivation_df = pd.DataFrame(all_subjects_summary)
    behav_df = pd.DataFrame(behav_df)
    return ave_preds_all_subs, behav_df, coactivation_df, all_sync_coactivation_matrices, all_seq_coactivation_matrices, all_ind_seq_coactivation_matrices


def make_all_reactivation_plots(behav_df, ave_preds_all_subs, res_dir, add_str, do_rel):
    Props = Properties if do_rel else Properties[0:2] + Properties[3:5]
    D = ave_preds_all_subs
    if do_rel: 
        labels = ["Shape", "Relation", "Color"]
        present = [D["Shape1_present"] + D["Shape2_present"], D["Relation_present"], D["Colour1_present"] + D["Colour2_present"]]
        absent = [D["Shape_absent"], D["Relation_absent"], D["Colour_absent"]]
    else:
        labels = ["Shape", "Color"]
        present = [D["Shape1_present"] + D["Shape2_present"], D["Colour1_present"] + D["Colour2_present"]]
        absent = [D["Shape_absent"], D["Colour_absent"]]
    if do_rel: 
        plot_average_preds_seaborn([present[1]], [absent[1]], labels=["Relation"], out_fn=f"{res_dir}/average_preds_rel_scenes_trained_scenes_tested_sns_t{add_str}.png")
        plot_average_preds_seaborn(present, absent, labels, out_fn=f"{res_dir}/average_preds_scenes_trained_scenes_tested_sns_t{add_str}.png")
    else:
        plot_average_preds_seaborn(present, absent, labels, out_fn=f"{res_dir}/average_preds_scenes_trained_scenes_tested_sns_t{add_str}.png")
    # plot_average_preds_seaborn(present, absent, labels, out_fn=f"{res_dir}/average_preds_scenes_trained_scenes_tested_sns_t{add_str}.png")

    # overall = averaged probabilities for each property (excluding relation that has a different baseline)
    does_reactivations_predict_behavioral(behav_df.query("Condition=='Overall Difference'"), out_fn=f"{res_dir}/regplot_perf_react_overalldiff_t{add_str}.png", scatter=False)

    ## Not averaged. Meaningless. Why? Maybe because there are multiple entries for each trial (S1, C1, ...) predicting the same thing.
    does_reactivations_predict_behavioral(behav_df.query("Condition=='Present'"), out_fn=f"{res_dir}/regplot_perf_react_present_t{add_str}.png")
    # does_reactivations_predict_behavioral(behav_df.query("Condition=='Difference'"), out_fn=f"{res_dir}/regplot_perf_react_diff_t{add_str}.png", scatter=False)
    for prop in Props:
        local_df = behav_df.query(f"Property=='{prop}'")
        does_reactivations_predict_behavioral(local_df.query("Condition=='Present'"), out_fn=f"{res_dir}/regplot_perf_react_present_{prop}_t{add_str}.png")
        # does_reactivations_predict_behavioral(local_df.query("Condition=='Difference'"), out_fn=f"{res_dir}/regplot_perf_react_diff_{prop}_t{add_str}.png", scatter=False)
        does_reactivations_predict_behavioral(local_df.query("Condition=='Present'"), y="RT", out_fn=f"{res_dir}/regplot_perf_react_present_RT_{prop}_t{add_str}.png")
    #     # does_reactivations_predict_behavioral(local_df.query("Condition=='Difference'"), y="RT", out_fn=f"{res_dir}/regplot_perf_react_diff_RT_{prop}_t{add_str}.png")

    # average over trials ~ subjects with overall more reactivations are overall better... but no single trial prediction! 
    ave_df = behav_df.groupby(["Subject", "Condition", "Property"]).mean("Performance").reset_index()
    # doesn not work, pval=1 in all cases ... 
    # # does_reactivations_predict_behavioral(ave_df.query("Condition=='Present'"), out_fn=f"{res_dir}/regplot_perf_react_present_aveTrials_t{add_str}.png")
    does_reactivations_predict_behavioral(ave_df.query("Condition=='Overall Difference'"), out_fn=f"{res_dir}/regplot_perf_react_overalldiff_aveTrials_t{add_str}.png", model_name='sub')
    # does_reactivations_predict_behavioral(ave_df.query("Condition=='Difference'"), y='RT', out_fn=f"{res_dir}/regplot_perf_react_diff_RT_aveTrials_t{add_str}.png")
    for prop in Props:
        local_df = ave_df.query(f"Property=='{prop}'")
        does_reactivations_predict_behavioral(local_df.query("Condition=='Present'"), out_fn=f"{res_dir}/regplot_perf_react_present_{prop}_aveTrials_t{add_str}.png", model_name='sub')
        # does_reactivations_predict_behavioral(local_df.query("Condition=='Difference'"), out_fn=f"{res_dir}/regplot_perf_react_diff_{prop}_aveTrials_t{add_str}.png")
        does_reactivations_predict_behavioral(local_df.query("Condition=='Present'"), y="RT", out_fn=f"{res_dir}/regplot_perf_react_present_RT_{prop}_aveTrials_t{add_str}.png", model_name='sub')
        # does_reactivations_predict_behavioral(local_df.query("Condition=='Difference'"), y="RT", out_fn=f"{res_dir}/regplot_perf_react_diff_RT_{prop}_aveTrials_t{add_str}.png")


gen_cond = "scenes"
subs = df['sub'].unique()
for label in ["PropAll", "Prop0"]: 
    print(f"Doing label {label}")
    df_prop = df[df["label"].str.contains(label, na=False)]
    do_rel = True if label=="PropAll" else False

    for train_cond in ["localizer_one_object_two_objects"]: #, "localizer_two_objects", "two_objects"]:
        print(f"Doing train condition {train_cond}")
        df_train = df_prop.query(f"train_cond == '{train_cond}'")
        df_wordpos = df[df["label"].str.contains("WordPos", na=False) & (df["train_cond"] == train_cond)]

        for t in ["0.2", "0.3", "0.4", "0.6", "0.8"]:
            add_str = f"{t}_{label}_{train_cond}"
            df_t = df_train[df_train["label"].str.contains(t, na=False)] # keep only the current training time
            if not len(df_t): 
                print(f"No data for {label}, traincond={train_cond} t={t} s")
                continue
            
            # results = get_replays_sophie_style(df_t, all_preds_data, train_cond, gen_cond, do_rel=do_rel)
            props = Properties if do_rel else Properties[0:2] + Properties[3:5]
            labels = [f"{prop}_present" for prop in props] + [f"{prop.rstrip('1')}_absent" for prop in props[0:-2]]
            results = get_replays_sophie_style(df_t, all_preds_data, train_cond, gen_cond, labels)
            ave_preds_all_subs, behav_df, coactivation_df, sync_coactivation_matrices, seq_coactivation_matrices, ind_seq_coactivation_matrices = results

            # synchronous coactivations
            ave_coactivation_matrix = average_coactivation_matrices(sync_coactivation_matrices)
            coact_mat_out_fn = f"{res_dir}/synchronous_coactivation_matrix_{add_str}.png"
            half_labels = properties if do_rel else properties[0:2] + properties[3:5]
            abs_idx = 3 if do_rel else 2
            red_cross_idx = 5 if do_rel else 4
            full_labels = [f"{l}_p" for l in half_labels] + [f"{l}_a" for l in half_labels[0:abs_idx]]
            plot_coactivation_heatmap(ave_coactivation_matrix, full_labels, coact_mat_out_fn, red_cross_idx=red_cross_idx)

            # synchronous coactivations without absent properties
            filtered_keys = [state for state in ave_coactivation_matrix if "absent" not in state] # Identify states to keep
            # Create a new dictionary with only the filtered keys
            ave_coactivation_matrix_no_abs = {state: {other_state: value for other_state, value in ave_coactivation_matrix[state].items() 
                                       if "absent" not in other_state} for state in filtered_keys}
            coact_mat_out_fn_no_abs = f"{res_dir}/synchronous_coactivation_matrix_no_abs_{add_str}.png"
            full_labels_no_abs = [f"{l}_p" for l in half_labels]
            plot_coactivation_heatmap(ave_coactivation_matrix_no_abs, full_labels_no_abs, coact_mat_out_fn_no_abs)

            # for sub, matrix in zip(subs, sync_coactivation_matrices):
            #     plot_coactivation_heatmap(matrix, full_labels, f"{res_dir}/synchronous_coactivation_matrix_{add_str}_{sub}.png")

            # sequential coactivations
            ave_coactivation_matrix = average_coactivation_matrices(seq_coactivation_matrices)
            coact_mat_out_fn = f"{res_dir}/sequential_coactivation_matrix_{add_str}.png"
            half_labels = properties if do_rel else properties[0:2] + properties[3:5]
            full_labels = [f"{l}_p" for l in half_labels] + [f"{l}_a" for l in half_labels[0:abs_idx]]
            plot_coactivation_heatmap(ave_coactivation_matrix, full_labels, coact_mat_out_fn, red_cross_idx=red_cross_idx)

            # sequential coactivations without absent properties
            filtered_keys = [state for state in ave_coactivation_matrix if "absent" not in state] # Identify states to keep
            # Create a new dictionary with only the filtered keys
            ave_coactivation_matrix_no_abs = {state: {other_state: value for other_state, value in ave_coactivation_matrix[state].items() 
                                       if "absent" not in other_state} for state in filtered_keys}
            coact_mat_out_fn_no_abs = f"{res_dir}/sequential_coactivation_matrix_no_abs_{add_str}.png"
            full_labels_no_abs = [f"{l}_p" for l in half_labels]
            plot_coactivation_heatmap(ave_coactivation_matrix_no_abs, full_labels_no_abs, coact_mat_out_fn_no_abs)

            # for sub, matrix in zip(subs, seq_coactivation_matrices):
            #     plot_coactivation_heatmap(matrix, full_labels, f"{res_dir}/seqential_coactivation_matrix_{add_str}_{sub}.png")


            # sequential coactivations with indirect transtion (A -> B -> C also count a coactivation for A -> C)
            ave_coactivation_matrix = average_coactivation_matrices(ind_seq_coactivation_matrices)
            coact_mat_out_fn = f"{res_dir}/indirect_sequential_coactivation_matrix_{add_str}.png"
            half_labels = properties if do_rel else properties[0:2] + properties[3:5]
            full_labels = [f"{l}_p" for l in half_labels] + [f"{l}_a" for l in half_labels[0:abs_idx]]
            plot_coactivation_heatmap(ave_coactivation_matrix, full_labels, coact_mat_out_fn, red_cross_idx=red_cross_idx)

            # sequential coactivations without absent properties
            filtered_keys = [state for state in ave_coactivation_matrix if "absent" not in state] # Identify states to keep
            # Create a new dictionary with only the filtered keys
            ave_coactivation_matrix_no_abs = {state: {other_state: value for other_state, value in ave_coactivation_matrix[state].items() 
                                       if "absent" not in other_state} for state in filtered_keys}
            coact_mat_out_fn_no_abs = f"{res_dir}/indirect_sequential_coactivation_matrix_no_abs_{add_str}.png"
            full_labels_no_abs = [f"{l}_p" for l in half_labels]
            plot_coactivation_heatmap(ave_coactivation_matrix_no_abs, full_labels_no_abs, coact_mat_out_fn_no_abs)


            ## Compare specific pairs
            pairs = [
                     (("Shape1_present", "Colour1_present"), ("Shape1_present", "Colour2_present")), # S1C1 > S1C2
                     (("Shape2_present", "Colour2_present"), ("Shape2_present", "Colour1_present")),

                     (("Shape1_present", "Colour1_present"), ("Shape2_present", "Colour1_present")), # S1C1 > S2C1
                     (("Shape2_present", "Colour2_present"), ("Shape1_present", "Colour2_present")),

                     (("Shape1_present", "Colour1_present"), ("Shape1_present", "Colour_absent")), # S1C1 > S1Cabs
                     (("Shape2_present", "Colour2_present"), ("Shape2_present", "Colour_absent")),

                     (("Shape1_present", "Colour1_present"), ("Shape_absent", "Colour1_present")), # S1C1 > SabsC1
                     (("Shape2_present", "Colour2_present"), ("Shape_absent", "Colour2_present")),

                     (("Shape1_present", "Colour1_present"), ("Colour1_present", "Shape1_present")), # S1C1 > C1S1
                     (("Shape2_present", "Colour2_present"), ("Colour2_present", "Shape2_present")),

                     (("Shape1_present", "Colour1_present"), ("Shape_absent", "Colour_absent")), # S1C1 > SabsCabs
                     (("Shape2_present", "Colour2_present"), ("Shape_absent", "Colour_absent")),
                     ]

            for i, pair in enumerate(pairs): # single pair plots
                df_pairs = compare_pairs(seq_coactivation_matrices, [pair], alternative='two-sided')
                plot_coactivation_comparison(df_pairs, f"{res_dir}/sequential_coactivation_pairs_comparison_two-sided_{add_str}_{i}.png")
                df_pairs_one_sided = compare_pairs(seq_coactivation_matrices, [pair], alternative='greater')
                plot_coactivation_comparison(df_pairs_one_sided, f"{res_dir}/sequential_coactivation_pairs_comparison_greater_{add_str}_{i}.png")

                df_pairs = compare_pairs(sync_coactivation_matrices, [pair], alternative='two-sided')
                plot_coactivation_comparison(df_pairs, f"{res_dir}/synchronous_coactivation_pairs_comparison_two-sided_{add_str}_{i}.png")
                df_pairs_one_sided = compare_pairs(sync_coactivation_matrices, [pair], alternative='greater')
                plot_coactivation_comparison(df_pairs_one_sided, f"{res_dir}/synchronous_coactivation_pairs_comparison_greater_{add_str}_{i}.png")


            # print(df_pairs)
            # print(df_pairs_one_sided)

            try:
                make_all_reactivation_plots(behav_df, ave_preds_all_subs, res_dir, add_str, do_rel)
            except:
                pass


# does_reactivations_predict_behavioral(behav_df.query("Condition=='Present'"), out_fn=f"{res_dir}/regplot_perf_react_present.png")
# does_reactivations_predict_behavioral(behav_df.query("Condition=='Difference'"), out_fn=f"{res_dir}/regplot_perf_react_diff.png")

# does_reactivations_predict_behavioral(behav_df.query("Condition=='Present'"), y="RT", out_fn=f"{res_dir}/regplot_perf_react_present_RT.png")
# does_reactivations_predict_behavioral(behav_df.query("Condition=='Difference'"), y="RT", out_fn=f"{res_dir}/regplot_perf_react_diff_RT.png")

print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")