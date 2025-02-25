import mne
import os.path as op
import os
from glob import glob
import pandas as pd
import numpy as np
import pickle
from random import choice
from collections import defaultdict
from scipy.stats import sem, pearsonr, zscore, ttest_ind
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning
# from scipy.linalg import toeplitz
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import statsmodels.formula.api as smf
import itertools
import warnings


from .params import *

def get_trial_preds_from_data(df_trial, all_preds_data):
    """ get the predictions for a given trial 
    df_trial should have a single entry. 
    """
    assert len(df_trial) == 1, f"len(df_trial)={len(df_trial)}"
    preds_idx = df_trial.index.values[0]
    preds = all_preds_data[preds_idx]
    return preds

# def get_present_or_absent_preds_one_trial(preds, props, do_rel):
#     """ take the predictions for a single trial
#     and returns them as 2 list of the 5 Properties
#     (Present and Absent), not averaged
#     """ 
#     # preds or shapes (0:3), then colors (3:6), then Relation (6:8).
#     s1, c1, rel, s2, c2 = props
#     Shape1_present = preds[:, 0:3][:, shapes.index(s1)]
#     Colour1_present = preds[:, 3:6][:, colors.index(c1)]
#     if do_rel:
#         Relation_present = preds[:, 6::][:, relations.index(rel)]
#     Shape2_present = preds[:, 0:3][:, shapes.index(s2)]
#     Colour2_present = preds[:, 3:6][:, colors.index(c2)]

#     shapes_absent = [s for s in shapes if s not in [s1, s2]]
#     colors_absent = [c for c in colors if c not in [c1, c2]]
#     relation_absent = [r for r in relations if r != rel][0]
#     Shape1_absent, Shape2_absent, Colour1_absent, Colour2_absent = [], [], [], []
#     for absent_shape in shapes_absent:
#         Shape1_absent.append(preds[:, 0:3][:, shapes.index(absent_shape)])
#         Shape2_absent.append(preds[:, 0:3][:, shapes.index(absent_shape)])
#     for absent_color in colors_absent:
#         Colour1_absent.append(preds[:, 3:6][:, colors.index(absent_color)])
#         Colour2_absent.append(preds[:, 3:6][:, colors.index(absent_color)])
#     if do_rel:
#         Relation_absent = [preds[:, 6::][:, relations.index(relation_absent)]] # cast to list, for consistency with other absents
#         return [Shape1_present, Colour1_present, Relation_present, Shape2_present, Colour2_present], [Shape1_absent, Colour1_absent, Relation_absent, Shape2_absent, Colour2_absent]
#     else:
#         return [Shape1_present, Colour1_present, Shape2_present, Colour2_present], [Shape1_absent, Colour1_absent, Shape2_absent, Colour2_absent]

def words2props(words):
    """ go from words ("triangle")
    to properties ("Shape")
    Keeping the number, if present 
    ("triangle1" -> "Shape1")
    """
    properties = []
    for word in words:
        if word in shapes:
            properties.append("Shape")
        elif word in colors:
            properties.append("Colour")
        elif word[0:-1] in shapes:
            properties.append("Shape" + word[-1])
        elif word[0:-1] in colors:
            properties.append("Colour" + word[-1])
        elif word in relations:
            properties.append("Relation")
    return properties


def get_present_or_absent_preds_one_trial_v2(preds, props, do_rel):
    """ take the predictions for a single trial
    and returns them as 2 lists of arrays of shape (n_samples, n_states)
    the length of the list depends on how many words were present or absent in this trial
    Also return a list that says what is in the list of lists
    order will always be S1, C1, R, S2, C2 BUT if S2 is absent (repeated S), then the 4th item will be C2 ... 
    preds or shapes (0:3), then colors (3:6), then Relation (6:8).
    """ 
    s1, c1, rel, s2, c2 = props
    present, absent = [], [] # probabilities
    present_words, absent_words = [], [] # corresponding properties
    present.append(preds[:, 0:3][:, shapes.index(s1)])
    present_words.append(f"{s1}1")
    present.append(preds[:, 3:6][:, colors.index(c1)])
    present_words.append(f"{c1}1")
    if do_rel:
        present.append(preds[:, 6::][:, relations.index(rel)])
        present_words.append(rel)
    if s1 != s2:
        present.append(preds[:, 0:3][:, shapes.index(s2)])
        present_words.append(f"{s2}2")
    if c1 != c2:
        present.append(preds[:, 3:6][:, colors.index(c2)])
        present_words.append(f"{c2}2")

    absent_shapes = [s for s in shapes if s not in [s1, s2]]
    absent_colors = [c for c in colors if c not in [c1, c2]]
    relation_absent = [r for r in relations if r != rel][0]
    absent.append(preds[:, 0:3][:, shapes.index(absent_shapes[0])])
    absent_words.append(absent_shapes[0])
    absent.append(preds[:, 3:6][:, colors.index(absent_colors[0])])
    absent_words.append(absent_colors[0])
    if do_rel:
        absent.append(preds[:, 6::][:, relations.index(relation_absent)])
        absent_words.append(relation_absent)
    if len(absent_shapes) == 2:
        absent.append(preds[:, 0:3][:, shapes.index(absent_shapes[1])])
        absent_words.append(absent_shapes[1])
    if len(absent_colors) == 2:
        absent.append(preds[:, 3:6][:, colors.index(absent_colors[1])])
        absent_words.append(absent_colors[1])

    return present, absent, present_words, absent_words


# def update_present_or_absent_preds_one_trial(preds_sub_by_presence, present, absent, do_rel):
#     """ update the dict with preselected present
#     and absent probabilities averaged over the window.
#     """ 
#     # preds of shape n_samples, n_properties: shape (0:3), then colors (3:6), then Relation (6:8).
#     if do_rel:
#         Shape1_present, Colour1_present, Relation_present, Shape2_present, Colour2_present = present
#         # Shape1_absent, Colour1_absent, Relation_absent, Shape2_absent, Colour2_absent = absent
#         Shape1_absent, Colour1_absent, Relation_absent = absent
#     else:
#         Shape1_present, Colour1_present, Shape2_present, Colour2_present = present
#         # Shape1_absent, Colour1_absent, Shape2_absent, Colour2_absent = absent
#         Shape1_absent, Colour1_absent = absent

#     preds_sub_by_presence["Shape1_present"].append(Shape1_present)
#     preds_sub_by_presence["Colour1_present"].append(Colour1_present)
#     if do_rel:
#         preds_sub_by_presence["Relation_present"].append(Relation_present)
#     preds_sub_by_presence["Shape2_present"].append(Shape2_present)
#     preds_sub_by_presence["Colour2_present"].append(Colour2_present)

#     # for absent_shape in shapes_absent:
#     preds_sub_by_presence["Shape1_absent"].append(Shape1_absent)
#     # preds_sub_by_presence["Shape2_absent"].append(Shape2_absent)
#     preds_sub_by_presence["Colour1_absent"].append(Colour1_absent)
#     # preds_sub_by_presence["Colour2_absent"].append(Colour2_absent)
#     if do_rel:
#         preds_sub_by_presence['Relation_absent'].append(Relation_absent)
#     return preds_sub_by_presence


def update_present_or_absent_preds_one_trial_v2(ave_preds_sub_by_presence, present, absent, present_props, absent_props):
    """ update the dict with preselected present
    and absent probabilities averaged over the window.
    """ 
    for pres_prob, pres_prop in zip(present, present_props):
        ave_preds_sub_by_presence[f"{pres_prop}_present"].append(pres_prob)
    for abs_prob, abs_prop in zip(absent, absent_props):
        ave_preds_sub_by_presence[f"{abs_prop}_absent"].append(abs_prob)
    return ave_preds_sub_by_presence


# def update_behav_df(behav_df, sub, perf, RT, ave_present, ave_absent, props):
#     """ Update the longform df with all behavioral results
#     """    
#     for pres_prop, preds in zip(props, ave_present):
#         behav_df["Subject"].append(sub)
#         behav_df["Condition"].append("Present")
#         behav_df["Property"].append(pres_prop)
#         behav_df["Reactivation"].append(preds.mean())
#         behav_df["Performance"].append(perf)
#         behav_df["RT"].append(RT)
#     for abs_prop, preds in zip(props, ave_absent):
#         behav_df["Subject"].append(sub)
#         behav_df["Condition"].append("Absent")
#         behav_df["Property"].append(abs_prop)
#         behav_df["Reactivation"].append(np.mean(preds))
#         behav_df["Performance"].append(perf)
#         behav_df["RT"].append(RT)
#     for i, (prop, _) in enumerate(zip(props, ave_absent)): # fancy trick to iterate only to the length of ave_absent, which is shorter than present and props (no 2nd item)
#         behav_df["Subject"].append(sub)
#         behav_df["Condition"].append("Difference")
#         behav_df["Property"].append(prop)
#         behav_df["Reactivation"].append(np.mean(ave_present[i]) - np.mean(ave_absent[i]))
#         behav_df["Performance"].append(perf)
#         behav_df["RT"].append(RT)
#     return behav_df


def update_behav_df_v2(behav_df, sub, perf, RT, ave_present, ave_absent, present_props, absent_props):
    """ Update the longform df with all behavioral results
    """    
    for pres_prop, preds in zip(present_props, ave_present):
        behav_df["Subject"].append(sub)
        behav_df["Condition"].append("Present")
        behav_df["Property"].append(pres_prop)
        behav_df["Reactivation"].append(preds.mean())
        behav_df["Performance"].append(perf)
        behav_df["RT"].append(RT)
    for abs_prop, preds in zip(absent_props, ave_absent):
        behav_df["Subject"].append(sub)
        behav_df["Condition"].append("Absent")
        behav_df["Property"].append(abs_prop)
        behav_df["Reactivation"].append(np.mean(preds))
        behav_df["Performance"].append(perf)
        behav_df["RT"].append(RT)
    # # overall reactivation for behavioral prediction (single per trial) - meaningless with the relation
    # behav_df["Subject"].append(sub)
    # behav_df["Condition"].append("Overall Difference (with Relation)")
    # behav_df["Property"].append("All with Rel")
    # behav_df["Reactivation"].append(np.mean(ave_present) - np.mean(ave_absent))
    # behav_df["Performance"].append(perf)
    # behav_df["RT"].append(RT)
    # overall but without the relation because chance level is different
    if "Relation" in present_props:
        ave_present.pop(present_props.index("Relation"))
        ave_absent.pop(absent_props.index("Relation"))
        behav_df["Subject"].append(sub)
        behav_df["Condition"].append("Overall Difference")
        behav_df["Property"].append("All")
        behav_df["Reactivation"].append(np.mean(ave_present) - np.mean(ave_absent))
        behav_df["Performance"].append(perf)
        behav_df["RT"].append(RT)
    # Cant do Difference because the number of present and absent do not always match (actually never matches: n_absent = 8 - n_present)
    # for i, prop in enumerate(props):
    #     behav_df["Subject"].append(sub)
    #     behav_df["Condition"].append("Difference")
    #     behav_df["Property"].append(prop)
    #     behav_df["Reactivation"].append(np.mean(ave_present[i]) - np.mean(ave_absent[i]))
    #     behav_df["Performance"].append(perf)
    #     behav_df["RT"].append(RT)
    return behav_df


# def get_subj_ave_preds(preds_by_presence_this_sub, ave_preds_all_subs, props):
#     """ Updates the across subjects dict with the values
#     for this subject
#     """
#     for prop in props:
#         for presence in ['present', 'absent']:
#             preds = preds_by_presence_this_sub[f"{prop}_{presence}"]
#             ave_preds_all_subs[f"{prop}_{presence}"].append(np.nanmean(preds))
#             if not len(preds): print(prop, presence)
#     return ave_preds_all_subs

def get_subj_ave_preds(preds_by_presence_this_sub, ave_preds_all_subs):
    """ Updates the across subjects dict with the values
    for this subject
    """
    for prop in preds_by_presence_this_sub.keys():
        preds = preds_by_presence_this_sub[prop]
        ave_preds_all_subs[prop].append(np.nanmean(preds))
        if not len(preds): print(prop, presence)
    return ave_preds_all_subs

# def restructure_data(present, absent, do_rel=True):
#     """Converts present and absent lists into a single list of arrays with corresponding labels.
#     NOT USED AFTER RESTRUCTURATION
#     Args:
#         present (list): List of (4 or) 5 arrays (n_samples, n_states) for present properties.
#         absent (list): List of lists of 1 or 2 arrays (n_samples, n_states) for absent properties.
#     Returns:
#         all_arrays (list of np.ndarray): Flattened list of (n_samples, n_states) arrays.
#         all_labels (list of tuples): Labels with (trial_idx, property, 'present'/'absent').
#     """
#     Props = Properties if do_rel else Properties[0:2] + Properties[3:5]
#     all_arrays, all_labels = [], []
#     for prop, arr in zip(Props, present): # Add present properties
#         all_arrays.append(arr)
#         all_labels.append((prop, "present"))
#     for prop, sublist in zip(Props, absent): # Add absent properties (list of lists)
#         for abs_idx, arr in enumerate(sublist):
#             all_arrays.append(arr)
#             all_labels.append((prop, "absent"))
#     return all_arrays, all_labels


# def remove_label_duplicates(preds, labels):
#     """
#     Removes one random occurrence of each duplicate entry in `labels`, 
#     ensuring the same elements are removed from `preds`.

#     Parameters:
#     - labels (list of tuples): List containing (feature, state) pairs.
#     - preds (list of numpy arrays): Corresponding prediction arrays.

#     Returns:
#     - filtered_labels (list of tuples): Labels with duplicates removed.
#     - filtered_preds (list of numpy arrays): Corresponding predictions.
#     """

#     # Dictionary to track first occurrence of each label
#     seen = {}
#     # List to store duplicate index pairs
#     duplicates = []

#     # Identify duplicates
#     for i, label in enumerate(labels):
#         if label in seen:
#             duplicates.append((seen[label], i))  # Store both indices of duplicate
#         else:
#             seen[label] = i  # Store first occurrence index

#     # Randomly select one occurrence to remove for each duplicate pair
#     indices_to_remove = set(choice(pair) for pair in duplicates)

#     # Filter out the selected indices
#     filtered_labels = [lbl for i, lbl in enumerate(labels) if i not in indices_to_remove]
#     filtered_preds = [pred for i, pred in enumerate(preds) if i not in indices_to_remove]

#     return filtered_preds, filtered_labels


# def average_label_duplicates(preds, labels):
#     """
#     Merges duplicate entries in `labels` by averaging their corresponding `preds` values.

#     Parameters:
#     - labels (list of tuples): List containing (feature, state) pairs.
#     - preds (list of numpy arrays): Corresponding prediction arrays.

#     Returns:
#     - merged_labels (list of tuples): Labels with duplicates merged.
#     - merged_preds (list of numpy arrays): Predictions averaged for duplicate entries.
#     """

#     label_dict = defaultdict(list)  # Dictionary to store predictions for each unique label

#     # Group predictions by label
#     for label, pred in zip(labels, preds):
#         label_dict[label].append(pred)

#     # Compute the average prediction for each unique label
#     merged_labels = list(label_dict.keys())
#     merged_preds = [np.mean(np.stack(pred_list), axis=0) for pred_list in label_dict.values()]
#     return merged_preds, merged_labels


# This one or the next? zscore or just percentile? 
def get_significant_reactivations(all_probs, threshold=2):
    """ get the significant reactivations
    for each property, and for each presence/absence
    threshold is applied to z-scored reactivations
    Maybe keep the actual value, to differentiate 
    "strong" from "weak" reactiations? 
    Use a scaler fit on the whole data, not just a trial?
    """
    significant_reactivations = [(zscore(arr) > threshold) for arr in all_probs]
    return significant_reactivations
    
# def get_significant_reactivations(preds, percentile):
#     """ get the significant reactivations
#     for each property, and for each presence/absence
#     """
#     significant_reactivations = {}
#     for prop in Properties:
#         for presence in ['present', 'absent']:
#             preds_prop = preds[f"{prop}_{presence}"]
#             threshold = np.percentile(preds_prop, percentile)
#             significant_reactivations[f"{prop}_{presence}"] = [pred for pred in preds_prop if pred > threshold]
#     return significant_reactivations


def get_reactivation_times(signif_react):
    """Extracts reactivation times for each property.
    Args:
        signif_react (list of np.ndarray): List of 1D boolean arrays (n_samples,).
    Returns:
        list of np.ndarray: Each entry is an array of time points where reactivation occurs.
    """
    return [np.where(reac)[0] for reac in signif_react]

                
def get_reactivation_episodes(signif_react):
    """Extracts consecutive reactivations per state, and their duration
    (nb of significant consecutive reactivations for this state)
    For a single trial
    Args:
        signif_react (list of np.ndarray): List of 1D boolean arrays (n_samples,).
    Returns:
        list of tuples: [(start, end, state, duration), ...] for all reactivation episodes.
    """
    consecutive_react = []
    for state, reac in enumerate(signif_react):
        times = np.where(reac)[0]  # Get activation time points
        if len(times) == 0:
            continue  # Skip if no activation

        # Identify consecutive_react by grouping consecutive time points
        start = times[0]
        for i in range(1, len(times)):
            if times[i] != times[i - 1] + 1:  # If not consecutive, end "episode"
                duration = times[i - 1] - start + 1
                consecutive_react.append((start, times[i - 1], state, duration))  # (start, end, state, duration)
                start = times[i]
        duration = times[-1] - start + 1
        consecutive_react.append((start, times[-1], state, duration))  # Store last episode of consecutive reactivations

    return consecutive_react


def get_sequential_reactivations(consecutive_react, iLag):
    """Extracts full chains of sequential reactivations up to the lag, storing gaps between them.
    (gap is the time elapsed since the reactivation of the previous item in the episode). Gap for first state is always None.
    Args:
        consecutive_react (list of tuples): [(start, end, state, duration), ...] for all episodes.
        iLag (int): Maximum lag for sequential transitions.
    Returns:
        episodes (list of lists): [[(state1, duration1, gap1), (state2, duration2, gap2), ...], ...]
    """
    episodes = []
    
    # Sort consecutive reactivations by start time
    consecutive_react.sort()  

    visited = set()

    for i, (start, end, state, duration) in enumerate(consecutive_react):
        if i in visited:
            continue  

        chain = [(state, duration, None)]  # First state has no gap
        visited.add(i)

        next_start = end  

        for j in range(i + 1, len(consecutive_react)):
            s2_start, s2_end, s2_state, s2_duration = consecutive_react[j]

            gap = s2_start - next_start  # Compute the gap

            # Ensure valid sequential transition within lag
            if 0 < gap <= iLag:
                chain.append((s2_state, s2_duration, gap))
                visited.add(j)
                next_start = s2_end  # Update search window
            else:
                break  
        
        # Only keep episodes with more than one state
        if len(chain) > 1:
            episodes.append(chain)

    return episodes


def get_synchronous_reactivations_pairs(consecutive_react):
    """Extracts synchronous reactivations where states overlap in time for any number of states.
    DETECTS ONLY PAIRS OF REACTIVATIONS
    Args:
        consecutive_react (list of tuples): [(start, end, state, duration), ...] for all episodes.
    Returns:
        synchronous_episodes (list of tuples): [(state1, state2, overlap_start, overlap_duration), ...]
    """
    synchronous_episodes = []

    # Sort by start time
    consecutive_react.sort()

    for i in range(len(consecutive_react)):
        t1_start, t1_end, s1, d1 = consecutive_react[i]

        for j in range(i + 1, len(consecutive_react)):
            t2_start, t2_end, s2, d2 = consecutive_react[j]

            # Check if they overlap
            if t1_end >= t2_start:  # Overlap condition
                overlap_start = t2_start
                overlap_end = min(t1_end, t2_end)  # The end of the overlap
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > 0:  # Ensure valid overlap
                    synchronous_episodes.append((s1, s2, overlap_start, overlap_duration))

    return synchronous_episodes


def get_synchronous_reactivations(consecutive_react, tolerance=-1):
    """Extracts synchronous reactivations where multiple states overlap in time, with a tolerance.

    Args:
        consecutive_react (list of tuples): [(start, end, state, duration), ...] for all episodes.
        tolerance (int): Time window within which a state can be included in an episode.
            tol = -1 -> strict overlap is necessary
            tol = 0 -> can be consecutive with no overlap
            tol > 0 -> overlap condition is relaxed, states can activate within `tol` timepoints of each other.

    Returns:
        list of tuples: [(states, overlap_start, overlap_duration)]
            where `states` is a set of overlapping states (only if >1 state).
    """
    synchronous_episodes = []
    
    # Sort events by start time
    consecutive_react.sort()

    active_states = []  # Keeps track of active states at any given time

    for start, end, state, _ in consecutive_react:
        # Remove states that ended before the valid window
        if tolerance == -1:
            active_states = [(s, s_end) for s, s_end in active_states if s_end > start]  # Strict overlap needed
        else:
            active_states = [(s, s_end) for s, s_end in active_states if s_end >= start - tolerance]  # Relaxed window

        # Add current state to active states
        active_states.append((state, end))

        # Calculate actual overlap window (strict for tol=-1, relaxed otherwise)
        overlap_start = start
        overlap_end = min(e for _, e in active_states)  # Find the earliest end time
        overlap_duration = overlap_end - overlap_start

        # Gather all currently active states
        active_set = {s for s, _ in active_states}

        merged = False
        for i in range(len(synchronous_episodes)):
            prev_states, prev_start, prev_duration = synchronous_episodes[i]
            prev_end = prev_start + prev_duration

            if tolerance == -1:
                # STRICT overlap: Only merge if the current start is before the previous end
                if overlap_start < prev_end:
                    prev_states.update(active_set)
                    new_end = min(prev_end, overlap_end)  # Ensure we don't extend duration incorrectly
                    synchronous_episodes[i] = (prev_states, prev_start, new_end - prev_start)
                    merged = True
                    break
            else:
                # Relaxed overlap condition
                if overlap_start <= prev_start + tolerance:
                    prev_states.update(active_set)
                    new_end = max(prev_end, overlap_end)
                    synchronous_episodes[i] = (prev_states, prev_start, new_end - prev_start)
                    merged = True
                    break

        if not merged:
            # Create a new synchronous episode
            synchronous_episodes.append((active_set, overlap_start, overlap_duration))

    # Remove episodes with only one state
    synchronous_episodes = [ep for ep in synchronous_episodes if len(ep[0]) > 1]

    return synchronous_episodes


def count_coactivations_pairs(synchronous_episodes, state1, state2):
    """Counts the number of times state1 and state2 activate together,
       and calculates the average overlap duration.
    
    Args:
        synchronous_episodes (list): List of episodes, each containing (states, overlap_start, overlap_duration).
        state1, state2: States to check.

    Returns:
        tuple: (count, average_overlap)
    """
    total_overlap = 0
    count = 0

    for episode in synchronous_episodes:
        states, _, overlap_duration = episode
        if {state1, state2}.issubset(states):  # Check if both states are present
            total_overlap += overlap_duration
            count += 1

    average_overlap = total_overlap / count if count > 0 else 0
    return count, average_overlap


def count_synchronous_coactivations(all_synchronous_episodes, state_set):
    """Counts occurrences where at least part of the state set is activated together.
    Args:
        all_synchronous_episodes (list): List of sets representing synchronous activations.
        state_set (set): The group of states to analyze.
    Returns:
        dict: Keys are the subset sizes (1-5), values are occurrence counts.
    """
    from ipdb import set_trace; set_trace()
    coactivation_counts = {i: 0 for i in range(1, len(state_set) + 1)}
    coactivation_overlaps = {i: [] for i in range(1, len(state_set) + 1)}
    coactivation_matrix = {state: {other_state: 0. for other_state in state_set} for state in state_set}

    for episode in all_synchronous_episodes:
        activated_states, _, overlap = episode
        nb_coactive_states = len(activated_states.intersection(state_set))

        if nb_coactive_states > 0:
            coactivation_counts[nb_coactive_states] += 1
            coactivation_overlaps[nb_coactive_states].append(overlap)
            
            for state in activated_states.intersection(state_set):
                for other_state in activated_states.intersection(state_set):
                    if state != other_state:
                        coactivation_matrix[state][other_state] += 1
    
    # Remove empty lists to avoid warnings
    coactivation_overlaps = {k: v for k, v in coactivation_overlaps.items() if v}
    
    ave_coactivation_overlaps = {k: np.mean(v) for k, v in coactivation_overlaps.items()}
    
    return coactivation_counts, ave_coactivation_overlaps, coactivation_matrix


def count_sequential_coactivations(sequential_episodes, state_set):
    """
    Counts occurrences where at least part of the state set appears in a sequence.

    Args:
        sequential_episodes (list of lists): Each episode is a list of (state, duration, gap).
        state_set (set): The group of states to analyze.

    Returns:
        tuple:
            - coactivation_counts (dict): Keys are the subset sizes (1 to max sequence length), values are occurrence counts.
            - ave_durations (dict): Average duration for each subset size.
            - ave_gaps (dict): Average gap between consecutive coactivations.
            - coactivation_matrix (dict of dicts): Counts of direct transitions between states in state_set.
    """
    coactivation_counts = defaultdict(int)
    coactivation_durations = defaultdict(list)
    coactivation_gaps = defaultdict(list)
    
    # Transition matrix: counts how often state A transitions to state B
    coactivation_matrix = {state: defaultdict(float) for state in state_set}

    for episode in sequential_episodes:
        # Extract states that are in the state_set
        filtered_episode = [(state, duration, gap) for state, duration, gap in episode if state in state_set]
        nb_coactive_states = len(filtered_episode)

        if nb_coactive_states > 0:
            coactivation_counts[nb_coactive_states] += 1
            durations, gaps = zip(*[(d, g) for _, d, g in filtered_episode])

            coactivation_durations[nb_coactive_states].extend(durations)
            coactivation_gaps[nb_coactive_states].extend(gaps)

            # Populate transition matrix
            for (state1, _, _), (state2, _, _) in zip(filtered_episode[:-1], filtered_episode[1:]):
                coactivation_matrix[state1][state2] += 1

    # Compute averages (avoiding empty lists)
    ave_durations = {k: np.mean(v) for k, v in coactivation_durations.items() if v}
    # ave_gaps = {k: np.mean(v) for k, v in coactivation_gaps.items() if v}
    # remove None values
    ave_gaps = {k: np.mean([v for v in vals if v is not None]) for k, vals in coactivation_gaps.items() if vals}

    return coactivation_counts, ave_durations, ave_gaps, coactivation_matrix


def average_coactivation_matrices(coactivation_matrices):
    """Averages coactivation matrices across multiple subjects."""
    states = list(coactivation_matrices[0].keys())
    avg_matrix = {state: {other_state: 0 for other_state in states} for state in states}
    num_subjects = len(coactivation_matrices)
    
    for matrix in coactivation_matrices:
        for state in states:
            for other_state in states:
                avg_matrix[state][other_state] += matrix[state][other_state] / num_subjects
    
    return avg_matrix


def plot_coactivation_heatmap(coactivation_matrix, labels, out_fn):
    """Plots a heatmap of state pair coactivations."""
    states = list(coactivation_matrix.keys())
    if len(labels) != len(states):
        print(f"Number of labels ({len(labels)}) must match number of states ({len(states)}).")
        from ipdb import set_trace; set_trace()
    # assert len(labels) == len(states), "Number of labels must match number of states."
    matrix = np.array([[coactivation_matrix[s1][s2] for s2 in states] for s1 in states])

    # Normalize by total number of reactivations
    total_reactivations = np.sum(matrix)
    if total_reactivations > 0: matrix /= total_reactivations
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="viridis") #, annot=True)
    # plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels at 45 degrees
    # plt.yticks(rotation=45, va='top')    # Rotate y-axis labels at 45 degrees
    # plt.xlabel("State")
    # plt.ylabel("State")
    plt.yticks(rotation=0, va='top')    # Rotate y-axis labels at 45 degrees
    
    ax.invert_yaxis() # Reverse y-axis

    # Draw a red line to separate present/absent conditions
    ax.axvline(len(labels) // 2, color='red', linewidth=2)  # Vertical line
    ax.axhline(len(labels) // 2, color='red', linewidth=2)  # Horizontal line
    # plt.title("Normalized Averaged State Pair Coactivation Heatmap")
    plt.tight_layout()
    plt.savefig(out_fn, dpi=600)
    plt.close()


def compute_np_significance(all_sync_react_this_subject, NP1, NP2, all_states):
    """ Computes how NP1 and NP2 coactivation compare to random pairs.
    
    Args:
        all_sync_react_this_subject: List of synchronous coactivations per trial.
        NP1, NP2: Tuples defining the two NPs to test.
        all_states: List or set of all possible states.

    Returns:
        summary_metrics: Dictionary with coactivation stats and significance.
    """
    # Get NP coactivations
    NP1_counts, NP1_overlap = count_coactivations_pairs(all_sync_react_this_subject, *NP1)
    NP2_counts, NP2_overlap = count_coactivations_pairs(all_sync_react_this_subject, *NP2)

    # Compute coactivations for all possible state pairs
    all_pairs = list(itertools.combinations(all_states, 2))
    # remove the NPs for random pairs
    all_pairs.remove(NP1) 
    all_pairs.remove(NP2)
    all_counts = []
    all_overlaps = []

    for pair in all_pairs:
        count, overlap = count_coactivations_pairs(all_sync_react_this_subject, *pair)
        all_counts.append(count)
        all_overlaps.append(overlap)

    # Convert to numpy for stats
    all_counts = np.array(all_counts)
    all_overlaps = np.array(all_overlaps)

    # Compute percentile ranks
    NP1_percentile = (np.sum(all_counts <= NP1_counts) / len(all_counts)) * 100
    NP2_percentile = (np.sum(all_counts <= NP2_counts) / len(all_counts)) * 100

    # Compute Z-scores
    mean_counts = np.mean(all_counts)
    std_counts = np.std(all_counts)
    NP1_z = (NP1_counts - mean_counts) / std_counts
    NP2_z = (NP2_counts - mean_counts) / std_counts

    # Overlap comparison
    mean_overlap = np.mean(all_overlaps)
    std_overlap = np.std(all_overlaps)
    NP1_overlap_z = (NP1_overlap - mean_overlap) / std_overlap
    NP2_overlap_z = (NP2_overlap - mean_overlap) / std_overlap

    # Store results
    summary_metrics = {
        "NP1_count": NP1_counts,
        "NP2_count": NP2_counts,
        "NP1_percentile": NP1_percentile,
        "NP2_percentile": NP2_percentile,
        "NP1_z_score": NP1_z,
        "NP2_z_score": NP2_z,
        "NP1_overlap": NP1_overlap,
        "NP2_overlap": NP2_overlap,
        "NP1_overlap_z": NP1_overlap_z,
        "NP2_overlap_z": NP2_overlap_z,
    }

    # print(f"X most co-activated pairs of states:")
    # for i in range(10):
    #     max_idx = np.argmax(all_counts)
    #     print(all_pairs[max_idx], all_counts[max_idx], all_overlaps[max_idx])
    #     all_counts[max_idx] = 0

    return summary_metrics


# def plot_average_preds_seaborn(all_present, all_absent, out_fn, labels=["S1", "C1", "R", "S2", "C2"]):
#     """
#     Bar plot of average predictions during the delay using Seaborn and statannotations.
#     """
#     # Prepare data for seaborn
#     data = []
#     for i, label in enumerate(labels):
#         for val in all_present[i]:
#             data.append([label, val, 'Present'])
#         for val in all_absent[i]:
#             data.append([label, val, 'Absent'])
    
#     df = pd.DataFrame(data, columns=['Category', 'Prediction', 'Condition'])
    
#     # Create plot
#     plt.figure(figsize=(10, 6))
#     ax = sns.barplot(data=df, x='Category', y='Prediction', hue='Condition', errorbar=('se', 1),
#                      palette={'Present': 'skyblue', 'Absent': 'orange'})
    
#     # Perform statistical tests and annotate
#     # pairs = [(label, label) for label in labels]
#     pairs = [((label, 'Present'), (label, 'Absent')) for label in labels]
#     annotator = Annotator(ax, pairs, data=df, x='Category', y='Prediction', hue='Condition')
#     annotator.configure(test='t-test_ind', text_format='star', loc='outside', verbose=1)
#     annotator.apply_and_annotate()
    
#     # Labels and legend
#     plt.ylabel('Average Predictions')
#     plt.legend(title='Condition')
#     plt.tight_layout()
    
#     # Save and close
#     plt.savefig(out_fn, dpi=400)
#     plt.close()

def plot_average_preds_seaborn(all_present, all_absent, labels, out_fn):
    """
    Bar plot of average predictions during the delay using Seaborn and statannotations.
    """
    # Prepare data for seaborn
    data = []
    for i, label in enumerate(labels):
        for val in all_present[i]:
            data.append([label, val, 'Present'])
        for val in all_absent[i]:
            data.append([label, val, 'Absent'])
    
    df = pd.DataFrame(data, columns=['Category', 'Prediction', 'Condition'])
    
    # Create plot
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(data=df, x='Category', y='Prediction', hue='Condition', errorbar=('se', 1),
                     palette={'Present': 'skyblue', 'Absent': 'orange'})
    
    # Perform statistical tests and annotate
    # pairs = [(label, label) for label in labels]
    pairs = [((label, 'Present'), (label, 'Absent')) for label in labels]
    annotator = Annotator(ax, pairs, data=df, x='Category', y='Prediction', hue='Condition')
    annotator.configure(test='t-test_ind', text_format='star', loc='outside', verbose=1)
    annotator.apply_and_annotate()
    
    # Labels and legend
    plt.ylabel('Average Predictions')
    plt.xlabel('') # remove xlabel
    plt.legend(title='Condition')
    plt.tight_layout()
    
    # Save and close
    plt.savefig(out_fn, dpi=400)
    plt.close()


def does_reactivations_predict_behavioral(df, out_fn, y="Performance", mixed='sub', scatter=True):
    """
    Regression plot of reactivations vs. behavioral performance.
    mixed: kind of random effect. 'sub' or 'sub+prop'
    Mixed-effects model (accounts for repeated measures within subjects)
    (for present items)
    """        
    # warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", ".*MLE*.")
    warnings.filterwarnings("ignore", ".*Hessian*.")

    if mixed == 'sub+prop': # hierarchical random effect of subject and property
        model = smf.mixedlm(f"{y} ~ Reactivation", df, groups=df["Subject"], re_formula="1 + Property").fit()
        # crossed random effects of subject and property
        # vc = {"Property": "0 + C(Property)"}  # Define Property as a random effect
        # model = smf.mixedlm(f"{y} ~ Reactivation", df, groups=df["Subject"], vc_formula=vc).fit()
    else:
    # random effect of subjects
        try:
            model = smf.mixedlm(f"{y} ~ Reactivation", df, groups=df["Subject"]).fit()
        except:
            print(f"Error in mixedlm for out_fn: {out_fn}")
            return

    # Scatter plot with subject-level data
    plt.figure(figsize=(8, 6))
    if scatter: 
        sns.scatterplot(data=df, x="Reactivation", y=y, hue="Subject", palette="tab10", alpha=0.6)

    # Add a regression line for the overall effect
    sns.regplot(data=df, x="Reactivation", y=y, scatter=False, color="black")

    plt.xlabel("Reactivation Strength")
    plt.ylabel("Behavioral Performance")
    # plt.title("Mixed-Effects Model: Reactivation vs. Performance")
    plt.title(f"tvalue: {model.tvalues["Reactivation"]:.3f} - pvalue: {model.pvalues["Reactivation"]:.3f}")
    ax = plt.gca()
    if scatter: ax.get_legend().remove()
    # plt.legend(title="Subject", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_fn, dpi=400)
    plt.close()



#### TDLM-style #####

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


def sequenceness_Crosscorr(rd, T, lag=1):
    """
    Compute sequenceness cross-correlation based on a transition matrix

    Parameters:
    - rd: np.ndarray
        Array with shape (n_samples, n_states), representing samples by states.
    - T: np.ndarray
        Transition matrix of interest with shape (n_states, n_states).
    - lag: int
        Number of samples by which the data should be shifted.

    Returns:
    - sf: float
        The computed sequenceness factor (cross-correlation).
    """
    n_samples, n_states = rd.shape

    orig = rd[:n_samples - 2 * lag] @ T
    proj = rd[lag:n_samples - lag]

    # Scale variance and compute cross-correlation
    corr_temp = np.full(n_states, np.nan)
    for i in range(n_states):
        if np.nansum(orig[:, i]) != 0 and np.nansum(proj[:, i]) != 0:
            corr_temp[i] = np.corrcoef(orig[:, i], proj[:, i])[0, 1]
    
    sf = np.nanmean(corr_temp)

    return sf


# def empirical_TRM(time_series, lag):
#     """
#     Compute the empirical transition matrix for stimulus activations.
#     DEPRECIATED, use the vectorized all_trials version.

#     Parameters:
#     - time_series: np.ndarray
#         A 2D array of shape (n_samples, n_stimuli), where each column represents 
#         the reactivation time series for a stimulus.
#     - lag: int
#         The time lag (Δt) for which the transition matrix is computed.

#     Returns:
#     - transition_matrix: np.ndarray
#         A 6x6 matrix of regression coefficients describing transitions between stimuli.
#     """
#     n_samples, n_stimuli = time_series.shape

#     # Initialize the transition matrix
#     transition_matrix = np.zeros((n_stimuli, n_stimuli))

#     # Iterate over each stimulus (as the target)
#     for target_stimulus in range(n_stimuli):
#         # Define the target variable (Y_i)
#         target = time_series[lag:, target_stimulus]  # Target variable starts from lag

#         # Define the predictors (lagged time series of all stimuli)
#         predictors = np.zeros((n_samples - lag, n_stimuli))
#         for stimulus in range(n_stimuli):
#             predictors[:, stimulus] = time_series[:n_samples - lag, stimulus]

#         # Fit the linear model
#         model = LinearRegression(fit_intercept=True)
#         model.fit(predictors, target)

#         # Store the coefficients in the transition matrix
#         transition_matrix[target_stimulus, :] = model.coef_

#     return transition_matrix


# def compute_TRM_all_trials(data, lag):
#     """
#     Compute empirical transition matrix across multiple trials.
#     DEPRECIATED, use the vectorized all_trials version.
    
#     Parameters:
#     - data: list of np.ndarray
#         A list where each element is an array of shape (n_times, n_states), representing 
#         time series data for each trial.
#     - lag: int
#         Time lag for the transition matrix.
    
#     Returns:
#     - transition_matrix: np.ndarray
#         The estimated transition matrix of shape (n_states, n_states).
#     """
#     import numpy as np
#     from sklearn.linear_model import LinearRegression

#     n_states = data[0].shape[1]
#     n_trials = len(data)

#     # Prepare design matrix and target vector
#     X = []
#     Y = []

#     for trial in data:
#         n_times = trial.shape[0]

#         # Generate lagged predictors and targets for this trial
#         if n_times > lag:
#             for t in range(lag, n_times):
#                 X.append(trial[t - lag])
#                 Y.append(trial[t])

#     # Convert to arrays
#     X = np.vstack(X)  # Shape: (total_samples, n_states)
#     Y = np.vstack(Y)  # Shape: (total_samples, n_states)

#     # Fit a separate linear model for each stimulus
#     transition_matrix = np.zeros((n_states, n_states))

#     for i in range(n_states):
#         model = LinearRegression(fit_intercept=True)  # No intercept for transition matrix
#         model.fit(X, Y[:, i])  # Predict stimulus i from lagged predictors
#         transition_matrix[:, i] = model.coef_

#     return transition_matrix


def compute_TRM_all_trials_vectorized(data, lag):
    """
    Compute empirical transition matrix for multiple trials at a given time lag.
    Parallelized implementation (single linear regression for all states with the matrix formulation)
    Much faster than the loopy version and numerically very close (but not identical).

    Parameters:
    - data: list of np.ndarray
        A list where each element is a trial matrix of shape (n_samples, n_states).
    - lag: int
        Time lag for the transition matrix.

    Returns:
    - b: np.ndarray
        Empirical transition matrix of shape (n_states, n_states).
    """
    # Concatenate all trials, ensuring no transitions between trials
    X_full = np.vstack(data)  # Combine all trials into one matrix
    trial_lengths = [trial.shape[0] for trial in data]
    
    # Identify rows to exclude due to lag truncation at trial boundaries
    invalid_indices = []
    cumulative_length = 0
    for length in trial_lengths:
        invalid_indices.extend(range(cumulative_length, cumulative_length + lag))  # Start of each trial
        invalid_indices.extend(range(cumulative_length + length - lag, cumulative_length + length))  # End of each trial
        cumulative_length += length

    # Valid indices after excluding invalid rows
    valid_indices = np.setdiff1d(np.arange(X_full.shape[0]), invalid_indices)

    # Create valid X and shifted X_dt
    X_trimmed = X_full[valid_indices, :]
    X_dt_indices = valid_indices + lag  # Shift indices for lag
    X_dt_indices = X_dt_indices[X_dt_indices < X_full.shape[0]]  # Ensure valid range
    X_dt = X_full[X_dt_indices, :]


    # Compute the empirical transition matrix
    b = np.linalg.pinv(X_trimmed.T @ X_trimmed) @ (X_trimmed.T @ X_dt)

    return b


def compute_TRM_single_trial(data, lag):
    """
    Compute empirical transition matrix for a single trial at a given time lag.
    Uses the vectorized implementation for a single trial.
    A Bit slower but more sensitive than the all trials version.
    Best so far. USE THIS ONE.

    Parameters:
    - data: np.ndarray
        Trial matrix of shape (n_samples, n_states).
    - lag: int
        Time lag for the transition matrix.
    Returns:
    - b: np.ndarray
        Empirical transition matrix of shape (n_states, n_states).
    """
    n_samples, n_states = data.shape

    # Exclude invalid rows at the start and end of the trial
    valid_indices = np.arange(lag, n_samples - lag)

    # Create valid X and shifted X_dt
    X_trimmed = data[valid_indices, :]
    X_dt = data[valid_indices + lag, :]

    # Compute the empirical transition matrix
    b = np.linalg.pinv(X_trimmed.T @ X_trimmed) @ (X_trimmed.T @ X_dt)

    return b


def compute_empirical_transition_matrix_matlab_style(X, maxLag):
    """
    Compute empirical transition matrix, matching MATLAB implementation.

    Parameters:
    - X: np.ndarray
        State time courses of shape (n_samples, n_states).
    - maxLag: int
        Maximum lag to consider.

    Returns:
    - betas: np.ndarray
        Empirical transition matrix, shape (n_states * maxLag, n_states).
    """
    n_samples, n_states = X.shape
    nbins = maxLag + 1

    # Construct the design matrix
    dm = np.zeros((n_samples, maxLag * n_states))
    for kk in range(n_states):
        temp = np.zeros((n_samples, nbins))
        for i in range(1, nbins):
            if i < n_samples:
                temp[i:, i] = X[:-i, kk]
        dm[:, kk * maxLag:(kk + 1) * maxLag] = temp[:, 1:]

    # Initialize betas
    betas = np.full((n_states * maxLag, n_states), np.nan)

    # Regression
    for ilag in range(1, maxLag + 1):
        zinds = np.arange(ilag - 1, n_states * maxLag, maxLag)
        predictors = np.hstack([dm[:, zinds], np.ones((n_samples, 1))])
        
        # Solve regression for each state
        for state in range(n_states):
            y = X[:, state]
            coefs = np.linalg.pinv(predictors) @ y
            betas[zinds, state] = coefs[:-1]  # Exclude intercept

    return betas



def second_level_analysis_matlab_style(betas, TF, TR, n_states, maxLag):
    """
    Perform second-level sequence analysis.

    Parameters:
    - betas: np.ndarray
        First-level regression results, shape (n_states * maxLag, n_states).
    - TR: np.ndarray
        Transition matrix for the main sequence, shape (n_states, n_states).
    - TR: np.ndarray
        Transition matrix for the alternative sequence, shape (n_states, n_states).
    - n_states: int
        Number of states.
    - maxLag: int
        Maximum lag.

    Returns:
    - results: dict
        Dictionary containing z-scored regression coefficients for each component.
    """
    # Reshape betas to (maxLag, n_states^2)
    betasnbins64 = betas.reshape(maxLag, n_states**2).T

    TF_flat = TF.flatten() # Flatten matrices
    TR_flat = TR.flatten()
    identity_flat = np.eye(n_states).flatten()
    constant_flat = np.ones((n_states, n_states)).flatten()

    # Design matrix for regression
    design_matrix = np.column_stack([TF_flat, TR_flat, identity_flat, constant_flat])

    bbb = np.linalg.pinv(design_matrix) @ betasnbins64 # Perform regression

    sf = zscore(bbb[0, :], axis=None) # Z-score normalization
    sb = zscore(bbb[1, :], axis=None)
    ss = zscore(bbb[2, :], axis=None)  # Autocorrelation
    sc = zscore(bbb[3, :], axis=None)  # Constant    

    return sf, sb, ss, sc


def second_level_analysis(etm, templates):
    """
    Perform second-level sequence analysis
    = Compute weights for each template matrix using GLM.

    Parameters:
    - etm: np.ndarray
        Empirical transition matrix of shape (n_states, n_states).
    - templates: list of np.ndarray
        List of template matrices (T_r) of shape (n_states, n_states).

    Returns:
    - Z: np.ndarray
        Array of weights (Z_r) corresponding to each template matrix.
    """
    # Reshape templates and etm into vectors for GLM
    etm_vector = etm.flatten()  # Shape: (n_states^2,)
    template_vectors = np.array([T.flatten() for T in templates])  # Shape: (n_templates, n_states^2)

    # Normalize each template vector
    scaler = MinMaxScaler()
    template_vectors = scaler.fit_transform(template_vectors)

    # Solve GLM: etm = T * Z
    Z, _, _, _ = np.linalg.lstsq(template_vectors.T, etm_vector, rcond=None)

    return Z


def inject_sequences(timecourses, sequences, lag=5, num_injections=100, state_amplitude=1, reverse=False):
    """
    Inject sequences into random time courses with a fixed lag between state activations.

    Parameters:
    - timecourses: np.ndarray
        Array of shape (n_states, n_times, n_trials), representing random time courses.
    - sequences: list of lists
        List of sequences to inject, where each sequence is a list of state indices (0-based).
    - lag: int
        Fixed lag between state activations (in timepoints).
    - num_injections: int
        Number of sequence injections per trial.
    - state_amplitude: float
        Amplitude of the injected states.

    Returns:
    - timecourses: np.ndarray
        Time courses with injected sequences.
    """
    n_states, n_times, n_trials = timecourses.shape

    for trial_idx in range(n_trials):
        for _ in range(num_injections):
            # Choose a random start time for the sequence injection
            max_start_time = n_times - len(sequences[0]) * lag
            if max_start_time <= 0:
                raise ValueError("Sequence length with lag exceeds time dimension.")
            start_time = np.random.randint(0, max_start_time)

            # Choose a random sequence to inject
            sequence = sequences[np.random.randint(len(sequences))]
            if reverse: sequence = sequence[::-1]

            # Inject the sequence into the time course
            for step_idx, state_idx in enumerate(sequence):
                time_idx = start_time + step_idx * lag
                timecourses[state_idx - 1, time_idx, trial_idx] += state_amplitude # go from 1-based to 0-based

    return timecourses


def compute_null_distribution(timecourses, templates, lag=1, n_permutations=1000):
    """
    Compute the null distribution of sequenceness values by permuting states for each trial.

    Parameters:
    - timecourses: np.ndarray
        Array of shape (n_states, n_times, n_trials), representing time courses.
    - templates: np.ndarray
        Transition matrix of interest, of shape (n_states, n_states).
    - lag: int
        Fixed lag to compute empirical transition matrix.
    - n_permutations: int
        Number of permutations for the null distribution.

    Returns:
    - null_distribution: np.ndarray
        Array of shape (n_permutations,) containing the null sequenceness values.
    """
    n_states, n_times, n_trials = timecourses.shape
    null_distribution_f = np.zeros(n_permutations) # Initialize the null distribution for forward
    null_distribution_b = np.zeros(n_permutations) # Initialize the null distribution for backward

    for perm_idx in range(n_permutations):
        permuted_timecourses = np.zeros_like(timecourses)

        for trial_idx in range(n_trials): # Permute states for each trial independently
            permuted_order = np.random.permutation(n_states)
            permuted_timecourses[:, :, trial_idx] = timecourses[permuted_order, :, trial_idx]

        # Compute the empirical transition matrix for the permuted data
        empirical_transition_matrix = compute_TRM_all_trials(permuted_timecourses, lag)

        Z = second_level_analysis(empirical_transition_matrix, templates)
        null_distribution_f[perm_idx] = Z[0]
        null_distribution_b[perm_idx] = Z[1] 

    return null_distribution_f, null_distribution_b


def combine_null_distributions_and_test(null_distributions, observed_values):
    """
    Combine null distributions across subjects/episodes/trials into a single statistic for each lag.

    Parameters:
    - null_distributions: np.ndarray
        Null distributions, shape (n_subjects, n_episodes, n_permutations, n_lags).
    - observed_values: np.ndarray
        Observed sequenceness values, shape (n_subjects, n_episodes, n_lags).

    Returns:
    - z_scores: np.ndarray
        Z-scores of observed values compared to null distributions, shape (n_lags,).
    - p_values: np.ndarray
        P-values of observed values compared to null distributions, shape (n_lags,).
    """
    n_subjects, n_episodes, n_permutations, n_lags = null_distributions.shape

    # Reshape and combine null distributions across subjects and episodes
    combined_null = null_distributions.reshape(-1, n_permutations, n_lags)
    combined_null = np.nanmean(combined_null, axis=0)  # Aggregate across subjects/episodes

    # Aggregate observed values across subjects and episodes
    combined_observed = np.nanmean(observed_values, axis=(0, 1))  # Shape (n_lags,)

    # Initialize statistics
    z_scores = np.zeros(n_lags)
    p_values = np.zeros(n_lags)

    for lag in range(n_lags):
        # Compute mean and std for null distribution at this lag
        null_mean = np.mean(combined_null[:, lag])
        null_std = np.std(combined_null[:, lag])

        # Compute z-score
        z_scores[lag] = (combined_observed[lag] - null_mean) / null_std
        # Compute p-value (one-tailed test)
        p_values[lag] = np.mean(combined_null[:, lag] >= combined_observed[lag])

    return z_scores, p_values










# def get_preds_and_sequenceness_for_cond(df, preds, train_cond, gen_cond, labels, maxLag=1, n_states=8, do_rel=True):
#     T_auto = np.eye(n_states)  # Autotransitions
#     T_const = np.ones((n_states, n_states))  # Uniform transitions
#     times = np.arange(maxLag)*10
#     subs = df['sub'].unique()
#     n_subs = len(subs)
#     minmaxScaler = MinMaxScaler()

#     df_cond = df.query(f"train_cond == '{train_cond}' and gen_cond == '{gen_cond}'")
#     sf = np.full((n_subs, maxLag), np.nan) # to store the average of all trials for each subject and lag
#     sb, sr = np.copy(sf), np.copy(sf) # also a random matrix, for comparison purpose
#     # preds_present, preds_absent = [], []
#     props = Properties if do_rel else Properties[0:2] + Properties[3:5]
#     ave_preds_all_subs = {f"{prop}_{presence}": [] for presence in ['present', 'absent'] for prop in props}
#     behav_df = {"Subject": [], "Condition": [], "Property": [], "Reactivation": [], "Performance": [], "RT": []}

#     # all_sync_react = [] # list of lists, for each subjects, for each trials, each synchronous reactivation
# #     [[{A, B}, t1, duration1], 
# #     [{C, D}, t2, duration2], 
# #     [{A, B, C, D, E}, t3, duration3], 
# #     ...]

#     for iLag in range(maxLag): # for each lag
#         if iLag > 0: continue # quick fix for just looking at the predictions, no replay

#         for iSub, sub in tqdm(enumerate(subs)):
#             df_sub = df_cond.query(f"sub=={sub}")
#             trial_ids = df_sub.trial_id.unique()
#             n_trials = len(trial_ids)

#             # all_sync_react.append([]) # for this subject
#             all_sync_react_this_subject = []
#             all_subjects_summary = []

#             sf_all_trials, sb_all_trials, sr_all_trials = [], [], []
#             # preds_present_all_trials, preds_absent_all_trials = [], []
#             preds_sub_by_presence = {f"{prop}_{presence}": [] for presence in ['present', 'absent'] for prop in Properties} # for this subject and lag
#             preds_sub_by_trial = [{f"{prop}_{presence}": [] for presence in ['present', 'absent'] for prop in Properties} for i in range(n_trials)] # for this subject and lag, each trial separately
#             perfs = []
#             RTs = []
#             for iTrial in range(n_trials):
#                 df_trial = df_sub.query(f"trial_id=='{trial_ids[iTrial]}'")
#                 ## Useless now that we have a single decoder for all properties (actually 2, Prop and PropAll)
#                 # if len(df_trial) == 10:
#                 #     print("\n weird, we get duplicate of each entry. Keeping only one of each line.")
#                 #     df_trial = df_trial.drop_duplicates(subset="train_time")
#                 # assert df_trial[Properties].nunique().sum() <= 5, f"More than five properties identified for trial {iTrial}: {trial_ids[iTrial]}"
#                 # if len(df_trial) != 5: from ipdb import set_trace; set_trace()
#                 # assert len(df_trial) == 5, f"Found more than the 5 entries for trial {iTrial}: {trial_ids[iTrial]}"
                
#                 s1, c1, rel, s2, c2 = df_trial.iloc[0][Properties].values
#                 # print(s1, c1, rel, s2, c2)
#                 # TF = get_TF_5words(s1, c1, rel, s2, c2)
#                 # TR = TF.T
#                 # rand_inds = np.random.permutation(n_states)
#                 # Trand = TF[rand_inds]
#                 # templates = [TF, TR, Trand, T_auto, T_const]

#                 if iLag == 0: # save preds of present vs absent words for barplot of average predictions

#                     preds_props = get_trial_preds_from_data(df_trial, all_preds_data)
#                     # update the dict of preds_sub_by_presence
#                     preds_sub_by_presence = add_present_or_absent_preds_one_trial(preds_props, [s1, c1, rel, s2, c2], preds_sub_by_presence, do_rel=do_rel)
                    
#                     perf = df_trial["Perf"].unique()
#                     assert len(perf) == 1, f"More than one performance value for trial {iTrial}: {trial_ids[iTrial]}" # useless if we test the uniqueness of each trial above
#                     perfs.append(perf[0])

#                     RT = df_trial["RT"].unique()
#                     assert len(RT) == 1, f"More than one RT value for trial {iTrial}: {trial_ids[iTrial]}"
#                     RTs.append(RT[0])

#                     # get predictions over the whole window, for each property, depending on whether it is present or absent
#                     # present: list of 5 arrays of shape (n_samples, n_states) for present
#                     # absent: list of lists of one or two arrays of shape (n_samples, n_states) for absent (depending on how many absent properties)
#                     present, absent = get_present_or_absent_preds_one_trial(preds_props, [s1, c1, rel, s2, c2], do_rel=do_rel)
#                     ave_present = [p.mean() for p in present]
#                     ave_absent = [np.mean([a.mean() for a in sublist]) for sublist in absent]
#                     Props = Properties if do_rel else Properties[0:2] + Properties[3:5]
#                     for pres_prop, preds in zip(Props, ave_present):
#                         behav_df["Subject"].append(sub)
#                         behav_df["Condition"].append("Present")
#                         behav_df["Property"].append(pres_prop)
#                         behav_df["Reactivation"].append(preds.mean())
#                         behav_df["Performance"].append(perf[0])
#                         behav_df["RT"].append(RT[0])
#                     for abs_prop, preds in zip(Props, ave_absent):
#                         behav_df["Subject"].append(sub)
#                         behav_df["Condition"].append("Absent")
#                         behav_df["Property"].append(abs_prop)
#                         behav_df["Reactivation"].append(np.mean(preds))
#                         behav_df["Performance"].append(perf[0])
#                         behav_df["RT"].append(RT[0])
#                     for i, prop in enumerate(Props):
#                         behav_df["Subject"].append(sub)
#                         behav_df["Condition"].append("Difference")
#                         behav_df["Property"].append(prop)
#                         behav_df["Reactivation"].append(np.mean(ave_present[i]) - np.mean(ave_absent[i]))
#                         behav_df["Performance"].append(perf[0])
#                         behav_df["RT"].append(RT[0])


#                     # get Sophie-style reactivations
#                     preds_this_trial, labels_this_trial = restructure_data(present, absent)
#                     # signif_react_present, signif_react_absent = get_significant_reactivations(present, absent)
#                     signif_react = get_significant_reactivations(preds_this_trial)
#                     reac_times = get_reactivation_times(signif_react)
#                     consecutive_react = get_reactivation_episodes(signif_react)

#                     if iLag == 0:
#                         sequential_episodes = get_sequential_reactivations(consecutive_react, iLag)
#                         # synchronous_episodes_pairs = get_synchronous_reactivations_pairs(consecutive_react)
#                         synchronous_episodes = get_synchronous_reactivations(consecutive_react, tolerance=-1) 
#                         all_sync_react_this_subject.extend(synchronous_episodes)

#                 # # trial_preds = np.concatenate([preds_shape, preds_color, preds_rel], axis=1)
#                 # trial_preds = np.concatenate(preds_props, axis=1)
#                 # trm = compute_TRM_single_trial(np.array(trial_preds), iLag)
#                 # trm = minmaxScaler.fit_transform(trm) # a priori no used in wimmer
#                 # from ipdb import set_trace; set_trace()
#                 # # 8 states but 14 reactivations ...
#                 # Z = second_level_analysis(trm, templates)

#                 # sf_all_trials.append(Z[0])
#                 # sb_all_trials.append(Z[1])
#                 # sr_all_trials.append(Z[2])

#             ## For this subject, get the average of the predictions
#             if iLag == 0:
#                 if len(preds_sub_by_presence["Shape1_present"]) == 0:
#                     from ipdb import set_trace; set_trace()
#                 # update the dict of averages
#                 ave_preds_all_subs = get_subj_ave_preds(preds_sub_by_presence, ave_preds_all_subs, props=props) 

#                 # synchronous coactivations
#                 NP1_counts, NP1_overlap = count_coactivations_pairs(all_sync_react_this_subject, 0, 1)
#                 NP2_counts, NP2_overlap = count_coactivations_pairs(all_sync_react_this_subject, 3, 4)

#                 # print(f"NP1 states co-activated {NP1_counts} times; average overlap: {NP1_overlap}")
#                 # print(f"NP2 states co-activated {NP2_counts} times; average overlap: {NP2_overlap}")

#                 first_five_states = {0, 1, 2, 3, 4}
#                 # subset_counts, ave_overlap_size = count_coactivations(all_sync_react_this_subject, first_five_states)
#                 # print("Coactivation frequencies for the first 5 states:")
#                 # for subset_size, count in subset_counts.items():
#                 #     print(f"{subset_size} states together: {count} times; average overlap: {ave_overlap_size[subset_size]}")

#                 subset_counts, ave_overlap_size = count_coactivations(all_sync_react_this_subject, first_five_states)
#                 # print("Coactivation frequencies for the first 5 states:")
#                 # for subset_size, count in subset_counts.items():
#                 #     print(f"{subset_size} states (tol=-1 = strict overlap)together: {count} times; average overlap: {ave_overlap_size[subset_size]}")

#                 all_states = set(range(11))  # Adjust based on your data
#                 NP1 = (0, 1)
#                 NP2 = (3, 4)
#                 summary = compute_np_significance(all_sync_react_this_subject, NP1, NP2, all_states)
#                 all_subjects_summary.append(summary)
#                 # print("\nSummary Statistics:")
#                 # for key, value in summary.items():
#                 #     print(f"{key}: {value:.3f}")
            

#         #     # mean over trials for this subject, lag and condition
#         #     sf[iSub, iLag] = np.nanmean(np.array(sf_all_trials), axis=0)
#         #     sb[iSub, iLag] = np.nanmean(np.array(sb_all_trials), axis=0)
#         #     srand[iSub, iLag] = np.nanmean(np.array(sr_all_trials), axis=0)

#         # sf[iSub] -= np.nanmean(sf[iSub]) # mean correct
#         # sb[iSub] -= np.nanmean(sb[iSub]) # mean correct
#         # srand[iSub] -= np.nanmean(srand[iSub]) # mean correct
#     coactivation_df = pd.DataFrame(all_subjects_summary)
#     return ave_preds_all_subs, behav_df, coactivation_df, sf, sb, sr





