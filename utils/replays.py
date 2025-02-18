import mne
import os.path as op
import os
from glob import glob
import pandas as pd
import numpy as np
import pickle
from scipy.stats import sem, pearsonr, zscore, ttest_ind
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.linear_model import LinearRegression
# from scipy.linalg import toeplitz
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import statsmodels.formula.api as smf
import itertools


from .params import *

def get_trial_preds_from_data(df_trial, all_preds_data):
    """ get the predictions for a given trial 
    df_trial should have a single entry. 
    """
    assert len(df_trial) == 1, f"len(df_trial)={len(df_trial)}"
    preds_idx = df_trial.index.values
    preds = all_preds_data[preds_idx[0]]
    return preds


# def get_trial_preds_from_data_OLD(df_trial, all_preds_data, labels):
#     """ get the predictions for a given trial 
#     df_trial might have more than 5 entries, but we filter according to the list of labels
#     OLD VERSION, FOR BEFORE FITTING A SINGLE DECODER FOR ALL PROPERTIES
#     """
#     preds_shape1_idx = df_trial.query(f"label=='{labels[0]}'").index.values
#     assert len(preds_shape1_idx) == 1, f"len(preds_shape1_idx)={len(preds_shape1_idx)} for {labels[0]}"
#     preds_shape1 = all_preds_data[preds_shape1_idx[0]]

#     preds_color1_idx = df_trial.query(f"label=='{labels[1]}'").index.values
#     assert len(preds_color1_idx) == 1, f"len(preds_color1_idx)={len(preds_color1_idx)} for {labels[1]}"
#     preds_color1 = all_preds_data[preds_color1_idx[0]]

#     preds_rel_idx = df_trial.query(f"label=='{labels[2]}'").index.values
#     assert len(preds_rel_idx) == 1, f"len(preds_rel_idx)={len(preds_rel_idx)} for {labels[2]}"
#     preds_rel = all_preds_data[preds_rel_idx[0]]

#     preds_shape2_idx = df_trial.query(f"label=='{labels[3]}'").index.values
#     assert len(preds_shape2_idx) == 1, f"len(preds_shape2_idx)={len(preds_shape2_idx)} for {labels[3]}"
#     preds_shape2 = all_preds_data[preds_shape2_idx[0]]

#     preds_color2_idx = df_trial.query(f"label=='{labels[4]}'").index.values
#     assert len(preds_color2_idx) == 1, f"len(preds_color2_idx)={len(preds_color2_idx)} for {labels[4]}"
#     preds_color2 = all_preds_data[preds_color2_idx[0]]

#     return [preds_shape1, preds_color1, preds_rel, preds_shape2, preds_color2]


def add_present_or_absent_preds_one_trial(preds, props, preds_sub_by_presence, do_rel):
    """ take the predictions for a single trial
    and put them in the dict that stores all MEAN preds
    with this subjects, as a function of the property
    and whether it was present or absent in the sentence.
    """ 
    # preds of shape n_samples, n_properties: shape (0:3), then colors (3:6), then Relation (6:8).
    s1, c1, rel, s2, c2 = props
    preds_sub_by_presence["Shape1_present"].append(preds[:, 0:3][:, shapes.index(s1)].mean())
    preds_sub_by_presence["Colour1_present"].append(preds[:, 3:6][:, colors.index(c1)].mean())
    if do_rel:
        preds_sub_by_presence["Relation_present"].append(preds[:, 6::][:, relations.index(rel)].mean())
    preds_sub_by_presence["Shape2_present"].append(preds[:, 0:3][:, shapes.index(s2)].mean())
    preds_sub_by_presence["Colour2_present"].append(preds[:, 3:6][:, colors.index(c2)].mean())

    shapes_absent = [s for s in shapes if s not in [s1, s2]]
    colors_absent = [c for c in colors if c not in [c1, c2]]
    relation_absent = [r for r in relations if r != rel][0]
    for absent_shape in shapes_absent:
        preds_sub_by_presence["Shape1_absent"].append(preds[:, 0:3][:, shapes.index(absent_shape)].mean())
        preds_sub_by_presence["Shape2_absent"].append(preds[:, 0:3][:, shapes.index(absent_shape)].mean())
    for absent_color in colors_absent:
        preds_sub_by_presence["Colour1_absent"].append(preds[:, 3:6][:, colors.index(absent_color)].mean())
        preds_sub_by_presence["Colour2_absent"].append(preds[:, 3:6][:, colors.index(absent_color)].mean())
    if do_rel:
        preds_sub_by_presence['Relation_absent'].append(preds[:, 6::][:, relations.index(relation_absent)].mean())

    return preds_sub_by_presence


# def add_present_or_absent_preds_one_trial_OLD(preds, props, preds_sub_by_presence):
#     """ take the predictions for a single trial
#     and put them in the dict that stores all MEAN preds
#     with this subjects, as a function of the property
#     and whether it was present or absent in the sentence.
#     """ 
#     s1, c1, rel, s2, c2 = props
#     preds_sub_by_presence["Shape1_present"].append(preds[0][:, shapes.index(s1)].mean())
#     preds_sub_by_presence["Colour1_present"].append(preds[1][:, colors.index(c1)].mean())
#     preds_sub_by_presence["Relation_present"].append(preds[2][:, relations.index(rel)].mean())
#     preds_sub_by_presence["Shape2_present"].append(preds[3][:, shapes.index(s2)].mean())
#     preds_sub_by_presence["Colour2_present"].append(preds[4][:, colors.index(c2)].mean())

#     shapes_absent = [s for s in shapes if s not in [s1, s2]]
#     colors_absent = [c for c in colors if c not in [c1, c2]]
#     relation_absent = [r for r in relations if r != rel][0]
#     for absent_shape in shapes_absent:
#         preds_sub_by_presence["Shape1_absent"].append(preds[0][:, shapes.index(absent_shape)].mean())
#         preds_sub_by_presence["Shape2_absent"].append(preds[3][:, shapes.index(absent_shape)].mean())
#     for absent_color in colors_absent:
#         preds_sub_by_presence["Colour1_absent"].append(preds[1][:, colors.index(absent_color)].mean())
#         preds_sub_by_presence["Colour2_absent"].append(preds[4][:, colors.index(absent_color)].mean())
#     preds_sub_by_presence['Relation_absent'].append(preds[2][:, relations.index(relation_absent)].mean())

#     return preds_sub_by_presence


def get_present_or_absent_preds_one_trial(preds, props, do_rel):
    """ take the predictions for a single trial
    and returns them as 2 list of the 5 Properties
    (Present and Absent), not averaged
    """ 
    # preds or shapes (0:3), then colors (3:6), then Relation (6:8).
    s1, c1, rel, s2, c2 = props
    Shape1_present = preds[:, 0:3][:, shapes.index(s1)]
    Colour1_present = preds[:, 3:6][:, colors.index(c1)]
    if do_rel:
        Relation_present = preds[:, 6::][:, relations.index(rel)]
    Shape2_present = preds[:, 0:3][:, shapes.index(s2)]
    Colour2_present = preds[:, 3:6][:, colors.index(c2)]

    shapes_absent = [s for s in shapes if s not in [s1, s2]]
    colors_absent = [c for c in colors if c not in [c1, c2]]
    relation_absent = [r for r in relations if r != rel][0]
    Shape1_absent, Shape2_absent, Colour1_absent, Colour2_absent = [], [], [], []
    for absent_shape in shapes_absent:
        Shape1_absent.append(preds[:, 0:3][:, shapes.index(absent_shape)])
        Shape2_absent.append(preds[:, 0:3][:, shapes.index(absent_shape)])
    for absent_color in colors_absent:
        Colour1_absent.append(preds[:, 3:6][:, colors.index(absent_color)])
        Colour2_absent.append(preds[:, 3:6][:, colors.index(absent_color)])
    if do_rel:
        Relation_absent = [preds[:, 6::][:, relations.index(relation_absent)]] # cast to list, for consistency with other absents
        return [Shape1_present, Colour1_present, Relation_present, Shape2_present, Colour2_present], [Shape1_absent, Colour1_absent, Relation_absent, Shape2_absent, Colour2_absent]
    else:
        return [Shape1_present, Colour1_present, Shape2_present, Colour2_present], [Shape1_absent, Colour1_absent, Shape2_absent, Colour2_absent]


# def get_present_or_absent_preds_one_trial_OLD(preds, props):
#     """ take the predictions for a single trial
#     and returns them as 2 list of the 5 Properties
#     (Present and Absent), not averaged
#     """ 
#     s1, c1, rel, s2, c2 = props
#     Shape1_present = preds[0][:, shapes.index(s1)] #.mean()
#     Colour1_present = preds[1][:, colors.index(c1)] #.mean()
#     Relation_present = preds[2][:, relations.index(rel)] #.mean()
#     Shape2_present = preds[3][:, shapes.index(s2)] #.mean()
#     Colour2_present = preds[4][:, colors.index(c2)] #.mean()

#     shapes_absent = [s for s in shapes if s not in [s1, s2]]
#     colors_absent = [c for c in colors if c not in [c1, c2]]
#     relation_absent = [r for r in relations if r != rel][0]
#     Shape1_absent, Shape2_absent, Colour1_absent, Colour2_absent = [], [], [], []
#     for absent_shape in shapes_absent:
#         Shape1_absent.append(preds[0][:, shapes.index(absent_shape)]) #.mean())
#         Shape2_absent.append(preds[3][:, shapes.index(absent_shape)]) #.mean())
#     for absent_color in colors_absent:
#         Colour1_absent.append(preds[1][:, colors.index(absent_color)]) #.mean())
#         Colour2_absent.append(preds[4][:, colors.index(absent_color)]) #.mean())
#     Relation_absent = [preds[2][:, relations.index(relation_absent)]] #.mean() # cast to list, for consistency with other absents

#     return [Shape1_present, Colour1_present, Relation_present, Shape2_present, Colour2_present], [Shape1_absent, Colour1_absent, Relation_absent, Shape2_absent, Colour2_absent]


def get_subj_ave_preds(preds_by_presence, ave_preds, props):
    """ Updates the across subjects dict with the values
    for this subject
    """
    for prop in props:
        for presence in ['present', 'absent']:
            preds = preds_by_presence[f"{prop}_{presence}"]
            ave_preds[f"{prop}_{presence}"].append(np.nanmean(preds))
            if not len(preds): print(prop, presence)
    return ave_preds



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


def restructure_data(present, absent):
    """Converts present and absent lists into a single list of arrays with corresponding labels.
    Args:
        present (list): List of 5 arrays (n_samples, n_states) for present properties.
        absent (list): List of lists of 1 or 2 arrays (n_samples, n_states) for absent properties.
    Returns:
        all_arrays (list of np.ndarray): Flattened list of (n_samples, n_states) arrays.
        all_labels (list of tuples): Labels with (trial_idx, property, 'present'/'absent').
    """
    all_arrays, all_labels = [], []
    for prop, arr in zip(Properties, present): # Add present properties
        all_arrays.append(arr)
        all_labels.append((prop, "present"))
    for prop, sublist in zip(Properties, absent): # Add absent properties (list of lists)
        for abs_idx, arr in enumerate(sublist):
            all_arrays.append(arr)
            all_labels.append((prop, "absent"))
    return all_arrays, all_labels


# This one or the next? zscore or just percentile? 
def get_significant_reactivations(all_probs, threshold=0.9):
    """ get the significant reactivations
    for each property, and for each presence/absence
    threshold is applied to z-scored reactivations
    Maybe keep the actual value, to differentiate 
    "strong" from "weak" reactiations? 
    """
    significant_reactivations = [(zscore(arr) > threshold) for arr in all_probs]
    return significant_reactivations
    # reactivation_present = [(zscore(arr) > threshold) for arr in present]
    # reactivation_absent = [[(zscore(arr) > threshold) for arr in lst] for lst in absent]
    # return reactivation_present, reactivation_absent

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

# def get_sequential_reactivations(signif_react, maxLag):
#     """ From significant reactivations
#     get the sequence of reactivations , Sophie-style
#     and the delay between them (up to maxLag)
#     """
#     sequences, delays = [], []
#     for react in signif_react: # is this ones and zeros? the id of the significant react? 
#         if not react: continue
#         sequences.append(react)
#         for lag in range(maxLag):
#             if react[lag] > 0:
#                 pass
                
def get_reactivation_episodes(signif_react):
    """Extracts consecutive reactivations per state, and their duration
    (nb of significant consecutive reactivations for this state)
    For a single trial
    Args:
        signif_react (list of np.ndarray): List of 1D boolean arrays (n_samples,).
    Returns:
        list of tuples: [(start, end, state, duration), ...] for all episodes.
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
    (gap is the time elapsed since the last reactivation). Gap for first state is always None.
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


def count_coactivations(all_synchronous_episodes, state_set):
    """Counts occurrences where at least part of the state set is activated together.
    Args:
        all_synchronous_episodes (list): List of sets representing synchronous activations.
        state_set (set): The group of states to analyze.
    Returns:
        dict: Keys are the subset sizes (1-5), values are occurrence counts.
    """
    coactivation_counts = {i: 0 for i in range(1, len(state_set) + 1)}
    coactivation_overlaps = {i: [] for i in range(1, len(state_set) + 1)}

    for episode in all_synchronous_episodes:
        activated_states, _, overlap = episode
        nb_coactive_states = len(activated_states.intersection(state_set))

        if nb_coactive_states > 0:
            coactivation_counts[nb_coactive_states] += 1
            coactivation_overlaps[nb_coactive_states].append(overlap)

    # Remove empty lists to avoid warnings
    coactivation_overlaps = {k: v for k, v in coactivation_overlaps.items() if v}
    
    ave_coactivation_overlaps = {k: np.mean(v) for k, v in coactivation_overlaps.items()}
    return coactivation_counts, ave_coactivation_overlaps


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

    return summary_metrics


# def plot_average_preds(all_present, all_absent, out_fn, labels=["S1","C1","R","S2","C2"]):
#     """ bar plot of average predictions during the delay
#     all_present: list of np.array of len(n_subs), average for each subject of the predictions 
#     The length of all_present should match that of all_absent, and it will define the number of bars.
#     labels: list of str, same length as all_present/absent. Labels for each of the predictions.
#     additional_str: str to add to the out_fn, where the decoders were trained on (ImgLoc, scenes, ...)
#     """
#     all_present_ave, all_present_sem = [], []
#     for i in range(len(all_present)):
#         all_present_ave.append(np.nanmean(all_present[i]))
#         all_present_sem.append(sem(all_present[i], nan_policy='omit'))
#     all_absent_ave, all_absent_sem = [], []
#     for i in range(len(all_absent)):
#         all_absent_ave.append(np.mean(all_absent[i]))
#         all_absent_sem.append(sem(all_absent[i], nan_policy='omit'))
    
#     # Bar plot
#     # labels = ['Shape', 'Color', 'Relation']
#     x = np.arange(len(labels))  # the label locations
#     width = 0.35  # the width of the bars
#     fig, ax = plt.subplots(figsize=(10, 6))
#     rects1 = ax.bar(x - width/2, all_present_ave, width, yerr=all_present_sem, label='Present', color='skyblue')
#     rects2 = ax.bar(x + width/2, all_absent_ave, width, yerr=all_absent_sem, label='Absent', color='orange')

#     # Add labels, title, and legend
#     ax.set_ylabel('Average Predictions')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.legend()

#     # Perform t-tests for each category
#     alpha = 0.05
#     p_values = []
#     for present, absent in zip(all_present, all_absent):
#         ttest = ttest_ind(present, absent)
#         p_values.append(ttest.pvalue)

#     # Add significance stars
#     for i, p_val in enumerate(p_values):
#         if p_val < alpha:
#             y_max = max(all_present_ave[i] + all_present_sem[i], all_absent_ave[i] + all_absent_sem[i])
#             ax.text(i, y_max + 0.05, '*', ha='center', va='bottom', fontsize=16, color='k')

#     # Save the plot
#     plt.tight_layout()
#     plt.savefig(out_fn, dpi=400)

#     plt.close()


def plot_average_preds_seaborn(all_present, all_absent, out_fn, labels=["S1", "C1", "R", "S2", "C2"]):
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
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='Category', y='Prediction', hue='Condition', errorbar=('se', 1),
                     palette={'Present': 'skyblue', 'Absent': 'orange'})
    
    # Perform statistical tests and annotate
    # pairs = [(label, label) for label in labels]
    pairs = [((label, 'Present'), (label, 'Absent')) for label in labels]
    # p_values = [ttest_ind(all_present[i], all_absent[i], nan_policy='omit').pvalue for i in range(len(labels))]
    annotator = Annotator(ax, pairs, data=df, x='Category', y='Prediction', hue='Condition')
    annotator.configure(test='t-test_ind', text_format='star', loc='outside')
    # annotator.set_pvalues(p_values)
    # annotator.annotate()
    annotator.apply_and_annotate()
    
    # Labels and legend
    plt.ylabel('Average Predictions')
    plt.legend(title='Condition')
    plt.tight_layout()
    
    # Save and close
    plt.savefig(out_fn, dpi=400)
    plt.close()


def does_reactivations_predict_behavioral(df, out_fn, y="Performance"):
    """
    Regression plot of reactivations vs. behavioral performance.
    (for present items)
    """
        
    # Mixed-effects model (accounts for repeated measures within subjects)
    # crossed random effects of subject and property
    # vc = {"Property": "0 + C(Property)"}  # Define Property as a random effect
    # model = smf.mixedlm(f"{y} ~ Reactivation", df, groups=df["Subject"], vc_formula=vc).fit()
    # hierarchical random effect of subject and property
    # model = smf.mixedlm(f"{y} ~ Reactivation", df, groups=df["Subject"], re_formula="1 + Property").fit()

    # random effect of subjects
    model = smf.mixedlm(f"{y} ~ Reactivation", df, groups=df["Subject"]).fit()
    print(model.summary())

    # Scatter plot with subject-level data
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Reactivation", y=y, hue="Subject", palette="tab10", alpha=0.6)

    # Add a regression line for the overall effect
    sns.regplot(data=df, x="Reactivation", y=y, scatter=False, color="black")

    plt.xlabel("Reactivation Strength")
    plt.ylabel("Behavioral Performance")
    # plt.title("Mixed-Effects Model: Reactivation vs. Performance")
    plt.title(f"tvalue: {model.tvalues["Reactivation"]:.3f} - pvalue: {model.pvalues["Reactivation"]:.3f}")
    ax = plt.gca()
    ax.get_legend().remove()
    # plt.legend(title="Subject", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_fn, dpi=400)
    plt.close()



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

