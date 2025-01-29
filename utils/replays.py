import mne
import os.path as op
import os
from glob import glob
import pandas as pd
import numpy as np
import pickle
from scipy.stats import sem, pearsonr, zscore
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.linear_model import LinearRegression
# from scipy.linalg import toeplitz
from tqdm import tqdm

from .params import *


def get_trial_preds_from_data(df_trial, all_preds_data, labels=["S1_0", "C1_0", "R_0", "S2_0", "C2_0"]):
    """ get the predictions for a given trial 
    df_trial might have more than 5 entries, because decoders may have been tested multiple times
    (but should then have the same predictions)
    So only take the first generalization (X_0), and test for uniqueness
    it is the same in every case because we generalize to the delay period. 
    Later 04_decod should not even save it. 
    """
    preds_shape1_idx = df_trial.query(f"label=='{labels[0]}'").index.values
    # if len (preds_shape1_idx) > 1: print(f"len(preds_shape1_idx)={len(preds_shape1_idx)} for {labels[0]}")
    # if len(preds_shape1_idx) > 1:
     # there are a few duplicates, not sure why. DO NOT Remove them because it fucks up the indexing, and as all_preds_data have the same indexing. 
        # df.drop_duplicates(inplace=True)
    # from ipdb import set_trace; set_trace()
    # assert len(preds_shape1_idx) == 1, f"len(preds_shape1_idx)={len(preds_shape1_idx)} for {labels[0]}"
    preds_shape1 = all_preds_data[preds_shape1_idx[0]]

    preds_color1_idx = df_trial.query(f"label=='{labels[1]}'").index.values
    # assert len(preds_color1_idx) == 1, f"len(preds_color1_idx)={len(preds_color1_idx)} for {labels[1]}"
    preds_color1 = all_preds_data[preds_color1_idx[0]]

    preds_rel_idx = df_trial.query(f"label=='{labels[2]}'").index.values
    # assert len(preds_rel_idx) == 1, f"len(preds_rel_idx)={len(preds_rel_idx)} for {labels[2]}"
    preds_rel = all_preds_data[preds_rel_idx[0]]

    preds_shape2_idx = df_trial.query(f"label=='{labels[3]}'").index.values
    # assert len(preds_shape2_idx) == 1, f"len(preds_shape2_idx)={len(preds_shape2_idx)} for {labels[3]}"
    preds_shape2 = all_preds_data[preds_shape2_idx[0]]

    preds_color2_idx = df_trial.query(f"label=='{labels[4]}'").index.values
    # assert len(preds_color2_idx) == 1, f"len(preds_color2_idx)={len(preds_color2_idx)} for {labels[4]}"
    preds_color2 = all_preds_data[preds_color2_idx[0]]


    return [preds_shape1, preds_color1, preds_rel, preds_shape2, preds_color2]


def add_present_or_absent_preds_one_trial(preds, props, preds_sub_by_presence):
    """ take the predictions for a single trial
    and put them in the dict that stores all preds
    with this subjects, as a function of the property
    and whether it was present or absent in the sentence.
    """ 
    s1, c1, rel, s2, c2 = props
    preds_sub_by_presence["Shape1_present"].append(preds[0][:, shapes.index(s1)].mean())
    preds_sub_by_presence["Colour1_present"].append(preds[1][:, colors.index(c1)].mean())
    preds_sub_by_presence["Relation_present"].append(preds[2][:, relations.index(rel)].mean())
    preds_sub_by_presence["Shape2_present"].append(preds[3][:, shapes.index(s2)].mean())
    preds_sub_by_presence["Colour2_present"].append(preds[4][:, colors.index(c2)].mean())

    shapes_absent = [s for s in shapes if s not in [s1, s2]]
    colors_absent = [c for c in colors if c not in [c1, c2]]
    relation_absent = [r for r in relations if r != rel][0]
    for absent_shape in shapes_absent:
        preds_sub_by_presence["Shape1_absent"].append(preds[0][:, shapes.index(absent_shape)].mean())
        preds_sub_by_presence["Shape2_absent"].append(preds[3][:, shapes.index(absent_shape)].mean())
    for absent_color in colors_absent:
        preds_sub_by_presence["Colour1_absent"].append(preds[1][:, colors.index(absent_color)].mean())
        preds_sub_by_presence["Colour2_absent"].append(preds[4][:, colors.index(absent_color)].mean())
    preds_sub_by_presence['Relation_absent'].append(preds[2][:, relations.index(relation_absent)].mean())

    ## TODO: Separate present first, present second, and absent? 
    return preds_sub_by_presence


def get_subj_ave_preds(preds_by_presence, ave_preds):
    """ Updates the across subjects dict with the values
    for this subject
    """
    for prop in Properties:
        for presence in ['present', 'absent']:
            preds = preds_by_presence[f"{prop}_{presence}"]
            ave_preds[f"{prop}_{presence}"].append(np.nanmean(preds))
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


def splits(data, num_splits=30):
    """
    Splits the data into `num_splits` equal-length segments,
    and then calculates the overall mean and SEM across splits.
    In place of the proper subject averaging. To be replaced.
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
        mean, stand_err = splits(present[i])
        present_ave.append(mean)
        present_sem.append(stand_err)
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
        mean, stand_err = splits(absent[i])
        absent_ave.append(mean)
        absent_sem.append(stand_err)
    
    # Bar plot
    labels = ['Shape', 'Color', 'Relation']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, present_ave, width, yerr=present_sem, label='Present', color='skyblue')
    rects2 = ax.bar(x + width/2, absent_ave, width, yerr=absent_sem, label='Absent', color='orange')

    # Add labels, title, and legend
    ax.set_ylabel('Average Predictions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

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


from scipy.stats import zscore

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

