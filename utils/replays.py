
def transition_matrix(sequence, n_states):
    # Initialize an n_states x n_states transition matrix with zeros
    T = np.zeros((n_states, n_states), dtype=int)
    # Loop over each pair of consecutive states in the sequence
    for (from_state, to_state) in zip(sequence[:-1], sequence[1:]):
        T[from_state - 1, to_state - 1] += 1  # Adjust indices to be zero-based
    return T
    # go from states to state indices
    # indices = np.argsort(episode)
    # T[from_state, to_state] += 1

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


def compute_empirical_transition_matrix(time_series, lag):
    """
    Compute the empirical transition matrix for stimulus activations.

    Parameters:
    - time_series: np.ndarray
        A 2D array of shape (n_samples, n_stimuli), where each column represents 
        the reactivation time series for a stimulus.
    - lag: int
        The time lag (Δt) for which the transition matrix is computed.

    Returns:
    - transition_matrix: np.ndarray
        A 6x6 matrix of regression coefficients describing transitions between stimuli.
    """
    n_samples, n_stimuli = time_series.shape

    # Initialize the transition matrix
    transition_matrix = np.zeros((n_stimuli, n_stimuli))

    # Iterate over each stimulus (as the target)
    for target_stimulus in range(n_stimuli):
        # Define the target variable (Y_i)
        target = time_series[lag:, target_stimulus]  # Target variable starts from lag

        # Define the predictors (lagged time series of all stimuli)
        predictors = np.zeros((n_samples - lag, n_stimuli))
        for stimulus in range(n_stimuli):
            predictors[:, stimulus] = time_series[:n_samples - lag, stimulus]

        # Fit the linear model
        model = LinearRegression(fit_intercept=True)
        model.fit(predictors, target)

        # Store the coefficients in the transition matrix
        transition_matrix[target_stimulus, :] = model.coef_

    return transition_matrix


# def compute_empirical_transition_matrix_all_trials(data, lag):
#     """
#     Compute empirical transition matrix across multiple trials.
    
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


def compute_empirical_transition_matrix_all_trials(data, lag=1):
    """
    Compute empirical transition matrix for multiple trials at a given time lag.

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
        empirical_transition_matrix = compute_empirical_transition_matrix_all_trials(permuted_timecourses, lag)

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
