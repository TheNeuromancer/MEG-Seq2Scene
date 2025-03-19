import numpy as np
import mne
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import entropy, ttest_1samp
from tqdm import trange


def bandpass_filter(data, sfreq, low, high):
    """Apply a bandpass filter to the data."""
    nyq = 0.5 * sfreq  # Nyquist frequency
    low /= nyq
    high /= nyq
    if low <= 0 or high >= 1:
        raise ValueError(f"Filter frequency out of bounds: {low * nyq}-{high * nyq} Hz (Nyquist = {nyq} Hz)")
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def compute_pac(epochs, picks, gamma_range=(70, 100), theta_range=(3, 8), n_theta_bins=18, n_permutations=100, average_trials=False):
    """Compute Phase-Amplitude Coupling (PAC) modulation index on MNE Epochs object."""
    sfreq = epochs.info['sfreq']
    data = epochs.get_data(picks=picks)  # Shape: (n_trials, n_sensors, n_times)
    n_trials, n_sensors, n_times = data.shape
    
    pac_values = np.zeros((n_sensors, n_trials))
    null_distributions = np.zeros((n_sensors, n_trials, n_permutations))
    
    for i in trange(n_sensors):
        for j in range(n_trials):
            trial_data = data[j, i, :]
            
            # Compute gamma amplitude
            gamma_filtered = bandpass_filter(trial_data, sfreq, gamma_range[0], gamma_range[1])
            gamma_envelope = np.abs(hilbert(gamma_filtered))
            
            # Compute theta phase
            theta_phases = []
            for f in range(theta_range[0], theta_range[1]):
                theta_filtered = bandpass_filter(trial_data, sfreq, f, f + 1)
                theta_phase = np.angle(hilbert(theta_filtered))
                theta_phases.append(theta_phase)
            theta_phases = np.mean(theta_phases, axis=0)  # Average across sub-bands
            
            # Bin gamma power by theta phase
            phase_bins = np.linspace(-np.pi, np.pi, n_theta_bins + 1)
            gamma_histogram = np.zeros(n_theta_bins)
            
            for b in range(n_theta_bins):
                indices = (theta_phases >= phase_bins[b]) & (theta_phases < phase_bins[b + 1])
                gamma_histogram[b] = np.mean(gamma_envelope[indices]) if np.any(indices) else 0
            
            gamma_histogram /= np.sum(gamma_histogram)  # Normalize
            
            # Compute KL divergence
            uniform_dist = np.ones(n_theta_bins) / n_theta_bins
            mi = entropy(gamma_histogram, uniform_dist) / np.log(n_theta_bins)
            pac_values[i, j] = mi
            
            # Phase scrambling for permutation test
            for p in range(n_permutations):
                shift = np.random.randint(500, n_times)  # Circular shift by at least 500 samples
                shifted_phase = np.roll(theta_phases, shift)
                
                shuffled_histogram = np.zeros(n_theta_bins)
                for b in range(n_theta_bins):
                    indices = (shifted_phase >= phase_bins[b]) & (shifted_phase < phase_bins[b + 1])
                    shuffled_histogram[b] = np.mean(gamma_envelope[indices]) if np.any(indices) else 0
                shuffled_histogram /= np.sum(shuffled_histogram)
                
                null_mi = entropy(shuffled_histogram, uniform_dist) / np.log(n_theta_bins)
                null_distributions[i, j, p] = null_mi
    
    # Compute z-scores
    mean_null = np.mean(null_distributions, axis=-1)
    std_null = np.std(null_distributions, axis=-1)
    z_scores = (pac_values - mean_null) / std_null
    
    # Compute within-subject t-statistics
    t_stats_trials = ttest_1samp(z_scores, 0, axis=1).statistic  # Across trials
    
    if average_trials:
        pac_values = np.mean(pac_values, axis=1)  # Average across trials
        z_scores = np.mean(z_scores, axis=1)
    
    return pac_values, z_scores, t_stats_trials
