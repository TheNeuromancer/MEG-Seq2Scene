from dataclasses import dataclass

@dataclass
class Config:
    """Class for keeping all parameters."""

    # paths and names
    version: str = "2"
    root_path: str = "/neurospin/unicog/protocols/MEG/Seq2Scene/"
    epochs_dir: str = "Epochs"
    all_subjects: tuple = ('01_js180232', '02_jm100042', '03_cr170417', '04_ag170045', '05_mb140004', '06_ll180197', '07_jv200206', \
                           '08_ch180036', '09_jl190711', '10_ma200371')

    # Epochs preprocessing parameters
    baseline: bool = True # apply baseline correction if True
    ch_var_reject: int = 50 # threshold for the variance-based channel rejection, in number of std
    epo_var_reject: int = 20 # threshold for the variance-based epochs rejection, in number of std
    ref_run: int = 8 # reference run for head position for maxwell filter
    l_freq: float = 0.03 # high-pass filter cutoff
    h_freq: float = 100 # low-pass filter cutoff
    notch: int = 50 # land line frequency
    sfreq: int = 100 # final sampling frequency

    ## Decoding parameters
    n_folds: int = 5 # number of shuffle splits
    crossval: str = "kfold" # cross-validation scheme. "kfold" or "sufflesplit"
    reduc_dim: float = 0 # dimensionality reduction
    cat: int = 0 # number of timepoints to concatenate
    mean: bool = False # Wether to average instead of concatenate if using the "cat" argument
    smooth: int = 11 # hanning smoothing window, in timesample,
    clip: bool = True # Whether to clip to the 5th and 95th percentile for each channel
    subtract_evoked: bool = False # Whether to subtract the evoked signal from the epochs
    avg_clf: bool = False # Whether to average classifiers across cval folds

    # # TRF delays
    # tstart: float = 0.
    # tstop: float = 1.


    def print_subjects_names(self):
        # convenience for bash scripts that needs to loop over subjects
        for sub in self.all_subjects:
            print(sub, end=' ')


if __name__ == "__main__":
    Config().print_subjects_names()