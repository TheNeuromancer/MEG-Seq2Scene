from dataclasses import dataclass

@dataclass
class Config:
    """Class for keeping all parameters."""

    # paths and names
    version: str = "14"
    root_path: str = "/neurospin/unicog/protocols/MEG/Seq2Scene/"
    epochs_dir: str = "Epochs_100hz"
    all_subjects: tuple = ('01_js180232', '02_jm100042', '03_cr170417', '04_ag170045', '05_mb140004', '06_ll180197', '07_jv200206', \
                           '08_ch180036', '09_jl190711', '10_ma200371', '11_rb210035', '12_mb160165', '13_lg170436', '14_eb180237', \
                           '15_ar160084', '16_er123987', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30')

    # Epochs preprocessing parameters
    baseline: bool = True # apply baseline correction if True
    ch_var_reject: int = 0 # threshold for the variance-based channel rejection, in number of std
    epo_var_reject: int = 0 # threshold for the variance-based epochs rejection, in number of std
    ref_run: int = 5 # reference run for head position for maxwell filter
    l_freq: float = 0.1 # high-pass filter cutoff
    h_freq: float = 30 # low-pass filter cutoff
    notch: int = 50 # land line frequency
    sfreq: int = 100 # final sampling frequency

    ## Decoding parameters
    n_folds: int = 5 # number of shuffle splits
    crossval: str = "kfold" # cross-validation scheme. "kfold" or "sufflesplit"
    reduc_dim: float = 0 # dimensionality reduction
    ## Decoding window parameters
    n_folds_win: int = 10 # number of shuffle splits for the window decoding analysis 
    crossval_win: str = "kfold" # cross-validation scheme for the window decoding analysis 
    reduc_dim_win: float = 0 # dimensionality reduction for the window decoding analysis 
    ## Decoding window parameters
    reduc_dim_sing: float = 0 # dimensionality reduction
    cat: int = 10 # number of timepoints to concatenate
    mean: bool = False # Wether to average instead of concatenate if using the "cat" argument
    smooth: int = 21 # hanning smoothing window, in timesample,
    clip: bool = True # Whether to clip to the 5th and 95th percentile for each channel
    subtract_evoked: bool = False # Whether to subtract the evoked signal from the epochs
    avg_clf: bool = False # Whether to average classifiers across cval folds
    autoreject: bool = False
    xdawn: bool = False
    quality_th: float = 0 #.75
    filter: str = "Perf==1"
    equalize_events: bool = True

    # # TRF delays
    # tstart: float = 0.
    # tstop: float = 1.


    def print_subjects_names(self):
        # convenience for bash scripts that needs to loop over subjects
        for sub in self.all_subjects:
            print(sub, end=' ')


if __name__ == "__main__":
    Config().print_subjects_names()
