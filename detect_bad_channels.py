import matplotlib
matplotlib.use('Agg') # no output to screen.
import matplotlib.pyplot as plt

import os.path as op
import os
from glob import glob
from ipdb import set_trace
import mne
import argparse
import pandas as pd
import pickle
import numpy as np

from utils.params import *
from utils.reject import *

parser = argparse.ArgumentParser(description='MEG basic preprocessing')
parser.add_argument('-r', '--root-path', default='/neurospin/unicog/protocols/MEG/Seq2Scene/', help='Path to parent project folder')
parser.add_argument('-p', '--subject', default='theo',help='subject name')
parser.add_argument('--l_freq', default=1., type=float, help='Low-pass filter frequency (0 for only high pass')
parser.add_argument('--h_freq', default=60, type=float, help='High_pass filter frequency(0 for only low pass)')
parser.add_argument('--notch', default=50, type=int, help='frequency of the notch filter (0 = no notch filtering)')
parser.add_argument('--plot', default=False, action='store_true', help='Whether to plot some channels and power spectrum')
parser.add_argument('--ch-var-reject', default=20, type=int, help='whether to reject channels based on temporal variance')

### TODO: OPTIONALLY PASS A LIST OF BAD MEG SENSORS AS ARGUMENT? OR A PATH TO A TXT OR CSV FILE CONTAINING THE BAD SENSORS

print(mne.__version__)
args = parser.parse_args()
print(args)

in_dir = op.join(args.root_path + '/Data', 'orig', args.subject)
all_runs_fns = glob(in_dir + '/*run*.fif')
all_runs_fns = sorted(all_runs_fns)

if args.subject=="theo": # skip the 5th blocks that have a problem (split in two)
    all_runs_fns = [run for run in all_runs_fns if "run05_1obj_first" not in run]

print(all_runs_fns)
n_runs = len(all_runs_fns)


for i_run, raw_fn_in in enumerate(all_runs_fns):
    print("doing file ", raw_fn_in)

    # Load data
    raw = mne.io.Raw(raw_fn_in, preload=True, verbose='error', allow_maxshield=True)

    # remvoe projs
    raw = raw.del_proj("all")
    
    # Band-pass the data channels
    if args.l_freq or args.h_freq:
        if args.l_freq == 0: args.l_freq = None
        if args.h_freq == 0: args.h_freq = None
        print("Filtering data between %s and %s (Hz)" %(args.l_freq, args.h_freq))
        raw.filter(
            args.l_freq, args.h_freq,
            l_trans_bandwidth='auto',
            h_trans_bandwidth='auto',
            filter_length='auto', phase='zero', fir_window='hamming',
            fir_design='firwin')


    if args.notch:
        print(f"applying notch filter at {args.notch} and 3 harmonics")
        notch_freqs = [args.notch, args.notch*2, args.notch*3, args.notch*4]
        raw = raw.notch_filter(notch_freqs)
    

    # detect and reject bad channels
    bads = get_deviant_ch(raw, thresh=args.ch_var_reject)
    
    pickle.dump(bads, open(f"{in_dir}/run{i_run+1}_bads.p", "wb"))
    print(f"Done with run {i_run+1}\n\n")
