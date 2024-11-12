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
import importlib

from utils.params import *
from utils.reject import *
from utils.commons import *

parser = argparse.ArgumentParser(description='Load and convert to MNE Raw, then preprocess, make Epochs and save')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='03_cr170417',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
# parser.add_argument('--plot', default=False, action='store_true', help='Whether to plot some channels and power spectrum')
# parser.add_argument('--show', default=False, action='store_true', help='Whether to show some channels and power spectrum ("need to be locally or ssh -X"')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
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
    run_nb = op.basename(raw_fn_in).split("_")[0].split("run")[1]
    out_fn = f"{in_dir}/run{run_nb}_bads.p"
    if op.exists(out_fn):
        print("output file alreay exists")
        if args.overwrite:
            print("Overwriting")
        else:
            print("Overwrite is set to false, moving on to next block")

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
    if args.ch_var_reject:
        bads = get_deviant_ch(raw, thresh=args.ch_var_reject)
    else:
        bads = []
    
    pickle.dump(bads, open(out_fn, "wb"))
    print(f"Done with run {run_nb}\n\n")
