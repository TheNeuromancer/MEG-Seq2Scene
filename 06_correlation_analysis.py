import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import mne
import numpy as np
# from ipdb import set_trace
import argparse
import pickle
import time
import importlib

from utils.decod import *

parser = argparse.ArgumentParser(description='MEG correlation analysis')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='30',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle sentence labels before training')
parser.add_argument('--freq-band', default='', help='name of frequency band to use for filtering (theta, alpha, beta, gamma)')
parser.add_argument('--filter', default='Complexity==2', help='md query to filter trials before anything else (eg to use only matching trials')
parser.add_argument('--train-cond', default='two_objects', help='localizer, one_object or two_objects')
parser.add_argument('--train-queries', help='Metadata query for training classes')
parser.add_argument('--test-cond', default=[], action='append', help='localizer, one_object or two_objects, should have the same length as test-queries')
parser.add_argument('--test-queries', default=[], action='append', help='Metadata query for testing classes')
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
parser.add_argument('--label', default='', help='help to identify the result latter')
parser.add_argument('--dummy', action='store_true', default=False, help='Accelerates everything so that we can test that the pipeline is working. Will not yield any interesting result!!')
parser.add_argument('--equalize_events', action='store_true', default=False, help='subsample majority event classes to get same number of trials as the minority class')
parser.add_argument('-x', '--xdawn', action='store_true',  default=False, help='Whether to apply Xdawn spatial filtering before training decoder')
parser.add_argument('-a', '--autoreject', action='store_true',  default=False, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('-r', '--response_lock', action='store_true',  default=None, help='Whether to Use response locked epochs or classical stim-locked')
parser.add_argument('--windows', default=[], action='append', help='tmin and tmax to crop the epochs, one for each train and test cond')
parser.add_argument('--mirror_img', action='store_true',  default=False, help='Whether to consider mirror images as the same or not')

# parser.add_argument('--n_comp', default=100, type=int, help='Number of PCA components to use for reconstruction')

# optionals, overwrite the config if passed
parser.add_argument('--sfreq', type=int, help='sampling frequency')
parser.add_argument('--n_folds', type=int, help='sampling frequency')

# not implemented
parser.add_argument('--localizer', action='store_true', default=False, help='Whether to use only electrode that were significant in the localizer')
parser.add_argument('--path2loc', default='Single_Chan_vs5/CMR_sent', help='path to the localizer results (dict with value 1 for each channel that passes the test, 0 otherwise')
parser.add_argument('--pval-thresh', default=0.05, type=float, help='pvalue threshold under which a channel is kept for the localizer')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
for arg in vars(args): # update config with arguments from the argparse
    if getattr(args, arg) is not None:
        setattr(config, arg, getattr(args, arg))
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)
print("matplotlib: ", matplotlib.__version__)
print("mne: ", mne.__version__)

# if len(args.test_cond) != len(args.test_queries[0]) or len(args.test_cond) != len(args.test_queries[1]):
#     raise RuntimeError("Test conditions and test-queries should have the same length")

np.random.seed(args.seed)
start_time = time.time()

###########################
######## TRAINING #########
###########################

### GET EPOCHS FILENAMES ###
print('\nStarting loading data')
train_fn, _, out_fn, _ = get_paths(args, "Correlation")
if args.mirror_img: out_fn += "_mirror"
if args.windows:
    args.windows = [w.replace(" ", "") for w in args.windows] # remove spaces
    wins = [f"#{'#'.join([args.windows[0], w])}#" for w in args.windows] # string to add to the out fns
    out_fn += wins[0]
### LOAD EPOCHS ###
epochs = load_data(args, train_fn)[0]
windows = [tuple([float(x) for x in win.split(",")]) for win in args.windows]
if windows: epochs = epochs.crop(*windows[0])
train_tmin, train_tmax = epochs[0].tmin, epochs[0].tmax
### GET DATA AND CONSTRUCT LABELS ###
matched, nonmatched = get_X_y_for_correlation(args, epochs, subsample_nonmatched=10)
## shape (n_trials, 2, nch, n_times)
nchan = epochs.info['nchan']
del epochs

# we are left with n_trials * n_times, which we can safely flatten to get the correlation for a single channel
r_matched, p_matched = np.zeros(nchan), np.zeros(nchan)
for ch in tqdm(range(nchan)):
    r, p = pearsonr(matched[:,0, ch].flatten(), matched[:,1, ch].flatten())
    r_matched[ch] = r
    p_matched[ch] = p

r_nonmatched, p_nonmatched = np.zeros(nchan), np.zeros(nchan)
for ch in tqdm(range(nchan)):
    r, p = pearsonr(nonmatched[:,0, ch].flatten(), nonmatched[:,1, ch].flatten())
    r_nonmatched[ch] = r
    p_nonmatched[ch] = p


print(np.mean(r_matched), np.mean(r_nonmatched))

save_results(out_fn, np.c_[r_matched, r_nonmatched], time=False, all_models=None, fn_end="R")

print("ALL DONE")
# from ipdb import set_trace; set_trace()