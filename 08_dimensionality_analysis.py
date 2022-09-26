import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import mne
import numpy as np
import argparse
import pickle
import time
import importlib
from ipdb import set_trace
from scipy.signal import detrend

from utils.dim import *

parser = argparse.ArgumentParser(description='MEG Decoding analysis')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='theo',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle sentence labels before training')
parser.add_argument('--freq-band', default='', help='name of frequency band to use for filtering (theta, alpha, beta, gamma)')
parser.add_argument('--filter', default='', help='md query to filter trials before anything else (eg to use only matching trials')
parser.add_argument('--train-cond', default='localizer', help='localizer, one_object or two_objects')
parser.add_argument('--train-query', help='Metadata query for training classes')
parser.add_argument('--test-cond', default=[], action='append', help='localizer, one_object or two_objects, should have the same length as test-queries')
parser.add_argument('--test-queries', default=[], action='append', help='Metadata query for testing classes')
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
parser.add_argument('--label', default='', help='help to identify the result latter')
parser.add_argument('--dummy', action='store_true', default=False, help='Accelerates everything so that we can test that the pipeline is working. Will not yield any interesting result!!')
parser.add_argument('--equalize_events', action='store_true', default=None, help='subsample majority event classes to get same number of trials as the minority class')
parser.add_argument('-x', '--xdawn', action='store_true',  default=None, help='Whether to apply Xdawn spatial filtering before training decoder')
parser.add_argument('-a', '--autoreject', action='store_true',  default=None, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('-r', '--response_lock', action='store_true',  default=None, help='Whether to Use response locked epochs or classical stim-locked')
parser.add_argument('--micro_ave', default=None, type=int, help='Trial micro-averaging to boost decoding performance')
parser.add_argument('--max_trials', default=None, type=int, help='Trial micro-averaging max nb of trials')
# parser.add_argument('--reconstruct', action='store_true', default=False, help='Whether to reconstruct stimulus and test accuracy')
parser.add_argument('--queries', default=[], action='append', help='If specified, reconstructs stimulus and test accuracy for each query')

# actually we set the nb of components to be highest possible (depending on the data), ie min(nchan, n_trials)
# parser.add_argument('--n_comp', default=250, type=int, help='Number of PCA components to use for reconstruction')
parser.add_argument('--detrend', action='store_true', default=False, help='detrend epochs beforehand')

# optionals, overwrite the config if passed
parser.add_argument('--sfreq', type=int, help='sampling frequency')
parser.add_argument('--n_folds', type=int, help='sampling frequency')

# not implemented
parser.add_argument('--localizer', action='store_true', default=False, help='Whether to use only electrode that were significant in the localizer')
parser.add_argument('--auc_thresh', default=0.55, type=float, help='pvalue threshold under which a channel is kept for the localizer')
parser.add_argument('--min_nb_ch', default=50, type=int, help='pvalue threshold under which a channel is kept for the localizer')
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

np.random.seed(args.seed)
start_time = time.time()

###########################
######## TRAINING #########
###########################

### GET EPOCHS FILENAMES ###
train_fn, test_fns, out_fn, test_out_fns = get_paths(args, dirname='Dimensionality')
if args.localizer: out_fn += f"_{args.auc_thresh}th"
print(out_fn)

print('\nStarting training')
### LOAD EPOCHS ###
if args.train_query:
    epochs = load_data(args, train_fn, [args.train_query])[0]
else:
    epochs = load_data(args, train_fn, [])[0]
    out_fn += "_all_trials"


tmin, tmax = epochs.tmin, epochs.tmax
### GET DATA AND CONSTRUCT LABELS ###
X = epochs.get_data().transpose((0,2,1)) # trials, times, channels
if args.detrend:
    out_fn += "_detrended"
    # from ipdb import set_trace; set_trace()
    detrend(X, axis=1, type='linear', overwrite_data=True)
n_trials, n_times, nchan = X.shape
if args.min_nb_ch > nchan: 
    print(f"Not enough channels left after localizer, exiting smoothly")
    exit()
times = np.linspace(tmin, tmax, n_times)

if args.micro_ave:
    print(f"Using extensive trial micro-averaging, starting with {len(X)} trials")
    X, _ = micro_averaging(X, np.ones(len(X)), args.micro_ave)
    if args.max_trials and len(X) > args.max_trials:
        indices = np.random.choice(np.arange(len(X)), args.max_trials, replace=False)
        X = X[indices]
    print(f"ending with {len(X)} trials")

def get_window_indices(times, start, stop):
    start_idx = np.argmin(np.abs(times - start))
    stop_idx = np.argmin(np.abs(times - stop))
    return start_idx, stop_idx

pca = PCA()
scaler = StandardScaler()

# if args.queries: all_PR_queries = {query: np.zeros(n_times) for query in args.queries}
win_length = .1 #.05 # 50 ms
all_windows = np.arange(tmin, tmax-win_length, win_length)
all_windows = np.round(all_windows, 2)
all_PR = np.zeros(len(all_windows))
# for t in tqdm(range(n_times)):
for w, win in enumerate(all_windows):
    # print(f"using window from {win} to {win+win_length}")
    win_start, win_stop = get_window_indices(times, win, win+win_length)
    # print(f"corresponding indices: {win_start}, {win_stop}")
    x = X[:, win_start:win_stop, :]

    # concat time and trials
    x = np.concatenate(x)
    x = scaler.fit_transform(x)
    # print(x)
    # from ipdb import set_trace; set_trace()

    pca.fit(x)
    all_PR[w] = participation_ratio(pca.explained_variance_)

print(all_PR)
save_results(out_fn, all_PR, fn_end="PR")
# if args.queries:
#     save_results(out_fn, reconstruct_L2, fn_end="reconstruction_L2")
