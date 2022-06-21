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
parser.add_argument('--train-queries', help='Metadata query for training classes')
parser.add_argument('--test-cond', default=[], action='append', help='localizer, one_object or two_objects, should have the same length as test-queries')
parser.add_argument('--test-queries', default=[], action='append', help='Metadata query for testing classes')
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
parser.add_argument('--label', default='', help='help to identify the result latter')
parser.add_argument('--dummy', action='store_true', default=False, help='Accelerates everything so that we can test that the pipeline is working. Will not yield any interesting result!!')
parser.add_argument('--equalize_events', action='store_true', default=False, help='subsample majority event classes to get same number of trials as the minority class')
parser.add_argument('-x', '--xdawn', action='store_true',  default=False, help='Whether to apply Xdawn spatial filtering before training decoder')
parser.add_argument('-a', '--autoreject', action='store_true',  default=False, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('--n_comp', default=100, type=int, help='Number of PCA components to use for reconstruction')

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
train_fn, test_fns, out_fn, test_out_fns = get_paths(args)

print('\nStarting training')
### LOAD EPOCHS ###
epochs = load_data(args, train_fn, args.train_queries)
# test_split_query_indices = get_split_indices(args.split_queries, epochs)
train_tmin, train_tmax = epochs[0].tmin, epochs[0].tmax
### GET DATA AND CONSTRUCT LABELS ###
if args.train_queries:
    X, y, nchan, ch_names = get_X_y_from_epochs_list(args, epochs, args.sfreq)
else: # no query, take all the data
    X = epochs[0].get_data()
    nchan = X.shape[1]
del epochs

if nchan < args.n_comp: # we can have at most nchan components 
    args.n_comp = nchan 

n_times = X.shape[2]
all_timepoints_orig = np.arange(0, n_times, 20) # +1
all_timepoints = all_timepoints_orig[all_timepoints_orig > args.n_comp]

X_splits = split_for_pca(args, X)

all_exvar = {}
all_MSE = {}
all_l2 = {}
time = {n_timepoints: [] in all_timepoints}
for n_timepoints in all_timepoints: # multiple windows
    print(n_timepoints)
    all_exvar[n_timepoints] = []
    all_MSE[n_timepoints] = []
    all_l2[n_timepoints] = []
    for i, (X_train, X_test) in enumerate(X_splits):

        exvar, MSE, l2 = PCA_dim(args, X_train, X_test)
        all_exvar[n_timepoints].append(exvar)
        all_MSE[n_timepoints].append(MSE)
        all_l2[n_timepoints].append(l2)

    # average over folds
    all_exvar[n_timepoints] = np.mean(all_exvar[n_timepoints], 0)
    all_MSE[n_timepoints] = np.mean(all_MSE[n_timepoints], 0)
    all_l2[n_timepoints] = np.mean(all_l2[n_timepoints], 0)


from ipdb import set_trace; set_trace()