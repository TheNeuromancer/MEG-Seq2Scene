import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import mne
import numpy as np
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
parser.add_argument('--reconstruct_queries', default=[], action='append', help='If specified, reconstructs stimulus and test accuracy for each query')

# actually we set the nb of components to be highest possible (depending on the data), ie min(nchan, n_trials)
# parser.add_argument('--n_comp', default=250, type=int, help='Number of PCA components to use for reconstruction')


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
train_fn, test_fns, out_fn, test_out_fns = get_paths(args, dirname='Dimensionality')
print(out_fn)

print('\nStarting training')
### LOAD EPOCHS ###
if args.train_query:
    epochs = load_data(args, train_fn, [args.train_query])[0]
else:
    epochs = load_data(args, train_fn, [])[0]
    out_fn += "_all_trials"
# test_split_query_indices = get_split_indices(args.split_queries, epochs)
train_tmin, train_tmax = epochs.tmin, epochs.tmax
### GET DATA AND CONSTRUCT LABELS ###
# if args.train_query:
    # X, y, nchan, ch_names = get_X_y_from_epochs_list(args, epochs, args.sfreq)
# else: # no query, take all the data
X = epochs.get_data()
n_trials, nchan, n_times = X.shape
if args.micro_ave: # just print, trial averaging happens later
    print(f"Using extensive trial micro-averaging, starting with {len(X)} trials")
#     X, _ = micro_averaging(X, np.ones(len(X)), args.micro_ave)
#     if args.max_trials and len(X) > args.max_trials:
#         indices = np.random.choice(np.arange(len(X)), args.max_trials, replace=False)
#         X = X[indices]
#     print(f"ending with {len(X)} trials")

if args.reconstruct_queries:
    print("ok")
    X_queries = []
    for query in args.reconstruct_queries:
        X_queries.append(epochs[query].get_data())
    if args.equalize_events:
        nb_eve = min([len(x) for x in X_queries])
        print(f"keeping {nb_eve} trials in each condition ")
        indices = [np.random.choice(np.arange(nb_eve), nb_eve, replace=False) for x in X_queries]
        X_queries = [x[inds] for x, inds in zip(X_queries, indices)]
        X = np.concatenate(X_queries, 0)
    reconstruct_L2 = np.zeros((n_times, len(X_queries)))
# del epochs

n_trials, nchan, n_times = X.shape
print(f"n_trials,: {n_trials,} nchan,: {nchan,} n_times: {n_times}")
n_comp = np.min([nchan, n_trials]) # at most min(nchan, n_trials) components in the PCA
out_fn += f"_{n_comp}comps" # update out_fn
print(out_fn)

pca = PCA(n_comp)
scaler = StandardScaler()

all_PR = np.zeros(n_times)
for t in tqdm(range(n_times)):
    x = scaler.fit_transform(X[:,:,t])

    if args.micro_ave:
        x, _ = micro_averaging(x, np.ones(len(x)), args.micro_ave)
        if args.max_trials and len(x) > args.max_trials:
            indices = np.random.choice(np.arange(len(x)), args.max_trials, replace=False)
            x = x[indices]
        # print(f"ending with {len(x)} trials")
    pca.fit(x)
    all_PR[t] = participation_ratio(pca.explained_variance_)
    # print(all_PR[t])

    if args.reconstruct_queries:
        for q, X_query in enumerate(X_queries):
            PCAed = pca.transform(scaler.transform(X_query[:,:,t]))
            components = pca.components_ # n_comp, nchan
            for i_comp in range(n_comp):
                # reconstruction = pca.transform(X_query[:,:,t])
                reconstruction = np.dot(PCAed[:, 0:i_comp+1], components[0:i_comp+1])
                reconstruct_L2[t, q] += np.linalg.norm(X_query[:,:,t] - reconstruction) / n_comp


save_results(out_fn, all_PR, fn_end="PR")
if args.reconstruct_queries:
    save_results(out_fn, reconstruct_L2, fn_end="reconstruction_L2")

# from ipdb import set_trace; set_trace()