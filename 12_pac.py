import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import mne
import numpy as np
# from ipdb import set_trace
import argparse
import pickle
import time
import importlib
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning) # ignore all future warnings

# local imports
from utils.decod import *
from utils.pac import *

parser = argparse.ArgumentParser(description='MEG Decoding analysis')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='01',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle sentence labels before training')
parser.add_argument('--freq-band', default='', help='name of frequency band to use for filtering (theta, alpha, beta, gamma)')
parser.add_argument('--timegen', action='store_true', default=False, help='Whether to test probe trained at one time point also on all other timepoints')
parser.add_argument('--label', default='', help='help to identify the result latter')
parser.add_argument('--dummy', action='store_true', default=False, help='Accelerates everything so that we can test that the pipeline is working. Will not yield any interesting result!!')
parser.add_argument('--test_quality', action='store_true', default=False, help='Change the out dirname and save some scores, used for testing the quality of single runs.')
parser.add_argument('--filter', default=None, help='md query to filter trials before anything else (eg to use only matching trials')
parser.add_argument('--train-cond', default='two_objects', help='localizer, one_object or two_objects')
parser.add_argument('--train-query', help='Metadata query for training classes')
parser.add_argument('--test-cond', default=[], action='append', help='localizer, one_object or two_objects, should have the same length as test-queries')
parser.add_argument('--test-query', default=[], action='append', help='Metadata query for testing classes')
parser.add_argument('--windows', default=[], action='append', help='tmin and tmax to crop the epochs, one for each train and test cond')
parser.add_argument('-x', '--xdawn', action='store_true',  default=None, help='Whether to apply Xdawn spatial filtering before training decoder')
parser.add_argument('-a', '--autoreject', action='store_true',  default=None, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('--quality_th', default=None, type=float, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
parser.add_argument('--equalize_split_events', action='store_true', default=None, help='subsample majority event classes IN EACH SPLIT QUERY to get same number of trials as the minority class')
parser.add_argument('-r', '--response_lock', action='store_true',  default=None, help='Whether to Use response locked epochs or classical stim-locked')
parser.add_argument('--micro_ave', default=None, type=int, help='Trial micro-averaging to boost decoding performance')

# optionals, overwrite the config if passed
parser.add_argument('--sfreq', type=int, help='sampling frequency')
parser.add_argument('--n_folds', type=int, help='sampling frequency')

# not used, kept for consistency
parser.add_argument('--t1', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--t2', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--equalize_events', action='store_true', default=False, help='subsample majority event classes to get same number of trials as the minority class')

# not implemented
parser.add_argument('--localizer', action='store_true', default=False, help='Whether to use only electrode that were significant in the localizer')
parser.add_argument('--auc_thresh', default=0.55, type=float, help='pvalue threshold under which a channel is kept for the localizer')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
for arg in vars(args): # update config with arguments from the argparse 
    if getattr(args, arg) is not None: # !! this is important, we can have only "None" as the default argument, else it will overwrite the config everytime !!
        setattr(config, arg, getattr(args, arg))
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)
print("matplotlib: ", matplotlib.__version__)
print("mne: ", mne.__version__)
version = "v1" if int(args.subject[0:2]) < 8 else "v2"

np.random.seed(args.seed)
start_time = time.time()

###########################
######## TRAINING #########
###########################

### GET EPOCHS FILENAMES ###
out_dir_name = "PAC"
train_fn, test_fns, out_fn, test_out_fns = get_paths(args, out_dir_name)

if args.windows:
    args.windows = [w.replace(" ", "") for w in args.windows] # remove spaces
    wins = [f"#{'#'.join([args.windows[0], w])}#" for w in args.windows] # string to add to the out fns
    for i, win in enumerate(wins): # fix the string, we need zeros before and after commas if only one digit
            wins[i] = wins[i].replace(",.", ",0.")
            wins[i] = wins[i].replace("#.", "#0.")
            wins[i] = wins[i].replace(".,", ".0,")
    out_fn += wins[0]
    for i in range(len(test_out_fns)): test_out_fns[i] += wins[i+1]

print('\nStarting training')
### LOAD EPOCHS ###
epochs = load_data(args, train_fn)[0]
windows = [tuple([float(x) for x in win.split(",")]) for win in args.windows]
if windows: 
    print(f"Using training time window: {windows[0]}s")
    epochs = epochs.crop(*windows[0])
train_tmin, train_tmax = epochs.tmin, epochs.tmax
print(train_tmin, train_tmax)


## GET QUERIES
class_queries = get_class_queries(args.train_query)

### PAC ###
print(f'\nStarting PAC. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')

for query in class_queries:
    epochs_query = epochs[query]
    print(f'Query: {query}, shape: {epochs_query.get_data().shape}')
    pac_values, zscores, tvals = compute_pac(epochs_query, picks='all', gamma_range=(70, 100), theta_range=(3, 8), n_theta_bins=18, n_permutations=100, average_trials=False)
    # shape n_sensors * n_trials
    from ipdb import set_trace; set_trace()

    print(f"Query: {query}, mean PAC: {pac_values.mean()}")
    print(f"Query: {query}, max channel, mean PAC over trials: {pac_values.mean(1).max()}")

    save_results(out_fn, pac_values, fn_end=f"{query}_pac", time=False, mds=epochs_query.metadata)

print(f'Total elasped time since the script began: {(time.time()-start_time)/60:.2f}min')