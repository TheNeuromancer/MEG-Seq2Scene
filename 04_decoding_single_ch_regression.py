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

# local imports
from utils.decod import *

parser = argparse.ArgumentParser(description='MEG Decoding analysis')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='theo',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle sentence labels before training')
parser.add_argument('--freq-band', default='', help='name of frequency band to use for filtering (theta, alpha, beta, gamma)')
parser.add_argument('--label', default='', help='help to identify the result latter')
parser.add_argument('--dummy', action='store_true', default=False, help='Accelerates everything so that we can test that the pipeline is working. Will not yield any interesting result!!')
parser.add_argument('-x', '--xdawn', action='store_true',  default=None, help='Whether to apply Xdawn spatial filtering before training decoder')
parser.add_argument('-a', '--autoreject', action='store_true',  default=None, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('--filter', default='', help='md query to filter trials before anything else (eg to use only matching trials')
parser.add_argument('--train-cond', default='two_objects', help='localizer, one_object or two_objects')
parser.add_argument('--train-query', help='Metadata query for training classes')
parser.add_argument('--test-cond', default=[], action='append', help='localizer, one_object or two_objects, should have the same length as test-queries')
parser.add_argument('--test-query', default=[], action='append', help='Metadata query for testing classes')
parser.add_argument('--quality_th', default=None, type=float, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('--micro_ave', default=None, type=int, help='Trial micro-averaging to boost decoding performance')

parser.add_argument('--windows', default=[], action='append', help='tmin and tmax to crop the epochs, one for each train and test cond')

# optionals, overwrite the config if passed
parser.add_argument('--sfreq', type=int, help='sampling frequency')
parser.add_argument('--n_folds', type=int, help='sampling frequency')

# not used, kept for consistency
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
parser.add_argument('--timegen', action='store_true', default=False, help='Whether to test probe trained at one time point also on all other timepoints')
parser.add_argument('--t1', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--t2', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--equalize_events', action='store_true', default=False, help='subsample majority event classes to get same number of trials as the minority class')
parser.add_argument('-r', '--response_lock', action='store_true',  default=None, help='Whether to Use response locked epochs or classical stim-locked')

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
# print(args)
# print("matplotlib: ", matplotlib.__version__)
# print("mne: ", mne.__version__)

if len(args.test_cond) != len(args.test_query):
    raise RuntimeError("Test conditions and test-queries should have the same length")

np.random.seed(args.seed)
start_time = time.time()

###########################
######## TRAINING #########
###########################

### GET EPOCHS FILENAMES ###
out_dir_name = "Regression_decoding_single_ch"
train_fn, test_fns, out_fn, test_out_fns = get_paths(args, out_dir_name)

## GET QUERIES
class_queries = get_class_queries(args.train_query)

if args.dummy:
    clf = LinearRegression(n_jobs=-1)
    setattr(args, 'n_folds', 2)
else:
    # clf = Ridge(class_weight='balanced')
    clf = RidgeCV(alphas=np.logspace(-4, 4, 9), cv=5)
# clf = mne.decoding.LinearModel(clf)
pipeline = make_pipeline(RobustScaler(), clf)

# print(args.windows)
epochs = load_data(args, train_fn)[0]
args.windows = [w.replace(" ", "") for w in args.windows] # remove spaces
windows = [tuple([float(x) for x in win.split(",")]) for win in args.windows]
all_Rs = {}
for window in windows:
    print(f'\nStarting training for window {window}')
    cropped_epochs = deepcopy(epochs).crop(*window)
    X, y, groups, test_split_query_indices = get_X_y_from_queries(cropped_epochs, class_queries, args.split_queries)
    X = X.mean(2) # average time points in the window

    all_Rs[str(window)] = []
    for ch in range(X.shape[1]):
        pipeline.fit(X[:,ch, np.newaxis], y)
        # all_Rs[str(window)].append(pipeline[1].filters_[0])
        all_Rs[str(window)].append(pipeline[1].coef_[0])
        # from ipdb import set_trace; set_trace()

    all_Rs[str(window)] = np.array(all_Rs[str(window)])

    ### PLOT PERFORMANCE ###
    plot_single_ch_perf(all_Rs[str(window)], epochs.info, f"{out_fn}_{str(window)}_R.png", vmin=-.1, vcenter=0, vmax=.1, score_label="R")

    print(f"mean R: {np.mean(all_Rs[str(window)])}")
    print(f"max R: {np.max(all_Rs[str(window)])}")


### SAVE RESULTS ###
pickle.dump(all_Rs, open(f"{out_fn}_all_windows_R.p", "wb"))
# save_results(out_fn, all_Rs) #, all_models)
# save_results(out_fn, accuracy, fn_end="acc")
# save_patterns(args, out_fn, all_models)

## Save epochs info
pickle.dump(epochs.info, open(f"{out_fn}_epo_info.p", "wb"))



print(f'Total elasped time since the script began: {(time.time()-start_time)/60:.2f}min')