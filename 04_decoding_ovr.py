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

parser = argparse.ArgumentParser(description='MEG Decoding analysis')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='theo',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle sentence labels before training')
parser.add_argument('--freq-band', default='', help='name of frequency band to use for filtering (theta, alpha, beta, gamma)')
parser.add_argument('--timegen', action='store_true', default=False, help='Whether to test probe trained at one time point also on all other timepoints')
parser.add_argument('--label', default='', help='help to identify the result latter')
parser.add_argument('--dummy', action='store_true', default=False, help='Accelerates everything so that we can test that the pipeline is working. Will not yield any interesting result!!')
parser.add_argument('--test_quality', action='store_true', default=False, help='Change the out dirname and save some scores, used for testing the quality of single runs.')
parser.add_argument('--filter', default=None, help='md query to filter trials before anything else (eg to use only matching trials')
parser.add_argument('--train-cond', default='localizer', help='localizer, one_object or two_objects')
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

if len(args.test_cond) != len(args.test_query):
    raise RuntimeError("Test conditions and test-queries should have the same length")

np.random.seed(args.seed)
start_time = time.time()

###########################
######## TRAINING #########
###########################

### GET EPOCHS FILENAMES ###
out_dir_name = "Decoding_ovr" if not args.test_quality else "Decoding_test_quality"
train_fn, test_fns, out_fn, test_out_fns = get_paths(args, out_dir_name)

if args.windows:
    args.windows = [w.replace(" ", "") for w in args.windows] # remove spaces
    wins = [f"#{'#'.join([args.windows[0], w])}#" for w in args.windows] # string to add to the out fns
    out_fn += wins[0]
    for i in range(len(test_out_fns)): test_out_fns[i] += wins[i+1]

# if not args.overwrite:

print('\nStarting training')
### LOAD EPOCHS ###
if args.response_lock:
    epochs = load_data(args, train_fn, crop_final=False)[0]
    epochs = to_response_lock_epochs(epochs, args.train_cond)
else:
    epochs = load_data(args, train_fn)[0]
windows = [tuple([float(x) for x in win.split(",")]) for win in args.windows]
if windows: epochs = epochs.crop(*windows[0])
train_tmin, train_tmax = epochs.tmin, epochs.tmax

## GET QUERIES
class_queries = get_class_queries(args.train_query)
n_times = len(epochs.times)

if args.dummy: # speed everything up for a dummy run
    clf = LinearRegression(n_jobs=-1)
    setattr(args, 'n_folds', 2)
else:
    clf_cv = StratifiedShuffleSplit(10, random_state=42) # help avoid warnings when there are very few trials in one class
    clf = LogisticRegressionCV(Cs=10, solver='liblinear', class_weight='balanced', multi_class='auto', n_jobs=-1, cv=clf_cv, max_iter=10000)
    # clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 9), cv=clf_cv, class_weight='balanced')
    # clf = RidgeClassifierCVwithProba(alphas=np.logspace(-4, 4, 9), cv=5, class_weight='balanced')
    # clf = GridSearchCV(clf, {"kernel":('linear', 'rbf', 'poly'), "C":np.logspace(-2, 4, 7)})
clf = OneVsRestClassifier(clf, n_jobs=1)
# clf = mne.decoding.LinearModel(clf)

### DECODE ###
print(f'\nStarting training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')
AUC, _, all_models, AUC_query = decode_ovr(args, clf, epochs, class_queries)
print(f'Finished training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min\n')

if args.test_quality: # save explicit score values, then exit
    quality_dir = f"{op.dirname(op.dirname(op.dirname(out_fn)))}/Quality_test"
    if not op.exists(quality_dir):
        os.makedirs(quality_dir)
    mean_AUC = AUC.mean()
    max_AUC = AUC.max()
    min_AUC = AUC.min()
    std_AUC = AUC.std()
    quality_fn = f"{op.basename(op.dirname(out_fn))}_{op.basename(out_fn)}_min{min_AUC:.3f}_mean{mean_AUC:.3f}_std{std_AUC:.3f}_max{max_AUC:.3f}.txt"
    with open(f"{quality_dir}/{quality_fn}", 'w') as f: # also save it the file just in case
        f.write(f"mean = {mean_AUC:3f}")
        f.write(f"max = {max_AUC:3f}")
        f.write(f"min = {min_AUC:3f}")
        f.write(f"std = {std_AUC:3f}")
    exit()

### SAVE RESULTS ###
save_results(out_fn, AUC) #, all_models)
# save_results(out_fn, accuracy, fn_end="acc")
# save_patterns(args, out_fn, all_models)
# save_best_pattern(out_fn, AUC, all_models) ## Save best model's pattern
### PLOT PERFORMANCE ###
plot_perf(args, out_fn, AUC, args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, \
          test_tmin=train_tmin, test_tmax=train_tmax, version=version)
if AUC_query is not None: # save the results for all the splits
    for i_query, query in enumerate(args.split_queries):
        if np.all(np.isnan(AUC_query[:,:,i_query])): continue # do not save if we only have nans (happens when the split query doesn't work for this subject, eg flash for the first subjects)
        query = '_'.join(query.split()) # replace spaces by underscores
        query = shorten_filename(query) # shorten string by removing unnecessary stuff
        save_results(out_fn+f'_for_{query}', AUC_query[:,:,i_query])
        plot_perf(args, out_fn+f'_for_{query}', AUC_query[:,:,i_query], args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=train_tmin, test_tmax=train_tmax, version=version)
        # save_preds(args, out_fn+f'_for_{query}', mean_preds_query[:,:,i_query])

# from ipdb import set_trace; set_trace()
print(f'Done with saving training plots and data. Elasped time since the script began: {(time.time()-start_time)/60:.2f}min')


###########################
######### TESTING #########
###########################

print('\n\nStarting testing')

### GET TEST DATA ###
for i_test, (cond, query, test_fn, test_out_fn) in enumerate(zip(args.test_cond, args.test_query, test_fns, test_out_fns)):
    print(f'testing on {cond}, output path: {test_out_fn}')

    ### LOAD EPOCHS ###
    epochs = load_data(args, test_fn)[0]
    if windows: epochs = epochs.crop(*windows[i_test+1]) # first window is for training
    test_tmin, test_tmax = epochs.tmin, epochs.tmax

    class_queries = get_class_queries(query)

    AUC, _, AUC_query = test_decode_ovr(args, epochs, class_queries, all_models)

    ### SAVE RESULTS ###
    save_results(test_out_fn, AUC)
    # save_results(test_out_fn, accuracy, fn_end="acc")
    ### PLOT PERFORMANCE ###
    plot_perf(args, test_out_fn, AUC, args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, \
              test_tmin=test_tmin, test_tmax=test_tmax, gen_cond=cond, version=version)
    if AUC_query is not None: # save the results for all the splits
        for i_query, query in enumerate(args.split_queries):
            query = '_'.join(query.split()) # replace spaces by underscores
            query = shorten_filename(query) # shorten string by removing unnecessary stuff
            save_results(test_out_fn+f'_for_{query}', AUC_query[:,:,i_query])
            plot_perf(args, test_out_fn+f'_for_{query}', AUC_query[:,:,i_query], args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, test_tmax=test_tmax, version=version, gen_cond=cond)
            # save_preds(args, test_out_fn+f'_for_{query}', mean_preds_query[:,:,i_query])

print(f'Total elasped time since the script began: {(time.time()-start_time)/60:.2f}min')