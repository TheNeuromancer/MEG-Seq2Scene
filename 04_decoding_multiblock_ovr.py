import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import mne
import numpy as np
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
parser.add_argument('-x', '--xdawn', action='store_true',  default=None, help='Whether to apply Xdawn spatial filtering before training decoder')
parser.add_argument('-a', '--autoreject', action='store_true',  default=None, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('--quality_th', default=None, type=float, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
parser.add_argument('--equalize_split_events', action='store_true', default=None, help='subsample majority event classes IN EACH SPLIT QUERY to get same number of trials as the minority class')
parser.add_argument('-r', '--response_lock', action='store_true',  default=None, help='Whether to Use response locked epochs or classical stim-locked')
parser.add_argument('--micro_ave', default=None, type=int, help='Trial micro-averaging to boost decoding performance')
# parser.add_argument('--add_null', action='store_true',  default=False, help='Whether to add fixation period "null" trials')
parser.add_argument('--null_prop', type=float,  default=0, help='Proportion of fixation period "null" trials')

parser.add_argument('--train-cond', default='localizer', help='NOT USED HERE, Changing it will have no effect')
parser.add_argument('--train-conds', default=[], action='append', help='localizer, one_object or two_objects, any subset of the three')
parser.add_argument('--train-query', help='Metadata query for training classes')
parser.add_argument('--test-cond', default=[], action='append', help='localizer, one_object or two_objects, should have the same length as test-queries')
parser.add_argument('--test-query', default=[], action='append', help='Metadata query for testing classes')
parser.add_argument('--windows', default=[], action='append', help='tmin and tmax to crop the epochs, one for each train and test cond')
parser.add_argument('--split_props', action='store_true', default=False, help='Separate each property instead of training a single OVR for shapes, images and relations')

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
if args.null_prop > 0 and args.equalize_events:
    raise RuntimeError("Cannot add null trials AND equalize events.")
if args.null_prop > 0 and not (len(args.windows) == 0 or args.windows[0][0] != args.windows[0][1]):
    raise RuntimeError("Cannot add null trials for multiple timepoints decoding. Only for a single decoder. Then you would have to add these trials inside the decoding loop.")

np.random.seed(args.seed)
start_time = time.time()


### GET EPOCHS FILENAMES ###
out_dir_name = "Decoding_multi"
_, test_fns, out_fn, test_out_fns = get_paths(args, out_dir_name)
# adjust the out_fns 
train_cond_str = '_'.join(natsorted(args.train_conds))
out_fn = out_fn.replace(f"{args.train_cond}", f"{train_cond_str}") # hacky but works. Replaces the "dummy" train_cond, that is not used, by the actual set of training conditions
print(out_fn)
## !! Would not work if you test on the localier ... but probably that won't happen
test_out_fns = [fn.replace(f"{args.train_cond}", f"{train_cond_str}") for fn in test_out_fns]
print(test_out_fns)

if args.windows:
    args.windows = [w.replace(" ", "") for w in args.windows] # remove spaces
    wins = [f"#{'#'.join([args.windows[0], w])}#" for w in args.windows] # string to add to the out fns
    for i, win in enumerate(wins): # fix the string, we need zeros before and after commas if only one digit
            wins[i] = wins[i].replace(",.", ",0.")
            wins[i] = wins[i].replace("#.", "#0.")
            wins[i] = wins[i].replace(".,", ".0,")
    out_fn += wins[0]
    for i in range(len(test_out_fns)): test_out_fns[i] += wins[i+1]

###########################
######## TRAINING #########
###########################

print('\nStarting training')

### LOAD MULTIPLE BLOCK TYPES ###
all_epochs = []  # To store all epochs before merging

for cond in args.train_conds:
    args.train_cond = cond  # Set current condition
    train_fn, _, _, _ = get_paths(args, out_dir_name)  # Get file paths
    
    epochs = load_data(args, train_fn)[0]

    # Complement the md to get query-compatibility
    if cond == "localizer":
        epochs.metadata["Property"] = epochs.metadata["Loc_word"].str.replace(r"^img_", "", regex=True)
        block_epo = [epochs]
    elif cond == "one_object": # get separate epochs for shape and colors, we'll use both
        epoS1, epoC1 = epochs.copy(), epochs.copy()
        epoS1.metadata["Property"] = epoS1.metadata["Shape1"]
        epoC1.metadata["Property"] = epoC1.metadata["Colour1"]
        epoC1 = epoC1.shift_time(-0.6, relative=True) # Need to roll the times so that t0 is the color onset.
        block_epo = [epoS1, epoC1]
    elif cond == "two_objects":
        epochs_orig = epochs.copy() # keep a copy for the null trials
        epoS1, epoC1 = epochs.copy(), epochs.copy()
        epoS1.metadata["Property"] = epoS1.metadata["Shape1"]
        epoC1.metadata["Property"] = epoC1.metadata["Colour1"]
        epoC1 = epoC1.shift_time(-0.6, relative=True) # Need to roll the times so that t0 is the color onset.
        epoR = epochs.copy()
        epoR.metadata["Property"] = epoR.metadata["Relation"]
        epoR = epoR.shift_time(-1.2, relative=True)
        epoS2, epoC2 = epochs.copy(), epochs.copy()
        epoS2.metadata["Property"] = epoS2.metadata["Shape2"]
        epoC2.metadata["Property"] = epoC2.metadata["Colour2"]
        epoS2 = epoS2.shift_time(-1.8, relative=True)
        epoC2 = epoC2.shift_time(-2.4, relative=True)
        block_epo = [epoS1, epoC1, epoR, epoS2, epoC2]
    else:
        raise RuntimeError(f"Condition {cond} not recognized")

    for epo in block_epo: epo.baseline = None # hack, but works (else concat does not work)
    
    # Crop to the windows (tested for single time point window only)
    windows = [tuple([float(x) for x in win.split(",")]) for win in args.windows]
    if windows:
        print(f"Using training time window: {windows[0]}s")
        for epo in block_epo: epo = epo.crop(*windows[0])

    all_epochs.extend(block_epo)
    
    for epo in block_epo: # print event counts
        print(f"\nLoaded {len(epo)} trials from {cond}") # Print trial count
        trial_counts = epo.metadata["Property"].value_counts().to_dict()
        print(trial_counts) 

epochs = mne.concatenate_epochs(all_epochs)  # Merge all epochs
del all_epochs  # Free up memory
train_tmin, train_tmax = epochs.tmin, epochs.tmax
print(train_tmin, train_tmax)

# all_class_queries = [get_class_queries(query) for query in args.train_query]
class_queries = get_class_queries(args.train_query)
n_times = len(epochs.times)


### GET DATA FROM THE FIXATION PERIOD
if args.null_prop > 0: 
    num_null = 10 # number of null trial per trial. 
    epochs_null = epochs_orig.crop(epochs_orig.tmin, 0)
    dat_null = epochs_null.get_data(picks='meg').squeeze()
    n_trials, n_chans, n_times_fixation = dat_null.shape
    random_indices = np.random.choice(n_times_fixation, size=(len(epochs_orig), num_null), replace=True)
    # Use advanced indexing to extract the selected time points
    dat_null = dat_null[np.arange(n_trials)[:, None, None], np.arange(n_chans)[None, :, None], random_indices[:, None, :]]
    dat_null = dat_null.transpose(0, 2, 1).reshape(n_trials * num_null, n_chans)
    # random_indices = np.random.choice(n_times_fixation, size=n_trials, replace=True)
    # dat_null = dat_null[np.arange(n_trials), :, random_indices]  # Shape (n_trials, n_chans)
    out_fn += f"_null{args.null_prop}"
else:
    dat_null = None
del epochs_orig

### DECODE ###
if args.dummy:
    clf = LinearRegression(n_jobs=-1)
    setattr(args, 'n_folds', 2)
else:
    # clf = LogisticRegression(C=1/0.006, solver='saga', class_weight='balanced', multi_class='auto', max_iter=1000000)
    # hyperparam optim found: [0.1, 'l1', 'liblinear', 'balanced']
    clf = LogisticRegression(C=0.1, penalty='l1', solver='saga', class_weight='balanced', multi_class='auto', max_iter=10000)
    # clf = SVC(kernel='rbf', class_weight='balanced', max_iter=-1, C=1, gamma=0.001, probability=True, random_state=42)
clf = OneVsRestClassifier(clf, n_jobs=1)

print(f'\nStarting training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')
if args.windows and args.windows[0].split(',')[0] == args.windows[0].split(',')[1]: # single time point decoding
    if args.split_props: # train a separte OVR for each property
        all_props_models, patterns, filters = [], [], []
        for props in [shapes, colors, relations]:
            epo_prop = epochs[f"Property in {props}"]
            query_prop = [q for q in class_queries if any([prop in q for prop in props])]
            if not query_prop: continue # If we don't train a relation decoder, this will be empty
            prop_models, prop_patterns, prop_filters, prop_mds = decode_ovr_single_tp(args, clf, epo_prop, query_prop, dat_null)
            all_props_models.append(deepcopy(prop_models))
            patterns.append(prop_patterns)
            filters.append(prop_filters)
        patterns = np.concatenate([np.atleast_2d(p) for p in patterns]) # Relation is not a true OVR, so single set of weights ... 
        filters = np.concatenate([np.atleast_2d(f) for f in filters])

    else: # OVR with all properties
        all_models, patterns, filters, mds = decode_ovr_single_tp(args, clf, epochs, class_queries, dat_null)
    save_results(out_fn, patterns, fn_end="patterns", time=False) # , mds=mds
    save_results(out_fn, filters, fn_end="filters", time=False)
else:
    print("This script was only made for single timepoint decoding. Use the args.windows argument")
    raise RuntimeError

print(f'Done with training. Elasped time since the script began: {(time.time()-start_time)/60:.2f}min')


###########################
######### TESTING #########
###########################

print('\n\nStarting testing')

### GET TEST DATA ###
for i_test, (cond, query, test_fn, test_out_fn) in enumerate(zip(args.test_cond, args.test_query, test_fns, test_out_fns)):
    print(f'testing on {cond}, output path: {test_out_fn}')

    ### LOAD EPOCHS ###
    epochs = load_data(args, test_fn)[0]
    if windows: 
        print(f"Using test time window: {windows[i_test+1]}s")
        epochs = epochs.crop(*windows[i_test+1]) # first window is for training
    test_tmin, test_tmax = epochs.tmin, epochs.tmax

    if args.split_props:
        preds_all_props = [] 
        for props, prop_models in zip([shapes, colors, relations], all_props_models):
            preds_prop, mds = test_decode_ovr_single_tp(args, epochs, prop_models)
            preds_all_props.append(preds_prop)
        preds = np.concatenate(preds_all_props, 2)

    else:
        preds, mds = test_decode_ovr_single_tp(args, epochs, all_models)
    
    ### SAVE RESULTS ###
    if args.windows and args.windows[0].split(',')[0] == args.windows[0].split(',')[1]: # single time point decoding
        # add a trial id to the md to help identification later on.
        mds['trial_id'] = mds['run_nb'].astype(str) + "_" + mds.index.astype(str)
        save_results(test_out_fn, preds, fn_end="preds", mds=mds)

print(f'Total elasped time since the script began: {(time.time()-start_time)/60:.2f}min')