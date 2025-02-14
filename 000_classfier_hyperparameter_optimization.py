import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import mne
import numpy as np
import argparse
import pickle
import time
import importlib
import itertools
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
# parser.add_argument('--timegen', action='store_true', default=False, help='Whether to test probe trained at one time point also on all other timepoints')
parser.add_argument('--label', default='', help='help to identify the result latter')
parser.add_argument('--dummy', action='store_true', default=False, help='Accelerates everything so that we can test that the pipeline is working. Will not yield any interesting result!!')
parser.add_argument('--test_quality', action='store_true', default=False, help='Change the out dirname and save some scores, used for testing the quality of single runs.')
parser.add_argument('--filter', default=None, help='md query to filter trials before anything else (eg to use only matching trials')
parser.add_argument('-x', '--xdawn', action='store_true',  default=None, help='Whether to apply Xdawn spatial filtering before training decoder')
parser.add_argument('-a', '--autoreject', action='store_true',  default=None, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('--quality_th', default=None, type=float, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
# parser.add_argument('--equalize_split_events', action='store_true', default=None, help='subsample majority event classes IN EACH SPLIT QUERY to get same number of trials as the minority class')
parser.add_argument('-r', '--response_lock', action='store_true',  default=None, help='Whether to Use response locked epochs or classical stim-locked')
parser.add_argument('--micro_ave', default=None, type=int, help='Trial micro-averaging to boost decoding performance')
parser.add_argument('--null_prop', type=float,  default=0, help='Proportion of fixation period "null" trials')
# optionals, overwrite the config if passed
parser.add_argument('--sfreq', type=int, help='sampling frequency')
parser.add_argument('--n_folds', type=int, help='sampling frequency')
parser.add_argument('--train-cond', default='localizer', help='NOT USED HERE, Changing it will have no effect')
parser.add_argument('--train-conds', default=['localizer', 'one_object'], action='append', help='localizer, one_object or two_objects, any subset of the three')
parser.add_argument('--test-cond', default=['two_objects'], action='append', help='localizer, one_object or two_objects, should have the same length as test-queries')
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

np.random.seed(args.seed)
start_time = time.time()

### GET EPOCHS FILENAMES ###
out_dir_name = "Decoding_opti"
_, test_fns, out_fn, _ = get_paths(args, out_dir_name)
# adjust the out_fns 
train_cond_str = '_'.join(args.train_conds)
out_fn = out_fn.replace(f"{args.train_cond}", f"{train_cond_str}") # hacky but works. Replaces the "dummy" train_cond, that is not used, by the actual set of training conditions


###########################
######## TRAINING #########
###########################

param_grid = {
    "solver": ["saga", "liblinear"],  # Both solvers support L1 and L2
    "penalty": ["l1", "l2"],  # L1 = Lasso, L2 = Ridge
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Regularization strength
    "class_weight": [None, "balanced"]  # Compare default vs. balanced weighting
}
# Create all combinations of hyperparameters
param_combinations = list(itertools.product(param_grid['C'], param_grid['penalty'], param_grid['solver'], param_grid['class_weight']))

print('\nStarting training')

class_queries = get_class_queries("Property")

# train on loc and test on 1obj - old
# # train_fn, test_fns, _, _ = get_paths(args, out_dir_name)  # Get file paths
# train_epochs = load_data(args, train_fn)[0]
# train_epochs = train_epochs.crop(0.2, 0.2)
# X_train, y_train, groups, _, _ = get_X_y_from_queries(train_epochs, class_queries, args.split_queries)
# X_train = X_train.squeeze()

# # test data
# test_epochs = load_data(args, test_fns[0])[0]
# epoS1, epoC1 = test_epochs.copy(), test_epochs.copy()
# epoS1.metadata["Property"] = epoS1.metadata["Shape1"]
# epoC1.metadata["Property"] = epoC1.metadata["Colour1"]
# epoC1 = epoC1.shift_time(-0.6, relative=True) # Need to roll the times so that t0 is the color onset.
# block_epo = [epoS1, epoC1]
# for epo in block_epo: epo.baseline = None # hack, but works
# for epo in block_epo: epo = epo.crop(0.2, 0.2)
# test_epochs = mne.concatenate_epochs(block_epo)
# X_test, y_test, groups, _, _ = get_X_y_from_queries(test_epochs, class_queries, args.split_queries)
# X_test = X_test.squeeze()

all_train_epochs = []
for cond in args.train_conds:
    args.train_cond = cond  # Set current condition
    train_fn, _, _, _ = get_paths(args, out_dir_name)  # Get file paths    
    epochs = load_data(args, train_fn)[0]

    if cond == "localizer":
        epochs.metadata["Property"] = epochs.metadata["Loc_word"].str.replace(r"^img_", "", regex=True)
        block_epo = [epochs]
    elif cond == "one_object": # get separate epochs for shape and colors, we'll use both
        epoS1, epoC1 = epochs.copy(), epochs.copy()
        epoS1.metadata["Property"] = epoS1.metadata["Shape1"]
        epoC1.metadata["Property"] = epoC1.metadata["Colour1"]
        epoC1 = epoC1.shift_time(-0.6, relative=True) # Need to roll the times so that t0 is the color onset.
        block_epo = [epoS1, epoC1]
    all_train_epochs.extend(block_epo)

for epo in all_train_epochs: epo.baseline = None # hack, but works
for epo in all_train_epochs: epo = epo.crop(0.2, 0.2)
train_epochs = mne.concatenate_epochs(all_train_epochs)
X_train, y_train, groups, _, _ = get_X_y_from_queries(train_epochs, class_queries, args.split_queries)
X_train = X_train.squeeze()

# test data -- two_objects
test_epochs = load_data(args, test_fns[0])[0]
epoS1, epoC1 = test_epochs.copy(), test_epochs.copy()
epoS1.metadata["Property"] = epoS1.metadata["Shape1"]
epoC1.metadata["Property"] = epoC1.metadata["Colour1"]
epoC1 = epoC1.shift_time(-0.6, relative=True) # Need to roll the times so that t0 is the color onset.
epoS2, epoC2 = test_epochs.copy(), test_epochs.copy()
epoS2.metadata["Property"] = epoS2.metadata["Shape2"]
epoC2.metadata["Property"] = epoC2.metadata["Colour2"]
epoS2 = epoS2.shift_time(-1.8, relative=True)
epoC2 = epoC2.shift_time(-2.4, relative=True)
all_test_epochs = [epoS1, epoC1, epoS2, epoC2]
for epo in all_test_epochs: epo.baseline = None # hack, but works
for epo in all_test_epochs: epo = epo.crop(0.2, 0.2)
test_epochs = mne.concatenate_epochs(all_test_epochs)
X_test, y_test, groups, _, _ = get_X_y_from_queries(test_epochs, class_queries, args.split_queries)
X_test = X_test.squeeze()


### DECODE ###
print(f'\nStarting training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')
# Iterate over each hyperparameter combination and train the model
sub_perfs, sub_params = [], []
for C, penalty, solver, class_weight in tqdm(param_combinations):
    clf = LogisticRegression(class_weight=class_weight, max_iter=10000, C=C, penalty=penalty, solver=solver)
    clf = OneVsRestClassifier(clf, n_jobs=1)
    pipeline = make_pipeline(RobustScaler(), clf)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')

    sub_perfs.append(test_auc)
    sub_params.append([C, penalty, solver, class_weight])

pickle.dump(sub_perfs, open(f"{out_fn}_sub_perfs_{args.subject}.pkl", "wb"))
pickle.dump(sub_params, open(f"{out_fn}_sub_params_{args.subject}.pkl", "wb"))

print(f'Done with training. Elasped time since the script began: {(time.time()-start_time)/60:.2f}min')

print(f"Best perf: {np.max(sub_perfs)}; params: {sub_params[np.argmax(sub_perfs)]}")

