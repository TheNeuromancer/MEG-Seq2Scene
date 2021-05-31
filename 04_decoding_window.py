import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import mne
import numpy as np
from ipdb import set_trace
import argparse
import pickle
import time
import importlib
from sklearn.metrics.pairwise import cosine_similarity

from utils.decod import *
from utils.RSA import load_data_rsa, get_paths_rsa, decode_ovr, test_decode_ovr, decode_window, angle_between ## TO CHANGE

parser = argparse.ArgumentParser(description='MEG Decoding analysis')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='theo',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle sentence labels before training')
parser.add_argument('--freq-band', default='', help='name of frequency band to use for filtering (theta, alpha, beta, gamma)')
parser.add_argument('--train-cond', default=[], action='append', help='localizer, one_object or two_objects')
parser.add_argument('--train-query', default=[], action='append', help='Metadata query for training classes')
parser.add_argument('--label', default='', help='help to identify the result latter')
parser.add_argument('--dummy', action='store_true', default=False, help='Accelerates everything so that we can test that the pipeline is working. Will not yield any interesting result!!')
parser.add_argument('-x', '--xdawn', action='store_true',  default=False, help='Whether to apply Xdawn spatial filtering before training decoder')
parser.add_argument('--filter', default='', help='md query to filter trials before anything else (eg to use only matching trials')

parser.add_argument('--windows', action='append', default=[], help='list of time windows to train classifiers, test generalization and compute angles')

# optionals, overwrite the config if passed
parser.add_argument('--sfreq', type=int, help='sampling frequency')
parser.add_argument('--n_folds', type=int, help='sampling frequency')

# useless here, kept for consistency
parser.add_argument('--test-cond', default=[], action='append', help='localizer, one_object or two_objects, should have the same length as test-queries')
parser.add_argument('--test-query', default=[], action='append', help='Metadata query for testing classes')
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
parser.add_argument('--t1', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--t2', action='append', default=[], help="Metadata query for generalization test")

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

if len(args.test_cond) != len(args.test_query):
    raise RuntimeError("Test conditions and test-queries should have the same length")

np.random.seed(args.seed)
start_time = time.time()

### GET EPOCHS FILENAMES ###
print("TODO: make own utils for this script ... here we do not have the test condition saved in the fn AND WE NEED TO BE ABLE TO LOAD DIFFERENT BLOCK TYPES TO COMPARE THEM")
train_fn, base_out_fn = get_paths_rsa(args, "Decoding_window")
out_fn = f"{base_out_fn}"
print(f"full out fn: {out_fn}")
clf_idx = 2 if args.reduc_dim_window else 1
colors = ["vert", "bleu", "rouge"]
shapes = ["triangle", "cercle", "carre"]


###########################
######## TRAINING #########
###########################

### DECODE ###
all_windows_models = []
all_hyperplans = {}
print(f'\nStarting training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')
for window_str, query, cond in zip(args.windows, args.train_query, args.train_cond):
    window = tuple([float(x) for x in window_str.split(",")])
    all_hyperplans[window] = {}
    print(f"Doing window : {window}")

    ### LOAD EPOCHS ###
    epochs = load_data_rsa(args, train_fn)
    train_tmin, train_tmax = epochs[0].tmin, epochs[0].tmax
    if args.train_cond == "two_objects":
        epochs.metadata = complement_md(epochs.metadata)
    epo_window = epochs.copy().crop(*window)
    
    ## get queries
    print("hardcoded queries ... prety bad")
    if "Left" in args.train_query or "Right" in args.train_query:
        class_queries = [f"{args.train_query}=='{s}_{c}'" for s in shapes for c in colors]
    elif args.train_query == "Shape1+Colour1":
        class_queries = [f"Shape1=='{s}' and Colour1=='{c}'" for s in shapes for c in colors]
    elif args.train_query == "Shape2+Colour2":
        class_queries = [f"Shape2=='{s}' and Colour2=='{c}'" for s in shapes for c in colors]
    else:
        raise RuntimeError(f"Wrong query: {args.train_query}")
    print(class_queries)

    ## DECODE
    AUC, accuracy, ovr_model = decode_window(args, epo_window, class_queries)
    all_windows_models.append(ovr_model)
    
    
    # inter-fold consistency
    for i_class, classe in enumerate(class_queries):
        print(f"\n doing class {classe}")
        all_angles = []
        all_cos = []
        all_hyperplans[window][classe] = []
        for i1 in range(args.n_folds):
            all_hyperplans[window][classe].append(ovr_model[i1][clf_idx].estimators_[i_class].coef_.ravel())
            for i2 in range(i1, args.n_folds):
                a1, a2 = ovr_model[i1][clf_idx].estimators_[i_class].coef_.ravel(), ovr_model[i2][clf_idx].estimators_[i_class].coef_.ravel()
                all_angles.append(np.degrees(angle_between(a1, a2)))
                all_cos.append(cosine_similarity(a1[np.newaxis,:], a2[np.newaxis,:])[0][0])
        print(f"inter folds consistency for query: {classe}: {np.mean(all_angles)} +/- {np.std(all_cos)} degrees and {np.mean(all_cos)} +/- {np.std(all_cos)}")

    # inter-condition angles
    all_cond_hyperplans = []
    for i_class, classe in enumerate(class_queries):
        all_cond_hyperplans.append(np.mean([ovr_model[i][clf_idx].estimators_[i_class].coef_.ravel() for i in range(args.n_folds)], 0))
            
    for i_c1, classe1 in enumerate(class_queries):
        for i_c2, classe2 in enumerate(class_queries):
            a1, a2 = all_cond_hyperplans[i_c1], all_cond_hyperplans[i_c2]
            print(f"\n doing classes {classe1} vs {classe2}: {np.degrees(angle_between(a1, a2))} degrees")
            print(f"and cosine_similarity: {cosine_similarity(a1[np.newaxis,:], a2[np.newaxis,:])[0][0]}")

print(f'Finished training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min\n')

set_trace()

## ACROSS WINDOWS 
all_hyperplans_all_windows = []
for i_win, (window, window_models) in enumerate(zip(args.windows, all_windows_models)):
    all_hyperplans_all_windows.append([])
    for i_class, classe in enumerate(class_queries):
        all_hyperplans_all_windows[-1].append(np.mean([window_models[i][clf_idx].estimators_[i_class].coef_.ravel() for i in range(args.n_folds)], 0))

for i_c, classe in enumerate(class_queries):
        class_hyperplanes = [hyperplan[i_c] for hyperplan in all_hyperplans_all_windows]
        print(f"\n doing windows {args.windows} for classe {classe}: {np.degrees(angle_between(*class_hyperplanes))}")
        print(f"and cosine_similarity: {cosine_similarity(class_hyperplanes)}") #a1[np.newaxis,:], a2[np.newaxis,:])}")

set_trace()



if not args.dummy:
    ### SAVE RESULTS ###
    save_results(out_fn, AUC) #, all_models)
    save_results(out_fn, accuracy, fn_end="acc.npy")
    # save_preds(args, out_fn, mean_preds)
    # save_patterns(args, out_fn, all_models)
    # pickle.dump(ch_names, open(out_fn + '_ch_names.p', 'wb'))

    ### PLOT PERFORMANCE ###
    version = "v1" if int(args.subject[0:2]) < 8 else "v2"
    plot_perf(args, out_fn, AUC, args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=train_tmin, test_tmax=train_tmax, version=version)


print(f'Done with saving training plots and data. Elasped time since the script began: {(time.time()-start_time)/60:.2f}min')



###########################
######### TESTING #########
###########################

print('\n\nStarting testing')

### GET TEST DATA ###
for test_cond, test_query  in zip(args.test_cond, args.test_query):
    test_out_fn = f"{base_out_fn}_tested_on_{test_cond}_with_cond_{test_cond}"
    print(f'testing on {test_cond}, output path: {test_out_fn}')

    #check that the test fn is ok
    test_fn = f"{op.dirname(train_fn)}/{test_cond}-epo.fif" 
    print(test_fn)

    ### LOAD EPOCHS ###
    epochs = load_data_rsa(args, test_fn)
    test_tmin, test_tmax = epochs.tmin, epochs.tmax
    if test_cond == "two_objects":
        epochs.metadata = complement_md(epochs.metadata)

    # if len(all_models) != args.n_folds:
    #     print('\nWATCH OUT, inconsistency in the number of cv folds\n')
    # if len(all_models[0]) != X.shape[2]:
    #     print('\nWATCH OUT, inconsistency in the number of timepoints\n')
    #     print(f"found {len(all_models[0])} trained models and {X.shape[2]} test time point")

    print("hardcoded queries ... prety bad")
    if "Left" in test_query or "Right" in test_query:
        class_queries = [f"{test_query}=='{s}_{c}'" for s in shapes for c in colors]
    elif test_query == "Shape1+Colour1":
        class_queries = [f"Shape1=='{s}' and Colour1=='{c}'" for s in shapes for c in colors]
    elif test_query == "Shape2+Colour2":
        class_queries = [f"Shape2=='{s}' and Colour2=='{c}'" for s in shapes for c in colors]
    else:
        raise RuntimeError(f"Wrong query: {test_query}")
    print(class_queries)

    AUC, accuracy = test_decode_ovr(args, epochs, class_queries, all_models)

    if not args.dummy:
        ### SAVE RESULTS ###
        save_results(test_out_fn, AUC)
        save_results(out_fn, accuracy, fn_end="acc.npy")
        # save_preds(args, test_out_fn, mean_preds)

        ### PLOT PERFORMANCE ###
        plot_perf(args, test_out_fn, AUC, args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, test_tmax=test_tmax, gen_cond=test_cond, version=version)

        # # # plot pred
        # plot_perf(args, out_fn+f"_tested_on_{test_cond}", mean_preds, test_cond, ylabel='prediction')


print(f'Total elasped time since the script began: {(time.time()-start_time)/60:.2f}min')