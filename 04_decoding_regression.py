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

from utils.decod import *

parser = argparse.ArgumentParser(description='MEG Decoding analysis')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='theo',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle sentence labels before training')
parser.add_argument('--freq-band', default='', help='name of frequency band to use for filtering (theta, alpha, beta, gamma)')
parser.add_argument('--timegen', action='store_true', default=False, help='Whether to test probe trained at one time point also on all other timepoints')
parser.add_argument('--filter', default=None, help='md query to filter trials before anything else (eg to use only matching trials')
parser.add_argument('--train-cond', default='localizer', help='localizer, one_object or two_objects')
parser.add_argument('--train-query', help='Metadata query for training classes')
parser.add_argument('--test-cond', default=[], action='append', help='localizer, one_object or two_objects, should have the same length as test-queries')
parser.add_argument('--test-query-1', default=[], action='append', help='Metadata query for testing classes')
parser.add_argument('--test-query-2', default=[], action='append', help='Metadata query for testing classes')
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
parser.add_argument('--t1', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--t2', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--label', default='', help='help to identify the result latter')
parser.add_argument('--dummy', action='store_true', default=False, help='Accelerates everything so that we can test that the pipeline is working. Will not yield any interesting result!!')
parser.add_argument('--equalize_events', action='store_true', default=None, help='subsample majority event classes to get same number of trials as the minority class')
parser.add_argument('--equalize_split_events', action='store_true', default=None, help='subsample majority event classes IN EACH SPLIT QUERY to get same number of trials as the minority class')
parser.add_argument('-x', '--xdawn', action='store_true',  default=None, help='Whether to apply Xdawn spatial filtering before training decoder')
parser.add_argument('-a', '--autoreject', action='store_true',  default=None, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('--quality_th', default=None, type=float, help='Whether to apply Autoreject on the epochs before training decoder')
parser.add_argument('--micro_ave', default=None, type=int, help='Trial micro-averaging to boost decoding performance')

# optionals, overwrite the config if passed
parser.add_argument('--sfreq', type=int, help='sampling frequency')
parser.add_argument('--n_folds', type=int, help='sampling frequency')

parser.add_argument('--localizer', action='store_true', default=False, help='Whether to use only electrode that were significant in the localizer')
parser.add_argument('--auc_thresh', default=0.5, type=float, help='pvalue threshold under which a channel is kept for the localizer')

# not implemented
parser.add_argument('--windows', default=[], action='append', help='tmin and tmax to crop the epochs, one for each train and test cond')
parser.add_argument('-r', '--response_lock', action='store_true',  default=None, help='Whether to Use response locked epochs or classical stim-locked')
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

if len(args.test_cond) != len(args.test_query_1) or len(args.test_cond) != len(args.test_query_2):
    raise RuntimeError("Test conditions and test-queries should have the same length")

np.random.seed(args.seed)
start_time = time.time()

###########################
######## TRAINING #########
###########################

### GET EPOCHS FILENAMES ###
train_fn, test_fns, out_fn, test_out_fns = get_paths(args, dirname="Regression_decoding")

print('\nStarting training')
### LOAD EPOCHS ###
epochs = load_data(args, train_fn)[0] # OVR-style data_loading: keep all, pass class queries to decode func which create the X and y.
train_tmin, train_tmax = epochs[0].tmin, epochs[0].tmax
### GET DATA AND CONSTRUCT LABELS ###
# X, y, nchan, ch_names = get_X_y_from_epochs_list(args, epochs, args.sfreq)
# del epochs
class_queries = get_class_queries(args.train_query)

if args.dummy:
    clf = LinearRegression(n_jobs=-1)
    setattr(args, 'n_folds', 2)
else:
    # clf = Ridge(class_weight='balanced')
    clf = RidgeCV(alphas=np.logspace(-4, 4, 9), cv=5)
clf = mne.decoding.LinearModel(clf)

### DECODE ###
print(f'\nStarting training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')
all_models, R, R_split_queries = regression_decode(args, epochs, class_queries, clf)
print(f'Finished training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min\n')

### SAVE RESULTS ###
save_results(out_fn, R, all_models, fn_end="R")
save_best_pattern(out_fn, R, all_models) ## Save best model's pattern
# save_results(out_fn, R2, all_models, fn_end="R2")

# ### PLOT PERFORMANCE ### # no split query à priori
# version = "v1" if int(args.subject[0:2]) < 8 else "v2"
# plot_perf(args, out_fn, R2, args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=train_tmin, test_tmax=train_tmax, version=version)
# save_best_pattern(out_fn, R2, all_models) ## Save best model's pattern
# if R2_query is not None: # save the results for all the splits
#     for i_query, query in enumerate(corrected_split_queries):
#         query = '_'.join(query.split()) # replace spaces by underscores
#         query = shorten_filename(query) # shorten string by removing unnecessary stuff
#         save_results(out_fn+f'_for_{query}', R2_query[:,:,i_query])
#         plot_perf(args, out_fn+f'_for_{query}', R2_query[:,:,i_query], args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=train_tmin, test_tmax=train_tmax, version=version)
#         # save_preds(args, out_fn+f'_for_{query}', mean_preds_query[:,:,i_query])

    # # save every contrast - full minus every split
    # for i_query, query in enumerate(corrected_split_queries):
    #     query = '_'.join(query.split()) # replace spaces by underscores
    #     query = shorten_filename(query) # shorten string by removing unnecessary stuff
    #     save_results(out_fn+f'_for_{query}_full_minus_split', R2 - R2_query[:,:,i_query])
    #     plot_perf(args, out_fn+f'_for_{query}_full_minus_split', R2 - R2_query[:,:,i_query], args.train_cond, contrast=True, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=train_tmin, test_tmax=train_tmax, version=version)

# # plot pred
# if mean_preds_query is not None: # save the results for all the splits
#     for i_query, query in enumerate(corrected_split_queries):
#         query = '_'.join(query.split()) # replace spaces by underscores
#         query = shorten_filename(query) # shorten string by removing unnecessary stuff
#         # save_preds(args, out_fn+f'_for_{query}', mean_preds_query[:,:,i_query])
#         # plot_perf(args, out_fn+f'_for_{query}', mean_preds_query[:,:,i_query], ylabel='prediction')

#     # save every contrast - full minus every split
#     for i_query, query in enumerate(corrected_split_queries):
#         query = '_'.join(query.split()) # replace spaces by underscores
#         query = shorten_filename(query) # shorten string by removing unnecessary stuff
#         # save_preds(args, out_fn+f'_for_{query}_full_minus_split', mean_preds - mean_preds_query[:,:,i_query])
#         # plot pred
#         # plot_perf(args, out_fn+f'_for_{query}_full_minus_split', 
#         #     mean_preds - mean_preds_query[:,:,i_query], ylabel='prediction', contrast=True)

print(f'Done with saving training plots and data. Elasped time since the script began: {(time.time()-start_time)/60:.2f}min')


    # ### LOAD MODELS FROM FILE ###
    # print('No training specified. Loading saved models')
    # all_models = pickle.load(open(out_fn + '_all_models.p', 'rb'))
    # print(f'Done loading models')


###########################
######### TESTING ######### À priori pas besoin
###########################

# print('\n\nStarting testing')
# ### GET TEST DATA ###
# for test_cond, test_fn, test_out_fn, test_query_1, test_query_2  in zip(args.test_cond, test_fns, test_out_fns, args.test_query_1, args.test_query_2):
#     print(f'testing on {test_cond}, output path: {test_out_fn}')

#     ### LOAD EPOCHS ###
#     epochs = load_data(args, test_fn, test_query_1, test_query_2)
#     test_split_query_indices, corrected_split_queries = get_split_indices(args.split_queries, epochs)
#     test_tmin, test_tmax = epochs[0].tmin, epochs[0].tmax
#     ### GET DATA AND CONSTRUCT LABELS ###
#     # X, y, nchan, ch_names = get_data(args, epochs, args.sfreq)
#     X, y, nchan, ch_names = get_X_y_from_epochs_list(args, epochs, args.sfreq)
#     del epochs

#     # if all_models.shape[0] != X.shape[2]: # X should have a different shape if train and test cond are different
#     #     print('\nWATCH OUT, inconsistency in the number of timepoints\n')
#     if all_models.shape[1] != args.n_folds:
#         print('\nWATCH OUT, inconsistency in the number of cv folds\n')

#     print(f"found {all_models.shape[0]} trained models and {X.shape[2]} test time point")
#     R2, accuracy, mean_preds, R2_query = test_decode(args, X, y, all_models, test_split_query_indices)

#     ### SAVE RESULTS ###
#     save_results(test_out_fn, R2)
#     # save_results(test_out_fn, accuracy, fn_end="acc")
#     # save_preds(args, test_out_fn, mean_preds)

#     ### PLOT PERFORMANCE ###
#     plot_perf(args, test_out_fn, R2, args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, test_tmax=test_tmax, gen_cond=test_cond, version=version)

#     # # # plot pred
#     # plot_perf(args, out_fn+f"_tested_on_{test_cond}", mean_preds, test_cond, ylabel='prediction')

#     if R2_query is not None: # save the results for all the splits
#         for i_query, query in enumerate(corrected_split_queries):
#             query = '_'.join(query.split()) # replace spaces by underscores
#             query = shorten_filename(query) # shorten string by removing unnecessary stuff
#             save_results(test_out_fn+f'_for_{query}', R2_query[:,:,i_query])
#             plot_perf(args, test_out_fn+f'_for_{query}', R2_query[:,:,i_query], args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, test_tmax=test_tmax, gen_cond=test_cond, version=version)

print(f'Total elasped time since the script began: {(time.time()-start_time)/60:.2f}min')