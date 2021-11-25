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

from utils.decod import *

parser = argparse.ArgumentParser(description='MEG Decoding analysis')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='theo',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle sentence labels before training')
parser.add_argument('--freq-band', default='', help='name of frequency band to use for filtering (theta, alpha, beta, gamma)')
# parser.add_argument('--C', default=1, type=float, help='Regularization parameter')
parser.add_argument('--timegen', action='store_true', default=False, help='Whether to test probe trained at one time point also on all other timepoints')
parser.add_argument('--filter', default='', help='md query to filter trials before anything else (eg to use only matching trials')
parser.add_argument('--train-cond', default='localizer', help='localizer, one_object or two_objects')
parser.add_argument('--train-query-1', help='Metadata query for training classes')
parser.add_argument('--train-query-2', help='Metadata query for training classes')
parser.add_argument('--test-cond', default=[], action='append', help='localizer, one_object or two_objects, should have the same length as test-queries')
parser.add_argument('--test-query-1', default=[], action='append', help='Metadata query for testing classes')
parser.add_argument('--test-query-2', default=[], action='append', help='Metadata query for testing classes')
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
parser.add_argument('--t1', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--t2', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--label', default='', help='help to identify the result latter')
parser.add_argument('--dummy', action='store_true', default=False, help='Accelerates everything so that we can test that the pipeline is working. Will not yield any interesting result!!')
parser.add_argument('--equalize_events', action='store_true', default=False, help='subsample majority event classes to get same number of trials as the minority class')
parser.add_argument('-x', '--xdawn', action='store_true',  default=False, help='Whether to apply Xdawn spatial filtering before training decoder')

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

if len(args.test_cond) != len(args.test_query_1) or len(args.test_cond) != len(args.test_query_2):
    raise RuntimeError("Test conditions and test-queries should have the same length")

np.random.seed(args.seed)
start_time = time.time()

###########################
######## TRAINING #########
###########################

### GET EPOCHS FILENAMES ###
train_fn, test_fns, out_fn, test_out_fns = get_paths(args)

print('\nStarting training')
### LOAD EPOCHS ###
epochs = load_data(args, train_fn, args.train_query_1, args.train_query_2)
test_split_query_indices = get_split_indices(args.split_queries, epochs)
train_tmin, train_tmax = epochs[0].tmin, epochs[0].tmax
### GET DATA AND CONSTRUCT LABELS ###
X, y, nchan, ch_names = get_X_y_from_epochs_list(args, epochs, args.sfreq)
del epochs

n_times = X.shape[2]

if args.dummy:
    clf = LinearRegression(n_jobs=-1)
else:
    # clf = RidgeClassifier(class_weight='balanced')
    # clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 9), cv=5, class_weight='balanced', scoring='roc_auc')
    # clf = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=10000, verbose=False)
    clf = LogisticRegressionCV(Cs=10, class_weight='balanced', solver='lbfgs', max_iter=10000, verbose=False, cv=5, n_jobs=1)
    # clf = SVC()
    # clf = GridSearchCV(clf, {"kernel":('linear', 'rbf', 'poly'), "C":np.logspace(-4, 4, 9)})
# clf = mne.decoding.LinearModel(clf)


### DECODE ###
print(f'\nStarting training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')
all_models, AUC, accuracy, AUC_query = decode(args, X, y, clf, n_times, test_split_query_indices)
print(f'Finished training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min\n')

if not args.dummy:
    ### SAVE RESULTS ###
    save_results(out_fn, AUC) #, all_models)
    save_results(out_fn, accuracy, fn_end="acc")
    # save_preds(args, out_fn, mean_preds)
    # save_patterns(args, out_fn, all_models)
    pickle.dump(ch_names, open(out_fn + '_ch_names.p', 'wb'))

    ### PLOT PERFORMANCE ###
    version = "v1" if int(args.subject[0:2]) < 8 else "v2"
    plot_perf(args, out_fn, AUC, args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=train_tmin, test_tmax=train_tmax, version=version)

    # # # plot pred
    # plot_perf(args, out_fn, mean_preds, ylabel='prediction')

    if AUC_query is not None: # save the results for all the splits
        for i_query, query in enumerate(args.split_queries):
            query = '_'.join(query.split()) # replace spaces by underscores
            query = shorten_filename(query) # shorten string by removing unnecessary stuff
            save_results(out_fn+f'_for_{query}', AUC_query[:,:,i_query])
            plot_perf(args, out_fn+f'_for_{query}', AUC_query[:,:,i_query], args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=train_tmin, test_tmax=train_tmax, version=version)

        # # save every contrast - full minus every split
        # for i_query, query in enumerate(args.split_queries):
        #     query = '_'.join(query.split()) # replace spaces by underscores
        #     query = shorten_filename(query) # shorten string by removing unnecessary stuff
        #     save_results(out_fn+f'_for_{query}_full_minus_split', AUC - AUC_query[:,:,i_query])
        #     plot_perf(args, out_fn+f'_for_{query}_full_minus_split', AUC - AUC_query[:,:,i_query], args.train_cond, contrast=True, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=train_tmin, test_tmax=train_tmax, version=version)


# # plot pred
# if mean_preds_query is not None: # save the results for all the splits
#     for i_query, query in enumerate(args.split_queries):
#         query = '_'.join(query.split()) # replace spaces by underscores
#         query = shorten_filename(query) # shorten string by removing unnecessary stuff
#         # save_preds(args, out_fn+f'_for_{query}', mean_preds_query[:,:,i_query])
#         # plot_perf(args, out_fn+f'_for_{query}', mean_preds_query[:,:,i_query], ylabel='prediction')

#     # save every contrast - full minus every split
#     for i_query, query in enumerate(args.split_queries):
#         query = '_'.join(query.split()) # replace spaces by underscores
#         query = shorten_filename(query) # shorten string by removing unnecessary stuff
#         # save_preds(args, out_fn+f'_for_{query}_full_minus_split', mean_preds - mean_preds_query[:,:,i_query])
#         # plot pred
#         # plot_perf(args, out_fn+f'_for_{query}_full_minus_split', 
#         #     mean_preds - mean_preds_query[:,:,i_query], ylabel='prediction', contrast=True)

print(f'Done with saving training plots and data. Elasped time since the script began: {(time.time()-start_time)/60:.2f}min')


    # ### LOAD MODELS FROM FILE ###
    # print('No training specified. Loading saved models')
    # all_models = pickle.load(open(out_fn + '_all_models.p', 'rb'))
    # print(f'Done loading models')


###########################
######### TESTING #########
###########################

print('\n\nStarting testing')

### GET TEST DATA ###
for test_cond, test_fn, test_out_fn, test_query_1, test_query_2  in zip(args.test_cond, test_fns, test_out_fns, args.test_query_1, args.test_query_2):
    print(f'testing on {test_cond}, output path: {test_out_fn}')

    ### LOAD EPOCHS ###
    epochs, test_split_query_indices = load_data(args, test_fn, test_query_1, test_query_2)
    test_tmin, test_tmax = epochs[0].tmin, epochs[0].tmax
    ### GET DATA AND CONSTRUCT LABELS ###
    X, y, nchan, ch_names = get_data(args, epochs, args.sfreq)
    del epochs

    if len(all_models) != args.n_folds:
        print('\nWATCH OUT, inconsistency in the number of cv folds\n')
    if len(all_models[0]) != X.shape[2]:
        print('\nWATCH OUT, inconsistency in the number of timepoints\n')
        print(f"found {len(all_models[0])} trained models and {X.shape[2]} test time point")

    AUC, accuracy, mean_preds, AUC_query = test_decode(args, X, y, all_models, test_split_query_indices)

    if not args.dummy:
        ### SAVE RESULTS ###
        save_results(test_out_fn, AUC)
        save_results(out_fn, accuracy, fn_end="acc")
        # save_preds(args, test_out_fn, mean_preds)

        ### PLOT PERFORMANCE ###
        plot_perf(args, test_out_fn, AUC, args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, test_tmax=test_tmax, gen_cond=test_cond, version=version)

        # # # plot pred
        # plot_perf(args, out_fn+f"_tested_on_{test_cond}", mean_preds, test_cond, ylabel='prediction')

        if AUC_query is not None: # save the results for all the splits
            for i_query, query in enumerate(args.split_queries):
                query = '_'.join(query.split()) # replace spaces by underscores
                query = shorten_filename(query) # shorten string by removing unnecessary stuff
                save_results(out_fn+f'_for_{query}', AUC_query[:,:,i_query])
                plot_perf(args, out_fn+f'_for_{query}', AUC_query[:,:,i_query], args.train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, test_tmax=test_tmax, gen_cond=test_cond, version=version)

            # # save every contrast - full minus every split
            # for i_query, query in enumerate(args.split_queries):
            #     query = '_'.join(query.split()) # replace spaces by underscores
            #     query = shorten_filename(query) # shorten string by removing unnecessary stuff
            #     save_results(out_fn+f'_for_{query}_full_minus_split', AUC - AUC_query[:,:,i_query])
            #     plot_perf(args, out_fn+f'_for_{query}_full_minus_split', AUC - AUC_query[:,:,i_query], args.train_cond, contrast=True, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=train_tmin, test_tmax=train_tmax, version=version)



print(f'Total elasped time since the script began: {(time.time()-start_time)/60:.2f}min')