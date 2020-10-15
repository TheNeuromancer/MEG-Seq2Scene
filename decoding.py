import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import mne
import numpy as np
from ipdb import set_trace
import argparse
import pickle
import time

from utils.decod import *

parser = argparse.ArgumentParser(description='MEG Decoding analysis')
parser.add_argument('-r', '--root-path', default='/neurospin/unicog/protocols/MEG/Seq2Scene/', help='root path')
parser.add_argument('-i', '--in-dir', default='Data/Epochs/', help='input directory')
parser.add_argument('-o', '--out-dir', default='/Epochs/', help='output directory')
parser.add_argument('-s', '--subject', default='theo',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('-v', '--version', default=1, type=int, help='version of the script')
parser.add_argument('--reduc-dim', default=0, type=float, help='number of PCA components, or percentage of variance to keep. If 0, do not apply PCA')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--cat', default=0, type=int, help='How many timesteps to concatenate (default to one)')
parser.add_argument('--mean', default=False, action='store_true', help='Whether to actually average over instead of concatenating consecutive timesteps')
parser.add_argument('--smooth', default=5, type=int, help='Whether to apply gaussian kernel smoothing and on how many timesteps')
parser.add_argument('--baseline', action='store_true', default=False, help='Whether to apply baseline correction')
parser.add_argument('--mask_baseline', action='store_true', default=False, help='Whether to apply baseline correction from mask to first word onset')
parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle sentence labels before training')
parser.add_argument('--tmin', default=-.5, type=float, help='Start of the epochs')
parser.add_argument('--tmax', default=4., type=float, help='End of the epochs')
parser.add_argument('--C', default=1, type=float, help='Regularization parameter')
parser.add_argument('--timegen', action='store_true', default=False, help='Whether to test probe trained at one time point also on all other timepoints')
parser.add_argument('--sfreq', default=100, type=int, help='Final sampling frequency to use (applying resampling)')
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
parser.add_argument('--t1', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--t2', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--localizer', action='store_true', default=False, help='Whether to use only electrode that were significant in the localizer')
parser.add_argument('--path2loc', default='Single_Chan_vs5/CMR_sent', help='path to the localizer results (dict with value 1 for each channel that passes the test, 0 otherwise')
parser.add_argument('--pval-thresh', default=0.05, type=float, help='pvalue threshold under which a channel is kept for the localizer')
parser.add_argument('--subtract-evoked', action='store_true', default=False, help='Whether to subtract the evoked signal from the epochs')
parser.add_argument('--clip', default=False, action='store_true', help='Whether to clip to the 5th and 95th percentile for each channel')
parser.add_argument('--crossval', default='kfold',help='cross-validation scheme. "kfold" or "sufflesplit"')
parser.add_argument('--n_folds', default=5, type=int, help='Number of cross-validation folds')
parser.add_argument('--avg-clf', default=False, action='store_true', help='Whether to average classifiers across cval folds')
parser.add_argument('--freq-band', default='', help='name of frequency band to use for filtering (theta, alpha, beta, gamma)')

parser.add_argument('--train-query-1', help='Metadata query for training classes')
parser.add_argument('--train-query-2', help='Metadata query for training classes')
parser.add_argument('--test-query-1', help='Metadata query for testing classes')
parser.add_argument('--test-query-2', help='Metadata query for testing classes')
parser.add_argument('--train-cond', default='localizer', help='localizer, one_object or two_objects')
parser.add_argument('--test-cond', default=[], action='append', help='localizer, one_object or two_objects')

print(mne.__version__)
args = parser.parse_args()
print(args)

np.random.seed(args.seed)

start_time = time.time()


###########################
######## TRAINING #########
###########################

### GET EPOCHS FILENAMES ###
train_fn, test_fns, out_fn = get_paths(args)

if op.exists(out_fn + '_AUC.npy'): # warn and stop if args.overwrite is set to False
    print('\noutput files for training already exist...')
    if args.overwrite:
        print('overwrite is set to True ... overwriting\n')
    else:
        print('overwrite is set to False ... exiting')
        exit()

print('\nStarting training')
### LOAD EPOCHS ###
epochs, test_split_query_indices = load_data(args, train_fn, args.train_query_1, args.train_query_2)
train_tmin, train_tmax = epochs[0].tmin, epochs[0].tmax
### GET DATA AND CONSTRUCT LABELS ###
X, y, nchan, ch_names = get_data(args, epochs, args.sfreq)
del epochs

n_times = X.shape[2]


clf = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000, verbose=False)
# grid_logreg = {'C':np.logspace(-3, 3., 7)}
# clf = GridSearchCV(logreg, grid_logreg, n_jobs=30, cv=5, scoring='roc_auc', iid=True)

clf = mne.decoding.LinearModel(clf)


### DECODE ###
print(f'\nStarting training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')
all_models, AUC, AUC_query = decode(args, X, y, clf, n_times, test_split_query_indices)
print(f'Finished training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min\n')

### SAVE RESULTS ###
save_results(args, out_fn, AUC, all_models)
# save_preds(args, out_fn, mean_preds)
save_patterns(args, out_fn, all_models)
pickle.dump(ch_names, open(out_fn + '_ch_names.p', 'wb'))

### PLOT PERFORMANCE ###
plot_perf(args, out_fn, AUC, args.train_cond, train_tmin=train_tmin, 
    train_tmax=train_tmax, test_tmin=train_tmin, test_tmax=train_tmax)

# # # plot pred
# plot_perf(args, out_fn, mean_preds, ylabel='prediction')

if AUC_query is not None: # save the results for all the splits
    for i_query, query in enumerate(args.split_queries):
        query = '_'.join(query.split()) # replace spaces by underscores
        query = shorten_filename(query) # shorten string by removing unnecessary stuff
        save_results(args, out_fn+f'_for_{query}', AUC_query[:,:,i_query])
        plot_perf(args, out_fn+f'_for_{query}', AUC_query[:,:,i_query], args.train_cond, 
            train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=train_tmin, test_tmax=train_tmax)

    # save every contrast - full minus every split
    for i_query, query in enumerate(args.split_queries):
        query = '_'.join(query.split()) # replace spaces by underscores
        query = shorten_filename(query) # shorten string by removing unnecessary stuff
        save_results(args, out_fn+f'_for_{query}_full_minus_split', AUC - AUC_query[:,:,i_query])
        plot_perf(args, out_fn+f'_for_{query}_full_minus_split', AUC - AUC_query[:,:,i_query], args.train_cond, 
            contrast=True, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=train_tmin, test_tmax=train_tmax)


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
for test_cond, test_fn in zip(args.test_cond, test_fns):
    print(f'testing on {test_cond}')

    if op.exists(out_fn+f"tested_on_{test_cond}" + '_AUC.npy'): # warn and stop if args.overwrite is set to False
        print('\noutput files for training already exist...')
        if args.overwrite:
            print('overwrite is set to True ... overwriting\n')
        else:
            print('overwrite is set to False ... loading models and moving on to the next test queries\n')

    ### LOAD EPOCHS ###
    epochs, test_split_query_indices = load_data(args, test_fn, args.test_query_1, args.test_query_2)
    test_tmin, test_tmax = epochs[0].tmin, epochs[0].tmax
    ### GET DATA AND CONSTRUCT LABELS ###
    X, y, nchan, ch_names = get_data(args, epochs, args.sfreq)
    del epochs

    if len(all_models) != args.n_folds:
        print('\nWATCH OUT, inconsistency in the number of cv folds\n')
    if len(all_models[0]) != X.shape[2]:
        print('\nWATCH OUT, inconsistency in the number of timepoints\n')
        print(f"found {len(all_models[0])} trained models and {X.shape[2]} test time point")

    AUC, mean_preds = test_decode(args, X, y, all_models)

    ### SAVE RESULTS ###
    save_results(args, out_fn+f"_tested_on_{test_cond}", AUC)
    # save_preds(args, out_fn+f"_tested_on_{test_cond}", mean_preds)

    ### PLOT PERFORMANCE ###
    plot_perf(args, out_fn+f"_tested_on_{test_cond}", AUC, test_cond, \
        train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, test_tmax=test_tmax, train_cond=args.train_cond)

    # # # plot pred
    # plot_perf(args, out_fn+f"_tested_on_{test_cond}", mean_preds, test_cond, ylabel='prediction')


print(f'Total elasped time since the script began: {(time.time()-start_time)/60:.2f}min')