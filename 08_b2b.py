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

# local imports
from utils.b2b import *

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
parser.add_argument('-x', '--xdawn', action='store_true',  default=False, help='Whether to apply Xdawn spatial filtering before training decoder')
parser.add_argument('--filter', default='', help='md query to filter trials before anything else (eg to use only matching trials')
# parser.add_argument('--train-cond', default='localizer', help='localizer, one_object or two_objects')
# parser.add_argument('--train-query', help='Metadata query for training classes')
parser.add_argument('--train_cond', default=[], action='append', help='localizer, one_object or two_objects, should have the same length as queries')
parser.add_argument('--queries', default=[], action='append', help='Metadata query for testing classes')
# parser.add_argument('--windows', default=[], action='append', help='tmin and tmax to crop the epochs, one for each conds')

# optionals, overwrite the config if passed
parser.add_argument('--test-cond', default=[], action='append', help='localizer, one_object or two_objects, should have the same length as test-queries')
parser.add_argument('--test-query', default=[], action='append', help='Metadata query for testing classes')
parser.add_argument('--sfreq', type=int, help='sampling frequency')
parser.add_argument('--n_folds', type=int, help='sampling frequency')

# not used, kept for consistency
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
parser.add_argument('--t1', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--t2', action='append', default=[], help="Metadata query for generalization test")
parser.add_argument('--equalize_events', action='store_true', default=False, help='subsample majority event classes to get same number of trials as the minority class')

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

# if len(args.train_cond) != len(args.queries):
#     raise RuntimeError("Test conditions and test-queries should have the same length")

np.random.seed(args.seed)
start_time = time.time()

###########################
######## TRAINING #########
###########################

### GET EPOCHS FILENAMES ###
fns, test_fns, out_fn, test_out_fns = get_paths(args, "B2B")
print(fns)

print('\nStarting training')
### LOAD EPOCHS ###
all_epochs = []
for fn in fns:
	all_epochs.append(load_data(args, fn)[0])

# get time from longest epochs
longest_epo_idx = np.argmax([len(epo.times) for epo in all_epochs])
times = all_epochs[longest_epo_idx].times
# print(times)

# if args.windows:
#     args.windows = [w.replace(" ", "") for w in args.windows] # remove spaces
#     wins = [f"#{'#'.join([args.windows[0], w])}#" for w in args.windows] # string to add to the out fns
#     out_fn += wins[0]
#     for i in range(len(test_out_fns)): test_out_fns[i] += wins[i+1]
# windows = [tuple([float(x) for x in win.split(",")]) for win in args.windows]
# if windows: all_epochs = [epo.crop(*windows[0]) for epo in all_epochs]

## GET QUERIES
class_queries = []
for query in args.queries:
    class_queries.extend(get_class_queries(query))
print(class_queries)


### B2B ###
print(f'\nStarting training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')
all_betas = run_b2b(args, all_epochs, class_queries)
print(f'Finished training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min\n')

get_position = True if np.any(["two_objects" in cond for cond in args.train_cond]) else False
legend_labels = class_queries2legend_labels(class_queries, get_position=get_position)
plot_betas(betas=all_betas, times=times, labels=legend_labels, out_fn=f'{out_fn}_all_betas.png')

unique_conds, groups_indices = group_conds(legend_labels)
grouped_betas = np.array([np.mean(all_betas[:, indices], 1) for indices in groups_indices]).T
plot_betas(betas=grouped_betas, times=times, labels=unique_conds, out_fn=f'{out_fn}_betas_grouped.png')

save_results(out_fn, all_betas, fn_end="all_betas.npy")
save_results(out_fn, grouped_betas, fn_end="betas_grouped.npy")
save_pickle(f"{out_fn}_all_labels.p", legend_labels)
save_pickle(f"{out_fn}_labels_grouped.p", unique_conds)



# if not args.dummy:
#     ### SAVE RESULTS ###
#     save_results(out_fn, AUC) #, all_models)
#     # save_patterns(args, out_fn, all_models)

#     ### PLOT PERFORMANCE ###
#     version = "v1" if int(args.subject[0:2]) < 8 else "v2"
#     plot_perf(args, out_fn, AUC, args.cond, tmin=tmin, tmax=tmax, \
#               test_tmin=tmin, test_tmax=tmax, version=version)


# print(f'Done with saving training plots and data. Elasped time since the script began: {(time.time()-start_time)/60:.2f}min')


# ###########################
# ######### TESTING #########
# ###########################

# print('\n\nStarting testing')

# ### GET TEST DATA ###
# for i_test, (cond, query, test_fn, test_out_fn)  in enumerate(zip(args.test_cond, args.test_query, test_fns, test_out_fns)):
#     print(f'testing on {cond}, output path: {test_out_fn}')

#     ### LOAD EPOCHS ###
#     epochs = load_data(args, test_fn)[0]
#     if windows: epochs = epochs.crop(*windows[i_test+1]) # first window is for training
#     test_tmin, test_tmax = epochs.tmin, epochs.tmax

#     class_queries = get_class_queries(query)

#     AUC, accuracy = test_decode_ovr(args, epochs, class_queries, all_models)

#     if not args.dummy:
#         ### SAVE RESULTS ###
#         save_results(test_out_fn, AUC)
#         save_results(out_fn, accuracy, fn_end="acc.npy")

#         ### PLOT PERFORMANCE ###
#         plot_perf(args, test_out_fn, AUC, args.cond, tmin=tmin, tmax=tmax, \
#                   test_tmin=test_tmin, test_tmax=test_tmax, gen_cond=cond, version=version)

print(f'Total elasped time since the script began: {(time.time()-start_time)/60:.2f}min')