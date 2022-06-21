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
from itertools import permutations, combinations

# local imports
from utils.decod import *

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
parser.add_argument('--test_quality', action='store_true', default=False, help='Change the out directory name, used for testing the quality of single runs.')
parser.add_argument('--filter', default='', help='md query to filter trials before anything else (eg to use only matching trials')

parser.add_argument('--windows', action='append', default=[], help='list of time windows to train classifiers, test generalization and compute angles')
parser.add_argument('--test-all-times', action='store_true', default=False, help='whether to test all time points after training on a single window')

# optionals, overwrite the config if passed
parser.add_argument('--sfreq', type=int, help='sampling frequency')
parser.add_argument('--n_folds', type=int, help='sampling frequency')

# useless here, kept for consistency
parser.add_argument('--equalize_events', action='store_true', default=False, help='subsample majority event classes to get same number of trials as the minority class')
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
train_fns, _, out_fn, _ = get_paths(args, "Decoding_window")
clf_idx = 2 if args.reduc_dim_win else 1


###########################
######## TRAINING #########
###########################

# clf = LogisticRegression(C=1, class_weight='balanced', solver='liblinear', multi_class='auto')
clf = LogisticRegressionCV(Cs=10, solver='liblinear', class_weight='balanced', multi_class='auto', n_jobs=-1, cv=5, max_iter=10000)
# tuned_parameters = [{'kernel': ['linear'], 'C': np.logspace(-2, 4, 7)},
#                     {'kernel': ['rbf'], 'gamma': [1, .1, 1e-2, 1e-3, 1e-4], 'C': np.logspace(-1, 3, 5)},
#                     {'kernel': ['poly'], 'degree': [2, 3, 4, 5, 6], 'C': np.logspace(-1, 3, 5)}]
# clf = SVC(class_weight='balanced')
# clf = GridSearchCV(clf, tuned_parameters, scoring='roc_auc', cv=10, refit=True, verbose=1)
clf = OneVsRestClassifier(clf, n_jobs=1)

### DECODE ###
all_windows_models = []
all_hyperplans = {}
all_epochs = []
all_windows_epochs = []
all_windows_queries = []
all_windows_AUC = []
all_windows_accuracy = []
print(f'\nStarting training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')
for window_str, query, cond, train_fn in zip(args.windows, args.train_query, args.train_cond, train_fns):
    window = tuple([float(x) for x in window_str.split(",")])
    all_hyperplans[window] = {}
    print(f"Doing window : {window}, {train_fn}")

    ### LOAD EPOCHS ###
    epochs = load_data(args, train_fn)[0]
    all_epochs.append(epochs)
    epo_window = epochs.copy().crop(*window)
    all_windows_epochs.append(epo_window)
    
    ## GET QUERIES
    class_queries = get_class_queries(query)
    all_windows_queries.append(class_queries)
    print(class_queries)

    ## DECODE
    AUC, accuracy, ovr_model = decode_window(args, clf, epo_window, class_queries)
    all_windows_models.append(ovr_model)
    all_windows_accuracy.append(accuracy)
    all_windows_AUC.append(AUC)
    
    # # inter-fold consistency
    # for i_class, classe in enumerate(class_queries):
    #     print(f"\n doing class {classe}")
    #     all_angles = []
    #     all_cos = []
    #     all_hyperplans[window][classe] = []
    #     for i1 in range(args.n_folds):
    #         all_hyperplans[window][classe].append(ovr_model[i1][clf_idx].estimators_[i_class].coef_.ravel())
    #         for i2 in range(i1, args.n_folds):
    #             a1, a2 = ovr_model[i1][clf_idx].estimators_[i_class].coef_.ravel(), ovr_model[i2][clf_idx].estimators_[i_class].coef_.ravel()
    #             all_angles.append(np.degrees(angle_between(a1, a2)))
    #             all_cos.append(cosine_similarity(a1[np.newaxis,:], a2[np.newaxis,:])[0][0])
    #     print(f"inter folds consistency for query: {classe}: {np.mean(all_angles)} +/- {np.std(all_angles)} degrees and {np.mean(all_cos)} +/- {np.std(all_cos)}")

    # # inter-condition angles
    # all_cond_hyperplans = []
    # for i_class, classe in enumerate(class_queries):
    #     all_cond_hyperplans.append(np.mean([ovr_model[i][clf_idx].estimators_[i_class].coef_.ravel() for i in range(args.n_folds)], 0))
    # for i_c1, classe1 in enumerate(class_queries):
    #     for i_c2, classe2 in enumerate(class_queries):
    #         a1, a2 = all_cond_hyperplans[i_c1], all_cond_hyperplans[i_c2]
    #         print(f"\n doing classes {classe1} vs {classe2}: {np.degrees(angle_between(a1, a2))} degrees")
    #         print(f"and cosine_similarity: {cosine_similarity(a1[np.newaxis,:], a2[np.newaxis,:])[0][0]}")

    print("Done")


print(f'Finished training. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min\n')

## TEST DECODING
gen_AUC = {}
gen_accuracy = {}
for i_m, model in enumerate(all_windows_models):
    gen_AUC[f"{args.train_query[i_m]}-{args.train_cond[i_m]}"] = {}
    gen_accuracy[f"{args.train_query[i_m]}-{args.train_cond[i_m]}"] = {}
    for i_c, (epo_window, class_queries) in enumerate(zip(all_windows_epochs, all_windows_queries)):
        if i_m == i_c:  # already tested using cval
            test_AUC = all_windows_AUC[i_m]
            test_accuracy = all_windows_accuracy[i_m]
        else:
            test_AUC, test_accuracy = test_decode_window(args, epo_window, class_queries, model)
        gen_AUC[f"{args.train_query[i_m]}-{args.train_cond[i_m]}"][f"{args.train_query[i_c]}-{args.train_cond[i_c]}"] = test_AUC
        gen_accuracy[f"{args.train_query[i_m]}-{args.train_cond[i_m]}"][f"{args.train_query[i_c]}-{args.train_cond[i_c]}"] = test_accuracy
        print(f"Generalization perf from {args.train_query[i_m]} {args.train_cond[i_m]} {args.windows[i_m]}s \
to {args.train_query[i_c]} {args.train_cond[i_c]} {args.windows[i_c]}s: AUC = {test_AUC} ; accuracy = {test_accuracy}")

save_pickle(f"{out_fn}_AUC.p", gen_AUC)
save_pickle(f"{out_fn}_accuracy.p", gen_accuracy)



## TEST DECODING ALL TIME POINTs
nb_cat = int((window[1] - window[0]) * args.sfreq) + 1 # nb of timepoints to concatenate to get the same length as the training window
if args.test_all_times:
    gen_AUC = {}
    gen_accuracy = {}
    gen_AUC["windows"] = {} # save training windows
    for i_m, model in enumerate(all_windows_models):
        gen_AUC[f"{args.train_query[i_m]}-{args.train_cond[i_m]}"] = {}
        gen_accuracy[f"{args.train_query[i_m]}-{args.train_cond[i_m]}"] = {}
        gen_AUC["windows"][f"{args.train_query[i_m]}-{args.train_cond[i_m]}"] = window # save training windows
        for i_c, (epo, class_queries) in enumerate(zip(all_epochs, all_windows_queries)):
            # if i_m == i_c:  # already tested using cval
            #     test_AUC = all_windows_AUC[i_m]
            #     test_accuracy = all_windows_accuracy[i_m]
            # else:
            for margin in range(0, 10):
                try:
                    test_AUC, test_accuracy = test_decode_sliding_window(args, epo, class_queries, model, nb_cat+margin)
                except:
                    continue

            gen_AUC[f"{args.train_query[i_m]}-{args.train_cond[i_m]}"][f"{args.train_query[i_c]}-{args.train_cond[i_c]}"] = test_AUC
            gen_accuracy[f"{args.train_query[i_m]}-{args.train_cond[i_m]}"][f"{args.train_query[i_c]}-{args.train_cond[i_c]}"] = test_accuracy
    #         print(f"Generalization perf from {args.train_query[i_m]} {args.train_cond[i_m]} {args.windows[i_m]}s \
    # to {args.train_query[i_c]} {args.train_cond[i_c]} {args.windows[i_c]}s: AUC = {test_AUC} ; accuracy = {test_accuracy}")

    save_pickle(f"{out_fn}_AUC_allt.p", gen_AUC)
    save_pickle(f"{out_fn}_accuracy_allt.p", gen_accuracy)



# # ## ACROSS WINDOWS AVERAGING COSINES 
# # all_hyperplans_all_windows = []
# # for i_win, (window, window_models) in enumerate(zip(args.windows, all_windows_models)):
# #     all_hyperplans_all_windows.append([])
# #     for i_class, classe in enumerate(class_queries):
# #         all_hyperplans_all_windows[-1].append([window_models[i][clf_idx].estimators_[i_class].coef_.ravel() for i in range(args.n_folds)])
# # for i_c, classe in enumerate(class_queries):
# #         class_hyperplanes = np.array([hyperplan[i_c] for hyperplan in all_hyperplans_all_windows])
# #         shapes = class_hyperplanes.shape[0:2]
# #         class_hyperplanes = class_hyperplanes.reshape((-1, class_hyperplanes.shape[-1]))
# #         cos_sim = cosine_similarity(class_hyperplanes)
# #         cos_sim = cos_sim.reshape((*shapes, *shapes))
# #         cos_sim = cos_sim.mean(1).mean(2) # average over folds
# #         print(f"(cosine average) cosine_similarity: {cos_sim}") #a1[np.newaxis,:], a2[np.newaxis,:])}")

## ACROSS WINDOWS AVERAGING HYPERPLANES
all_hyperplans_all_windows = []
for i_win, (window, window_models) in enumerate(zip(args.windows, all_windows_models)):
    all_hyperplans_all_windows.append([])
    for i_class, classe in enumerate(class_queries):
        all_hyperplans_all_windows[-1].append(np.mean([window_models[i][clf_idx].estimators_[i_class].coef_.ravel() for i in range(args.n_folds)], 0))
all_cos_sims = []
for i_c, classe in enumerate(class_queries):
    class_hyperplanes = [hyperplan[i_c] for hyperplan in all_hyperplans_all_windows]
    cos_sim = cosine_similarity(class_hyperplanes)
    print(f"(hyperplanes average) cosine_similarity: {cos_sim}")
    # print(f"\n doing windows {args.windows} for classe {classe}: {np.degrees(angle_between(*class_hyperplanes))}")
    all_cos_sims.append(cos_sim[np.triu_indices(len(cos_sim), k=1)])

ave_cos_sims = np.mean(all_cos_sims, 0)

n_conds = len(args.train_cond)
idx_conds_comb = [x for x in combinations(np.arange(n_conds), 2)]
conds_comb = [x for x in combinations(args.train_cond, 2)]
query_comb = [x for x in combinations(args.train_query, 2)]
window_comb = [x for x in combinations(args.windows, 2)]
cos_sims_results = {}
for (cond1, cond2), (query1, query2), (win1, win2), cos in zip(conds_comb, query_comb, window_comb, ave_cos_sims):
    # cos_sims_results[f"{cond1}_{query1}_{win1}-vs-{cond2}_{query2}_{win2}"] = cos
    cos_sims_results[f"{query1}-{cond1}--vs--{query2}-{cond2}"] = cos
save_pickle(f"{out_fn}_cosine.p", cos_sims_results)



print(f'Total elasped time since the script began: {(time.time()-start_time)/60:.2f}min')
