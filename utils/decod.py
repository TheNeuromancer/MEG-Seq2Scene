import mne
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import numpy as np
# from ipdb import set_trace
from glob import glob
import os.path as op
import os
from natsort import natsorted
import pandas as pd
import pickle
from copy import deepcopy
from tqdm import trange
from sklearn.preprocessing import StandardScaler, RobustScaler, label_binarize, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, LogisticRegression, LogisticRegressionCV, LinearRegression, Ridge, RidgeCV
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, permutation_test_score, GridSearchCV # StratifiedGroupKFold, 
from sklearn.utils.extmath import softmax
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from scipy.signal import savgol_filter
from scipy.stats import ttest_1samp, sem, wilcoxon
from autoreject import AutoReject
from mne.stats import permutation_cluster_1samp_test, fdr_correction


# local import
from .commons import *
from .params import *
from .angles import *
# from .split import *


cmaptab10 = plt.cm.get_cmap('tab10', 10)

"""
******   MAIN FUNCTIONS   *******
*                               *
*          _---~~~~-_.          *
*        _(        )   )        *
*      ,   ) -~~- ( ,-' )_      *
*     (  `-,_..`., )-- '_,)     *
*    ( ` _)  (  -~( -_ `,  }    *
*    (_-  _  ~_-~~~~`,  ,' )    *
*      `~ -^(    __;-,((()))    *
*           ~~~~ {_ -_(())      *
*                 `'  }         *
*                   { }         *
*                               *
*********************************
Called in the main scripts """

# ///////////////////////////////////////////////////////// #
################## PATHS AND DATA LOADING ###################
# ///////////////////////////////////////////////////////// #

def load_data(args, fn, query_1='', query_2=''):
    epochs = mne.read_epochs(fn, preload=True, verbose=False)
    if args.filter: # filter metadata before anything else
        epochs = epochs[args.filter]
        if not len(epochs): 
            print(f"No event left after filtering with {args.filter}. Exiting smoothly")
            exit()
        else:
            print(f"We have {len(epochs)} events left after filtering with {args.filter}")
    if args.autoreject and not args.dummy:
        print(f"Applying autoreject in the first place")
        ar = AutoReject()
        epochs = ar.fit_transform(epochs) 
    if args.quality_th:
        print(f"Keeping only runs with a quality score above {args.quality_th}")
        cond = full_fn_to_short[op.basename(fn).replace('-epo.fif', '')]
        score_per_run = quality_from_cond(args.subject, cond)
        runs_to_keep = [k for k, v in score_per_run.items() if v > args.quality_th]
        print(f"Keeping runs {runs_to_keep}")
        epochs = epochs[f"run_nb in {runs_to_keep}"]
    if args.xdawn and not args.dummy:
        print("TO IMPLEMENT LOADING THE RIGHT EPOCHS")
        raise
        # Xdawn
    if args.subtract_evoked:
        epochs = epochs.subtract_evoked(epochs.average())
    # if "Right_obj" in query_1 or "Left_obj" in query_1 or "two_objects" in fn:
    print(fn)
    if "two_objects-epo.fif" in fn:
        epochs.metadata = complement_md(epochs.metadata)
        epochs.metadata['Complexity'] = epochs.metadata.apply(add_complexity_to_md, axis=1)
    if "Flash" not in epochs.metadata.keys():
        epochs.metadata['Flash'] = 0 # old subject did not have flashes
    if query_1 and query_2: # load 2 sub-epochs, one for each query
        epochs = [epochs[query_1], epochs[query_2]]
    elif query_1: # load a single sub-epochs
        epochs = [epochs[query_1]]
    else: # load the whole epochs (typically for RSA)
        epochs = [epochs]
    if query_1:
        print(f"Found {[len(epo) for epo in epochs]} events for queries {query_1} and {query_2}")
    if not all([len(epo) for epo in epochs]):
        print(f"No matching event for queries {query_1} and {query_2}. Exiting smoothly")
        exit()
    epochs = [epo.pick('meg') for epo in epochs]
    initial_sfreq = epochs[0].info['sfreq']

    if args.equalize_events and len(epochs) > 1:
        print(f"Equalizing event counts: ", end='')
        n_trials = min([len(epo) for epo in epochs])
        print(f"keeping: {n_trials} events in each class")
        epochs = [epo[np.random.choice(range(len(epo)), n_trials, replace=False)] for epo in epochs]


    ### SELECT ONLY CH THAT HAD AN EFFECT IN THE LOCALIZER
    if args.localizer:
        print("not implemented yet ...")
        raise
        path2loc = glob(f'{args.root_path}/Results/{args.path2loc}/{args.subject}/*minpval_MEG.p')[0]
        pval_dict = pickle.load(open(path2loc, 'rb'))
        print(f'Keeping {sum(np.array(list(pval_dict.values()))<args.pval_thresh)} MEG channels out of {len(pval_dict)} based on localizer results\n')
        ch_to_keep = []
        for ch in pval_dict:
            if pval_dict[ch] < args.pval_thresh:
                ch_to_keep.append(ch.split('_')[2])
        epochs = [epo.pick(ch_to_keep) for epo in epochs]


    if args.freq_band:
        # freq_bands = dict(delta=(1, 3.99), theta=(4, 7.99), alpha=(8, 12.99), beta=(13, 29.99), low_gamma=(30, 69.99), high_gamma=(70, 150))
        freq_bands = dict(low_high=(0.03, 80), low_low=(0.03, 40), high_low=(2, 40), high_vlow=(2, 20), 
                                      low_vlow=(0.03, 20), low_vhigh=(0.03, 160), high_vhigh=(2, 160), 
                                      vhigh_vhigh=(20, 160), vhigh_high=(20, 80), vhigh_low=(20, 40))
        fmin, fmax = freq_bands[args.freq_band]
        print("\nFILTERING WITH FREQ BAND: ", (fmin, fmax))

        # bandpass filter
        epochs = [epo.filter(fmin, fmax, n_jobs=-1) for epo in epochs]  

    if args.sfreq < epochs[0].info['sfreq']: 
        if args.sfreq < epochs[0].info['lowpass']:
            print(f"Lowpass filtering the data at the final sampling rate, {args.sfreq}Hz")
            # epochs.filter(None, args.sfreq, l_trans_bandwidth='auto', h_trans_bandwidth='auto', filter_length='auto', phase='zero', fir_window='hamming', fir_design='firwin')
            epochs = [epo.filter(None, args.sfreq, l_trans_bandwidth='auto', h_trans_bandwidth='auto', filter_length='auto', phase='zero', fir_window='hamming', fir_design='firwin') for epo in epochs]
        print(f"starting resampling from {epochs[0].info['sfreq']} to {args.sfreq} ... ")
        epochs = [epo.resample(args.sfreq) for epo in epochs]
        print("finished resampling ... ")

    if args.dummy:
        epochs = [epo.decimate(10) for epo in epochs]

    data = [epo.data if isinstance(epo, mne.time_frequency.EpochsTFR) else epo.get_data() for epo in epochs]

    metadatas = [epo.metadata for epo in epochs]

    # # add the temporal derivative of each channel as new feature
    # X = np.concatenate([X, np.gradient(X, axis=1)], axis=1)

    if args.smooth:
        print(f"Smoothing the data with a gaussian window of size {args.smooth}")
        for query_data in data:
            for i_trial in range(len(query_data)):
                for i_ch in range(len(query_data[i_trial])):
                    query_data[i_trial, i_ch] = smooth(query_data[i_trial, i_ch], window_len=5, window='hanning')

    if args.cat:
        ### !!! concatenating does not work anymore
        data = win_ave_smooth(data, nb_cat=args.cat, mean=args.mean)
        new_info = epochs[0].info.copy()
        if not args.mean: # create new channel names because concatenation of timepoints equate adding channels
            old_ch_names = deepcopy(new_info['ch_names'])
            for i in range(args.cat-1): new_info['ch_names'] += [f"{ch}_{i}" for ch in old_ch_names]
            new_info['nchan'] = new_info['nchan'] * args.cat
            old_info_chs = deepcopy(new_info['chs'])
            for i in range(args.cat-1): new_info['chs'] += deepcopy(old_info_chs)
            for i in range(new_info['nchan']): new_info['chs'][i]['ch_name'] = new_info['ch_names'][i] 
    else:
        new_info = epochs[0].info

    # print('zscoring each epoch')
    # for idx in range(X.shape[0]):
    #     for ch in range(X.shape[1]):
    #         X[idx, ch, :] = (X[idx, ch] - np.mean(X[idx, ch])) / np.std(X[idx, ch])

    epochs = [mne.EpochsArray(datum, new_info, metadata=meta, tmin=old_epo.times[0], verbose="warning") for datum, meta, old_epo in zip(data, metadatas, epochs)]

    # crop after getting high gammas and smoothing to avoid border issues
    block_type = op.basename(fn).split("-epo.fif")[0]
    tmin, tmax = tmin_tmax_dict[block_type]
    print('initial tmin and tmax: ', [(epo.times[0], epo.times[-1]) for epo in epochs])
    print('cropping to final tmin and tmax: ', tmin, tmax)
    epochs = [epo.crop(tmin, tmax) for epo in epochs]


    # CLIP THE DATA
    if args.clip:
        print('clipping the data at the 5th and 95th percentile for each channel')
        for query_data in data:
            for ch in range(query_data.shape[1]):
                # Get the 5th and 95th percentile and channel
                p5 = np.percentile(query_data[:, ch, :], 5)
                p95 = np.percentile(query_data[:, ch, :], 95)
                query_data[:, ch, :] = np.clip(query_data[:, ch, :], p5, p95)

    ### BASELINING
    if args.baseline:
        print('baselining...')
        epochs = [epo.apply_baseline((tmin, 0), verbose=0) for epo in epochs]

    # CLIP THE DATA
    if args.clip:
        print('clipping the data at the 5th and 95th percentile for each channel')
        for query_data in data:
            for ch in range(query_data.shape[1]):
                # Get the 5th and 95th percentile and channel
                p5 = np.percentile(query_data[:, ch, :], 5)
                p95 = np.percentile(query_data[:, ch, :], 95)
                query_data[:, ch, :] = np.clip(query_data[:, ch, :], p5, p95)

    return epochs
    

def get_class_queries(query):
    if query == "Colour1": 
        class_queries = [f"Colour1=='{c}'" for c in colors]
    elif query == "Shape1":
        class_queries = [f"Shape1=='{s}'" for s in shapes]
    elif query == "Colour2": 
        class_queries = [f"Colour2=='{c}'" for c in colors]
    elif query == "Shape2":
        class_queries = [f"Shape2=='{s}'" for s in shapes]
    elif query == "Shape1+Colour1":
        class_queries = [f"Shape1=='{s}' and Colour1=='{c}'" for s in shapes for c in colors]
    elif query == "Shape2+Colour2":
        class_queries = [f"Shape2=='{s}' and Colour2=='{c}'" for s in shapes for c in colors]
    elif "Left" in query or "Right" in query:
        class_queries = [f"{query}=='{s}_{c}'" for s in shapes for c in colors]
    elif query == "Relation": 
        class_queries = ["Relation==\"à gauche d'\"", "Relation==\"à droite d'\""]
    elif query == "Perf":
        class_queries = ["Perf==0", "Perf==1"]
    elif query == "Matching":
        class_queries = ["Matching=='match'", "Matching=='nonmatch'"]
    elif query == "Button":
        class_queries = ["Button=='left'", "Button=='right'"]
    elif query == "Flash":
        class_queries = ["Flash==0", "Flash==1"]
    elif query == "ColourMismatch": # single obj mismatch
        class_queries = ["Matching=='match'", "Error_type=='colour'"]
    elif query == "ShapeMismatch": # single obj mismatch
        class_queries = ["Matching=='match'", "Error_type=='shape'"]
    elif query == "PropMismatch": # scene mismatch
        class_queries = ["Matching=='match'", "Error_type=='l0'"]
    elif query == "BindMismatch": # scene mismatch
        class_queries = ["Matching=='match'", "Error_type=='l1'"]
    elif query == "RelMismatch": # scene mismatch
        class_queries = ["Matching=='match'", "Error_type=='l2'"]
    elif query == "Mismatches": # scene mismatches
        class_queries = ["Error_type=='l0'", "Error_type=='l1'", "Error_type=='l2'"]
    elif query == "SameShape": 
        class_queries = ["Shape1==Shape2 and Colour1!=Colour2", "Shape1!=Shape2 and Colour1!=Colour2"]
    elif query == "SameColour": 
        class_queries = ["Colour1==Colour2 and Shape1!=Shape2", "Colour1!=Colour2 and Shape1!=Shape2"]
    elif query == "SameObject": 
        class_queries = ["Colour1==Colour2 and Shape1==Shape2", "Colour1!=Colour2 and Shape1!=Shape2"]
    elif query == "Complexity":  # for regression decoding
        class_queries = ["Complexity==0", "Complexity==1", "Complexity==2"]
    else:
        raise RuntimeError(f"Wrong query: {query}")
    print(class_queries)
    return class_queries


# ///////////////////////////////////////////////////////// #
###################### DECODING PROPER ######################
# ///////////////////////////////////////////////////////// #

def get_averaged_clf(args, all_models, n_times):
    # all_models = n_folds*n_times
    # average classifier parameters over all crossval splits, works only for logreg
    

    avg_models = [[]] #[[deepcopy(all_models[0][t]) for t in range]
    for t in range(n_times):
        scalers = []
        clfs = []
        for i in range(args.n_folds):
            scalers.append(all_models[i][t][0])
            clfs.append(all_models[i][t][1])

        avg_models[0].append(deepcopy(all_models[0][t]))
        # scaler
        avg_models[0][-1][0].center_ = np.mean([scaler.center_ for scaler in scalers])
        avg_models[0][-1][0].scale_ = np.mean([scaler.scale_ for scaler in scalers])
        # clf
        avg_models[0][-1][1].coef_ = np.mean([clf.coef_ for clf in clfs])
        avg_models[0][-1][1].intercept_ = np.mean([clf.intercept_ for clf in clfs])

    # discard the model for each split
    # averaged_models = all_models[:,0]
    return avg_models


def decode(args, X, y, clf, n_times, test_split_query_indices):
    
    if args.dummy:
        cv = StratifiedKFold(n_splits=2, shuffle=False)
    else:
        cv = get_cv(args.train_cond, args.crossval, args.n_folds)
    
    if args.reduc_dim:
        pipeline = make_pipeline(RobustScaler(), PCA(args.reduc_dim), clf)
    else:
        pipeline = make_pipeline(RobustScaler(), clf)

    all_models = []
    if args.timegen:
        AUC = np.zeros((n_times, n_times))
        accuracy = np.zeros((n_times, n_times))
        mean_preds = np.zeros((n_times, n_times))
        if test_split_query_indices: # split the test indices according to the query
            AUC_test_query_split = np.zeros((n_times, n_times, len(test_split_query_indices)))
            AUC_test_query_counts = np.zeros((n_times, n_times, len(test_split_query_indices)))
            mean_preds_test_query_split = np.zeros((n_times, n_times, len(test_split_query_indices)))
        else:
            AUC_test_query_split = None
            AUC_test_query_counts = None
            mean_preds_test_query_split = None

        for train, test in cv.split(X, y):
            all_models.append([])
            for t in trange(n_times):
                pipeline.fit(X[train, :, t], y[train])
                all_models[-1].append(deepcopy(pipeline))
                for tgen in range(n_times):
                    
                    # normal test
                    pred = predict(pipeline, X[test, :, tgen])
                    AUC[t, tgen] += roc_auc_score(y_true=y[test], y_score=pred) / args.n_folds
                    mean_preds[t, tgen] += np.mean(pred) / args.n_folds
                    # accuracy[t, tgen] += accuracy_score(y[test], pred>0.5) / args.n_folds

                    if test_split_query_indices: # split the test indices according to the query
                        for i_query, split_indices in enumerate(test_split_query_indices):
                            test_query = test[np.isin(test, split_indices)]
                            if len(np.unique(y[test_query])) < 2: # not enough classes to compute AUC, just pass
                                continue
                            if args.equalize_split_events: # keep the same number of events for each classes
                                split_classes, split_counts = np.unique(y[test_query], return_counts=True)
                                if not np.all(split_counts[0] == split_counts): # subsample majority class
                                    nb_ev = split_counts.min()
                                    picked_all_classes = []
                                    for clas in split_classes:
                                        candidates = test_query[np.where(y[test_query]==clas)[0]]
                                        picked = np.random.choice(candidates, size=nb_ev, replace=False)
                                        picked_all_classes.extend(picked)
                                    test_query = np.array(picked_all_classes)
                            pred = predict(pipeline, X[test_query, :, tgen])
                            AUC_test_query_split[t, tgen, i_query] += roc_auc_score(y_true=y[test_query], y_score=pred) #/ args.n_folds
                            AUC_test_query_counts[t, tgen, i_query] += 1 # keep track of the number of AUC computed (should = n_folds)
        if test_split_query_indices: AUC_test_query_split = AUC_test_query_split / AUC_test_query_counts # replace division by n_folds because for many cases we don't have correct test indices in each fold.
                        
    # TDO: implement generalization over conditions for the non-timegen case
    else:
        AUC_test_query_split = None
        AUC = np.zeros(n_times)
        mean_preds = np.zeros(n_times)
        AUC_test_query_split = None
        mean_preds_test_query_split = None
        for train, test in cv.split(X, y):
            all_models.append([])
            for t in trange(n_times):
                pipeline.fit(X[train, :, t], y[train])
                all_models[-1].append(deepcopy(pipeline))
                pred = predict(pipeline, X[test, :, t])
                AUC[t] += roc_auc_score(y_true=y[test], y_score=pred) / args.n_folds
                mean_preds[t] += np.mean(pred) / args.n_folds

    print(f'mean trainning AUC: {AUC.mean():.3f}')
    print(f'max trainning AUC: {AUC.max():.3f}')

    if args.avg_clf: # average classifier parameters over all crossval splits
        if args.reduc_dim:
            print('cannot average classifier weights when using PCA because there can \
                be a different nb of selected components for each fold .. .exiting')
            raise
        all_models = get_averaged_clf(args, all_models, n_times)

    # put the pipeline object in an array without unpacking them
    all_models_array = np.empty((len(all_models), len(all_models[0])), dtype=object)
    all_models_array[:] = all_models # n_folds, n_times
    all_models_array = all_models_array.transpose(1,0) # go to shape n_times * n_folds

    return all_models_array, AUC, accuracy, AUC_test_query_split, mean_preds, mean_preds_test_query_split


def test_decode(args, X, y, all_models, test_split_query_indices):
    n_times_train, n_folds = all_models.shape
    n_times_test = X.shape[2]

    if args.timegen:
        AUC = np.zeros((n_times_train, n_times_test))
        mean_preds = np.zeros((n_times_train, n_times_test))
        accuracy = np.zeros((n_times_train, n_times_test))
        if test_split_query_indices: # split the test indices according to the query
            AUC_test_query_split = np.zeros((n_times_train, n_times_test, len(test_split_query_indices)))
        else:
            AUC_test_query_split = None
        for tgen in trange(n_times_test):
            t_data = X[:, :, tgen]
            for t in range(n_times_train):
                all_folds_preds = []
                for i_fold in range(n_folds):
                    pipeline = all_models[t, i_fold]
                    all_folds_preds.append(predict(pipeline, t_data))

                mean_folds_preds = np.mean(all_folds_preds, 0)
                AUC[t, tgen] = roc_auc_score(y_true=y, y_score=mean_folds_preds)
                mean_preds[t, tgen] = np.mean(mean_folds_preds)
                accuracy[t, tgen] += accuracy_score(y, mean_folds_preds>0.5)

                if test_split_query_indices: # split the test indices according to the query
                    for i_query, split_indices in enumerate(test_split_query_indices):
                        if len(np.unique(y[split_indices])) < 2: # not enough classes to compute AUC, just pass
                            continue
                        AUC_test_query_split[t, tgen, i_query] = roc_auc_score(y_true=y[split_indices], y_score=mean_folds_preds[split_indices]) #/ args.n_folds

    else: # diag only
        AUC = np.zeros(n_times_train)
        mean_preds = np.zeros(n_times_train)
        for t in range(n_times_train):
            t_data = X[:, :, t]
            all_folds_preds = []
            for i_fold in range(n_folds):
                pipeline = all_models[i_fold][t]
                all_folds_preds.append(predict(pipeline, t_data))
            AUC[t] = roc_auc_score(y_true=y, y_score=np.mean(all_folds_preds, 0))
            mean_preds[t] = np.mean(all_folds_preds)

    print(f'mean test AUC: {AUC.mean():.3f}')
    print(f'max test AUC: {AUC.max():.3f}')

    return AUC, accuracy, mean_preds, AUC_test_query_split



## OVR AND WINDOW DECODING


def decode_window(args, clf, epochs, class_queries):
    """ train single decoder for the whole time of the epochs
        class_queries: list of strings, pandas queries to get each class
    """
    X, y, groups, _ = get_X_y_from_queries(epochs, class_queries)
    n_trials = len(X)
    X = X.reshape((n_trials,-1)) # concatenate timepoint of the window
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    print(f"n_classes: {n_classes}, classes: {classes}, counts: {counts}")
    
    cv = get_cv(args.train_cond, args.crossval_win, args.n_folds_win)
    
    onehotenc = OneHotEncoder(sparse=False, categories='auto')
    onehotenc = onehotenc.fit(np.arange(n_classes).reshape(-1,1))

    if args.reduc_dim_win:
        pipeline = make_pipeline(RobustScaler(), PCA(args.reduc_dim_win), clf)
    else:
        pipeline = make_pipeline(RobustScaler(), clf)

    models_all_folds = []
    AUC, accuracy = 0, 0
    for train, test in cv.split(X, y, groups=groups): # groups is ignored for non-groupedKFold
        pipeline.fit(X[train], y[train])
        models_all_folds.append(deepcopy(pipeline))

        preds = predict(pipeline, X[test], multiclass=True)
        if preds.ndim == 2:
            preds = preds
        else:
            preds = onehotenc.transform(preds.reshape((-1,1)))
        
        AUC += roc_auc_score(y_true=y[test], y_score=preds, multi_class='ovr', average='weighted') / args.n_folds_win
        accuracy += accuracy_score(y[test], preds.argmax(1)) / args.n_folds_win

    return AUC, accuracy, models_all_folds


def test_decode_window(args, epochs, class_queries, trained_models):
    """ X: n_trials, n_sensors
        y: n_trials
        class_queries: list of strings, pandas queries to get each class
        trained_models: list of sklearn estimators, one for each class
    """
    X, y, groups, _ = get_X_y_from_queries(epochs, class_queries)
    n_trials = len(X)
    X = X.reshape((n_trials,-1)) # concatenate timepoint of the window
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    print(f"n_classes: {n_classes}, classes: {classes}, counts: {counts}")
    
    onehotenc = OneHotEncoder(sparse=False, categories='auto')
    onehotenc = onehotenc.fit(np.arange(n_classes).reshape(-1,1))
    AUC, accuracy = 0, 0
    for i_fold in range(args.n_folds_win):
        pipeline = trained_models[i_fold]
        preds = predict(pipeline, X, multiclass=True)
        if preds.ndim == 2:
            preds = preds
        else:
            preds = onehotenc.transform(preds.reshape((-1,1)))
        
        AUC += roc_auc_score(y_true=y, y_score=preds, multi_class='ovr', average='weighted') / args.n_folds_win
        accuracy += accuracy_score(y, preds.argmax(1)) / args.n_folds_win
    # ## AVERAGE PREDICTIONS OR PERFORMANCE? usually perf is better (for training at least)
    # all_folds_preds.append(y_pred)
    # mean_fold_pred = np.mean(all_folds_preds, 0)
    # AUC[t, tgen] = roc_auc_score(y_true=y, y_score=mean_fold_pred, multi_class='ovr')                

    return AUC, accuracy


def test_decode_sliding_window(args, epochs, class_queries, trained_models, nb_cat):
    """ X: n_trials, n_sensors, n_times
        y: n_trials
        class_queries: list of strings, pandas queries to get each class
        trained_models: list of sklearn estimators, one for each class
    """
    X, y, groups, _ = get_X_y_from_queries(epochs, class_queries)
    n_times = X.shape[2]
    X = win_ave_smooth(X, nb_cat, mean=False)[0]
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    print(f"n_classes: {n_classes}, classes: {classes}, counts: {counts}")
    
    onehotenc = OneHotEncoder(sparse=False, categories='auto')
    onehotenc = onehotenc.fit(np.arange(n_classes).reshape(-1,1))
       
    AUC = np.zeros(n_times)
    accuracy = np.zeros(n_times)
#    mean_preds = np.zeros(n_times)
    for t in range(n_times):
        t_data = X[:, :, t]
        all_folds_preds = []
        for i_fold in range(args.n_folds_win):
            pipeline = trained_models[i_fold]
            preds = predict(pipeline, t_data, multiclass=True)
            if preds.ndim == 2:
                preds = preds
            else:
                preds = onehotenc.transform(preds.reshape((-1,1)))

            AUC[t] += roc_auc_score(y_true=y, y_score=preds, multi_class='ovr', average='weighted') / args.n_folds_win
            accuracy[t] += accuracy_score(y, preds.argmax(1)) / args.n_folds_win
                
    return AUC, accuracy


def decode_ovr(args, clf, epochs, class_queries):
    """ X: n_trials, n_sensors, n_times
        y: n_trials
        class_queries: list of strings, pandas queries to get each class
    """
    n_times = len(epochs.times)
    if args.equalize_events:
        epochs = equalize_events_single_epo(epochs, class_queries)
    X, y, groups, test_split_query_indices = get_X_y_from_queries(epochs, class_queries, args.split_queries)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    print(f"n_classes: {n_classes}, classes: {classes}, counts: {counts}")
    if n_classes < 2:
        raise RuntimeError(f"did not find enough classes for queries {class_queries} and subjects {args.subject}")
    if np.min(counts) < args.n_folds:
        print(f"that's too few trials ... decreasing n_folds to {np.min(counts)}")
        setattr(args, 'n_folds', np.min(counts))

    cv = get_cv(args.train_cond, args.crossval, args.n_folds)

    onehotenc = OneHotEncoder(sparse=False, categories='auto')
    onehotenc = onehotenc.fit(np.arange(n_classes).reshape(-1,1))

    if args.reduc_dim:
        pipeline = make_pipeline(RobustScaler(), PCA(args.reduc_dim), clf)
    else:
        pipeline = make_pipeline(RobustScaler(), clf)

    accuracy = None
    AUC_test_query_split = np.zeros((n_times, n_times, len(test_split_query_indices))) if test_split_query_indices else None
    AUC_test_query_counts = np.zeros((n_times, n_times, len(test_split_query_indices))) if test_split_query_indices else None
    all_models = []
    if args.timegen:
        AUC = np.zeros((n_times, n_times))
        all_confusions = []
        for t in trange(n_times):
            all_models.append([])
            for train, test in cv.split(X, y, groups=groups): # groups is ignored for non-groupedKFold
                pipeline.fit(X[train, :, t], y[train])
                all_models[-1].append(deepcopy(pipeline))
                for tgen in range(n_times):
                    preds = predict(pipeline, X[test, :, tgen], multiclass=True)
                    preds = preds if preds.ndim == 2 else onehotenc.transform(preds.reshape((-1,1)))
                    if n_classes == 2: preds = preds[:,1] # not a proper OVR object, needs different method
                    if np.any(np.isnan(preds)):
                        print(f"nan in preds, probably due to lack of convergence of classifier, moving to next fold")
                        continue
                    AUC[t, tgen] += roc_auc_score(y_true=y[test], y_score=preds, multi_class='ovr', average='weighted') / args.n_folds
                    # if not n_classes == 2: # then single set of probabilities...
                    #     accuracy[t, tgen] += accuracy_score(y[test], preds.argmax(1)) / args.n_folds
                    if test_split_query_indices: # split the test indices according to the query
                        for i_query, split_indices in enumerate(test_split_query_indices):
                            test_query = test[np.isin(test, split_indices)]
                            if len(np.unique(y[test_query])) < n_classes: # not enough classes to compute AUC, just pass
                                continue
                            if args.equalize_split_events: # keep the same number of events for each classes
                                split_classes, split_counts = np.unique(y[test_query], return_counts=True)
                                if not np.all(split_counts[0] == split_counts): # subsample majority class
                                    nb_ev = split_counts.min()
                                    picked_all_classes = []
                                    for clas in split_classes:
                                        candidates = test_query[np.where(y[test_query]==clas)[0]]
                                        picked = np.random.choice(candidates, size=nb_ev, replace=False)
                                        picked_all_classes.extend(picked)
                                    test_query = np.array(picked_all_classes)
                            pred = predict(pipeline, X[test_query, :, tgen], multiclass=True)
                            pred = pred if pred.ndim == 2 else onehotenc.transform(pred.reshape((-1,1)))
                            if n_classes == 2: 
                                pred = pred[:,1] # not a proper OVR object, needs different method
                            AUC_test_query_split[t, tgen, i_query] += roc_auc_score(y_true=y[test_query], y_score=pred, multi_class='ovr', average='weighted') #/ args.n_folds
                            AUC_test_query_counts[t, tgen, i_query] += 1 # keep track of the number of AUC computed (should = n_folds)
        if test_split_query_indices: AUC_test_query_split = AUC_test_query_split / AUC_test_query_counts # replace division by n_folds because for many cases we don't have correct test indices in each fold.
    else:
        AUC = np.zeros(n_times)
        for t in trange(n_times):
            all_models.append([])
            for train, test in cv.split(X, y, groups=groups): # groups is ignored for non-groupedKFold
                pipeline.fit(X[train, :, t], y[train])
                all_models[-1].append(deepcopy(pipeline))
                preds = predict(pipeline, X[test, :, t], multiclass=True)
                preds = preds if preds.ndim == 2 else onehotenc.transform(preds.reshape((-1,1)))
                if n_classes == 2: preds = preds[:,1] # not a proper OVR object, needs different method
                AUC[t] += roc_auc_score(y_true=y[test], y_score=preds, multi_class='ovr', average='weighted') / args.n_folds

    # put the pipeline object in an array without unpacking them
    all_models_array = np.empty((len(all_models), len(all_models[0])), dtype=object)
    all_models_array[:] = all_models # n_times, n_folds

    print(f'mean training AUC: {AUC.mean():.3f}')
    print(f'max training AUC: {AUC.max():.3f}')
    return AUC, accuracy, all_models_array, AUC_test_query_split


def test_decode_ovr(args, epochs, class_queries, all_models):
    n_times_test = len(epochs.times)
    n_times_train, n_folds = all_models.shape

    X, y, _, test_split_query_indices = get_X_y_from_queries(epochs, class_queries, args.split_queries)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    if n_classes < 2:
        raise RuntimeError(f"did not find enough classes for queries {class_queries} and subjects {args.subject}")

    onehotenc = OneHotEncoder(sparse=False, categories='auto')
    onehotenc = onehotenc.fit(np.arange(n_classes).reshape(-1,1))

    # full of nans because here we replace the values (only once), whereas in the train we count  the occurences where we could manage to do the split queries, then divide by this. 
    if args.timegen:
        AUC_test_query_split = np.full((n_times_train, n_times_test, len(test_split_query_indices)), np.nan) if test_split_query_indices else None    
        AUC = np.zeros((n_times_train, n_times_test))
        accuracy = np.zeros((n_times_train, n_times_test))
        for tgen in trange(n_times_test):
            t_data = X[:, :, tgen]
            for t in range(n_times_train):
                all_folds_preds = []
                for i_fold in range(n_folds):
                    pipeline = all_models[t][i_fold]
                    preds = predict(pipeline, t_data, multiclass=True)
                    y_pred = preds if preds.ndim == 2 else onehotenc.transform(preds.reshape((-1,1)))
                    all_folds_preds.append(y_pred)
                mean_fold_pred = np.mean(all_folds_preds, 0)
                if n_classes == 2: mean_fold_pred = mean_fold_pred[:,1] # not a proper OVR object, needs different method
                AUC[t, tgen] = roc_auc_score(y_true=y, y_score=mean_fold_pred, multi_class='ovr')
                # accuracy[t, tgen] = accuracy_score(y, mean_fold_pred.argmax(1)) # dim error when n_classes = 2

                if test_split_query_indices: # split the test indices according to the query
                    for i_query, split_indices in enumerate(test_split_query_indices):
                        if len(np.unique(y[split_indices])) < n_classes: # not enough classes to compute AUC, just pass
                            continue
                        AUC_test_query_split[t, tgen, i_query] = roc_auc_score(y_true=y[split_indices], y_score=mean_fold_pred[split_indices], multi_class='ovr')
    else:
        raise NotImplementedError
    print(f'mean test AUC: {AUC.mean():.3f}')
    print(f'max test AUC: {AUC.max():.3f}')

    return AUC, accuracy, AUC_test_query_split



def decode_single_ch_ovr(args, clf, epochs, class_queries):
    """ X: n_trials, n_sensors, n_times
        y: n_trials
        class_queries: list of strings, pandas queries to get each class
    """
    if args.equalize_events:
        epochs = equalize_events_single_epo(epochs, class_queries)
    X, y, groups, _ = get_X_y_from_queries(epochs, class_queries, split_queries=[])
    nchan = X.shape[1]
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    print(f"n_classes: {n_classes}, classes: {classes}, counts: {counts}")
    if n_classes < 2:
        raise RuntimeError(f"did not find enough classes for queries {class_queries} and subjects {args.subject}")
    if np.min(counts) < args.n_folds:
        print(f"that's too few trials ... decreasing n_folds to {np.min(counts)}")
        setattr(args, 'n_folds', np.min(counts))

    cv = get_cv(args.train_cond, args.crossval, args.n_folds)

    onehotenc = OneHotEncoder(sparse=False, categories='auto')
    onehotenc = onehotenc.fit(np.arange(n_classes).reshape(-1,1))

    if args.reduc_dim_sing:
        pipeline = make_pipeline(RobustScaler(), PCA(args.reduc_dim_sing), clf)
    else:
        pipeline = make_pipeline(RobustScaler(), clf)

    all_models = []
    AUC = np.zeros(nchan)
    accuracy = np.zeros(nchan)
    for ch in trange(nchan):
        all_models.append([])
        for train, test in cv.split(X, y, groups=groups): # groups is ignored for non-groupedKFold
            pipeline.fit(X[train, ch, :], y[train])
            all_models[-1].append(deepcopy(pipeline))
            preds = predict(pipeline, X[test, ch, :], multiclass=True)
            if preds.ndim == 2:
                preds = preds
            else:
                preds = onehotenc.transform(preds.reshape((-1,1)))
            if n_classes == 2: preds = preds[:,1]
            AUC[ch] += roc_auc_score(y_true=y[test], y_score=preds, multi_class='ovr', average='weighted') / args.n_folds
            # accuracy[ch] = pipeline.score(X[test, ch, :], y[test])
            # if not n_classes == 2: # then single set of probabilities...
            #     accuracy[ch] += accuracy_score(y[test], preds.argmax(1)) / args.n_folds
    print(f'mean AUC: {AUC.mean():.3f}')
    print(f'max AUC: {AUC.max():.3f}')
    return AUC, accuracy, all_models


def test_decode_single_ch_ovr(args, epochs, class_queries, all_models):
    n_folds = len(all_models[0])
    nchan = len(all_models)
    X, y, _, _ = get_X_y_from_queries(epochs, class_queries, split_queries=[])
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    print(f"n_classes: {n_classes}, classes: {classes}, counts: {counts}")
    onehotenc = OneHotEncoder(sparse=False, categories='auto')
    onehotenc = onehotenc.fit(np.arange(n_classes).reshape(-1,1))

    AUC = np.zeros(nchan)
    accuracy = np.zeros(nchan)
    for ch in range(nchan):
        ch_data = X[:, ch, :]
        all_folds_preds = []
        for i_fold in range(n_folds):
            pipeline = all_models[ch][i_fold]
            preds = predict(pipeline, ch_data, multiclass=True)
            if preds.ndim == 2:
                preds = preds
            else:
                preds = onehotenc.transform(preds.reshape((-1,1)))
            all_folds_preds.append(preds)
        mean_fold_pred = np.mean(all_folds_preds, 0)
        AUC[ch] = roc_auc_score(y_true=y, y_score=mean_fold_pred, multi_class='ovr')
        accuracy[ch] = accuracy_score(y, mean_fold_pred.argmax(1))
    print(f'mean AUC: {AUC.mean():.3f}')
    print(f'max AUC: {AUC.max():.3f}')
    return AUC, accuracy


def regression_decode(args, epochs, class_queries, clf):
    n_times = len(epochs.times)
    if args.equalize_events:
        epochs = equalize_events_single_epo(epochs, class_queries)
    X, y, groups, test_split_query_indices = get_X_y_from_queries(epochs, class_queries, args.split_queries)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    print(f"n_classes: {n_classes}, classes: {classes}, counts: {counts}")
    if n_classes < 2:
        raise RuntimeError(f"did not find enough classes for queries {class_queries} and subjects {args.subject}")
    if np.min(counts) < args.n_folds:
        print(f"that's too few trials ... decreasing n_folds to {np.min(counts)}")
        setattr(args, 'n_folds', np.min(counts))

    if args.dummy:
        cv = StratifiedKFold(n_splits=2, shuffle=False)
    else:
        cv = get_cv(args.train_cond, args.crossval, args.n_folds)

    pipeline = make_pipeline(RobustScaler(), PCA(args.reduc_dim), clf) if args.reduc_dim else make_pipeline(RobustScaler(), clf)
    all_models = []
    if args.timegen:
        R2 = np.zeros((n_times, n_times))
        for train, test in cv.split(X, y):
            all_models.append([])
            for t in trange(n_times):
                pipeline.fit(X[train, :, t], y[train])
                all_models[-1].append(deepcopy(pipeline))
                for tgen in range(n_times):
                    R2[t, tgen] = pipeline.score(X[test, :, tgen], y[test]) # normal test
                    # pred = predict(pipeline, X[test, :, tgen])
                    # R2[t, tgen] += roc_auc_score(y_true=y[test], y_score=pred) / args.n_folds
                    # mean_preds[t, tgen] += np.mean(pred) / args.n_folds
                    # accuracy[t, tgen] += accuracy_score(y[test], pred>0.5) / args.n_folds  
    print(f'mean trainning R2: {R2.mean():.3f}')
    print(f'max trainning R2: {R2.max():.3f}')
    # put the pipeline object in an array without unpacking them
    all_models_array = np.empty((len(all_models), len(all_models[0])), dtype=object)
    all_models_array[:] = all_models # n_folds, n_times
    all_models_array = all_models_array.transpose(1,0) # go to shape n_times * n_folds
    return all_models_array, R2


### FROM THE MARSEILLE SCRIPT. NOT USED YET HERE.
# def test_decode_all_sentence(args, X, y, all_models):
#     X = get_sentence_representation(args, X)
    
#     all_folds_preds = []
#     for i_fold in range(args.n_folds):
#         pipeline = all_models[i_fold]
#         all_folds_preds.append(predict(pipeline, X))
#     AUC = roc_auc_score(y_true=y, y_score=np.mean(all_folds_preds, 0))
#     mean_preds = np.mean(all_folds_preds)

#     print('test AUC: ', AUC)
#     return AUC, mean_preds

# def decode_single_chan(args, X, y, clf):
#     cv = StratifiedKFold(n_splits=args.n_folds, shuffle=False) # do not shuffle here!! 
#     # that is because we stored the indices of the queries we want to split at test time.
#     # plus we shuffle during the data loading
#     if args.reduc_dim:
#         pipeline = make_pipeline(RobustScaler(), PCA(args.reduc_dim), clf)
#     else:
#         pipeline = make_pipeline(RobustScaler(), clf)

#     nchan = X.shape[1]
#     all_models = []
#     AUC = np.zeros((nchan))
#     for train, test in cv.split(X, y):
#         all_models.append([])
#         for ch in trange(nchan):
#             ch_data = X[train, ch, :]
#             pipeline.fit(ch_data, y[train])
#             all_models[-1].append(deepcopy(pipeline))
#             pred = predict(pipeline, X[test, ch, :])
#             AUC[ch] += roc_auc_score(y_true=y[test], y_score=pred) / args.n_folds

#     return all_models, AUC

# def test_decode_single_chan(args, X, y, all_models):    
#     nchan = X.shape[1]
#     AUC = np.zeros(nchan)
#     for ch in range(nchan):
#         ch_data = X[:, ch, :]
#         all_folds_preds = []
#         for i_fold in range(args.n_folds):
#             pipeline = all_models[i_fold][ch]
#             all_folds_preds.append(predict(pipeline, ch_data))
#         AUC[ch] = roc_auc_score(y_true=y, y_score=np.mean(all_folds_preds, 0))

#     return AUC


# def permutation_test_single_chan(args, X, y, clf):
#     """
#     Only returns the score and pval, for now no way to get the trained models and generalize them to new conditions
#     """
#     cv = StratifiedKFold(n_splits=args.n_folds, shuffle=False) # do not shuffle here!! 
#     # that is because we stored the indices of the queries we want to split at test time.
#     # plus we shuffle during the data loading
#     if args.reduc_dim:
#         pipeline = make_pipeline(RobustScaler(), PCA(args.reduc_dim), clf)
#     else:
#         pipeline = make_pipeline(RobustScaler(), clf)

#     nchan = X.shape[1]
#     pvals = np.zeros((nchan))
#     AUC = np.zeros((nchan))
#     for ch in trange(nchan):
#         score, perm_scores, pvalue = permutation_test_score(pipeline, X[:, ch, :], y, scoring="roc_auc", 
#                                                         cv=cv, n_permutations=args.n_perm, n_jobs=-2)
#         AUC[ch] = score
#         pvals[ch] = pvalue

#     return pvals, AUC



# ///////////////////////////////////////////////////////// #
#################### SAVING AND PLOTTING ####################
# ///////////////////////////////////////////////////////// #


def save_results(out_fn, results, time=True, all_models=None, fn_end="AUC"):
    """ Generic results saving to .npy func """
    print('Saving results')
    if results.ndim > 1 or not time:
        np.save(f"{out_fn}_{fn_end}.npy", results)
    else:
        np.save(f"{out_fn}_{fn_end}_diag.npy", results)
    if all_models:
        pickle.dump(all_models, open(out_fn + '_all_models.p', 'wb'))
    return

def save_best_pattern(out_fn, AUC, all_models):
    """ Get the time of best performance and
    save the corresponding model's pattern, 
    after applying the scaler's inverse transform (as it should)
    Problem: it might not be the same timepoint across subjects...
    """
    if AUC.ndim > 1:
        AUC = np.diag(AUC)
    best_tp = np.argmax(AUC)
    pipeline_each_fold = [all_models[best_tp][i] for i in range(len(all_models[0]))] # all_models is n_times * n_folds
    # pattern_each_fold = [mne.decoding.get_coef(p, attr='patterns_', inverse_transform=True) for p in pipeline_each_fold]
    pattern_each_fold = [p[0].inverse_transform(np.atleast_2d(p[-1].patterns_)) for p in pipeline_each_fold] # atleast_2d because for classical decoding it will be 1d, but 2d for ovr, and scaler expects 2d
    pattern = np.mean(pattern_each_fold, 0)
    np.save(f"{out_fn}_best_pattern_t{best_tp}.npy", pattern)


def save_pickle(out_fn, results):
    """ Generic results to .p saving func """
    print(f'Saving results to {out_fn}')
    pickle.dump(results, open(out_fn, 'wb'))


def save_preds(args, out_fn, preds):
    print('Saving predictions')
    if args.timegen:
        preds_diag = np.diag(preds)
        np.save(out_fn + '_preds.npy', preds)
    else:
        preds_diag = preds
        np.save(out_fn + '_preds_diag.npy', preds_diag)
    return


def save_patterns(args, out_fn, all_models): ## depecated, check dimensionality of all_models
    print('Saving patterns')
    n_folds = len(all_models)
    n_times = len(all_models[0])
    n_chan = len(all_models[0][0][-1].patterns_) # [-1] because the clf is the last in the pipeline

    # average patterns? You probably ill never have to analyse non-averaged patterns ...
    # aze = list(map(list, zip(*l)))
    patterns = np.zeros((n_folds, n_times, n_chan))
    for i_f in range(n_folds):
        for i_t in range(n_times):
            patterns[i_f, i_t] = all_models[i_f][i_t][-1].patterns_
    np.save(out_fn + '_patterns.npy', patterns)


def plot_perf(args, out_fn, data_mean, train_cond, train_tmin, train_tmax, test_tmin, test_tmax, ylabel="AUC", contrast=False, gen_cond=None, version="v1", window=False):
    """ plot performance of individual subject,
    called during training by decoding.py script
    """
    if gen_cond is None or args.windows:
        plot_diag(data_mean=data_mean, data_std=None, out_fn=out_fn, train_cond=train_cond, 
            train_tmin=train_tmin, train_tmax=train_tmax, ylabel=ylabel, contrast=contrast, version=version, window=window)

    if args.timegen:
        plot_GAT(data_mean=data_mean, out_fn=out_fn, train_cond=train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, 
                 test_tmax=test_tmax, ylabel=ylabel, contrast=contrast, gen_cond=gen_cond, slices=[], version=version, window=window)
    return


def plot_diag(data_mean, out_fn, train_cond, train_tmin, train_tmax, data_std=None, ylabel="AUC", contrast=False, version="v1", window=False):
    n_times_train = data_mean.shape[0]
    times_train = np.linspace(train_tmin, train_tmax, n_times_train)
    if window: # Decoding inside subwindow
        word_onsets, image_onset = [], []
    else:
        word_onsets, image_onset = get_onsets(train_cond, version=version)

    # DIAGONAL PLOT
    fig, ax = plt.subplots()
    for w_onset in word_onsets:
         fig.axes[0].axvline(x=w_onset, linestyle='--', color='k')
    for img_onset in image_onset:
         fig.axes[0].axvline(x=img_onset, linestyle='-', color='k')
    fig.axes[0].axhline(y=0.5, color='k', linestyle='-', alpha=.5)
    
    if data_mean.ndim > 1: # we have the full timegen
        data_mean_diag = np.diag(data_mean)
    else: # already have the diagonal
        data_mean_diag = data_mean
    plot = plt.plot(times_train, data_mean_diag)
    if data_std is not None:
        if data_std.ndim > 1: # we have the full timegen
            data_std_diag = np.diag(data_std)
        else: # already have the diagonal
            data_std_diag = data_std
        ax.fill_between(times_train, data_mean_diag-data_std_diag, data_mean_diag+data_std_diag, alpha=0.2)
    plt.ylabel(ylabel)
    plt.xlabel("Time (s)")

    plt.savefig(f'{out_fn}_{ylabel}_diag.png')
    plt.close()


def plot_multi_diag(data, out_fn, train_cond, train_tmin, train_tmax, data_std=None, ylabel="AUC", contrast=False, version="v1", cmap_name='hsv', cmap_groups=[], labels=[]):
    word_onsets, image_onset = get_onsets(train_cond, version=version)
    n_plots = data.shape[0]
    n_times_train = data.shape[1]
    times_train = np.linspace(train_tmin, train_tmax, n_times_train)
    if len(cmap_groups): # give the same color for each member of the group, typically for each subjects
        cmap = plt.cm.get_cmap(cmap_name, len(np.unique(cmap_groups)))
    else: # classical 1 color per plot
        cmap = plt.cm.get_cmap(cmap_name, n_plots)

    # DIAGONAL PLOT
    fig, ax = plt.subplots()
    for w_onset in word_onsets:
         fig.axes[0].axvline(x=w_onset, linestyle='--', color='k')
    for img_onset in image_onset:
         fig.axes[0].axvline(x=img_onset, linestyle='-', color='k')
    fig.axes[0].axhline(y=0.5, color='k', linestyle='-', alpha=.5)
    
    if data.ndim > 2: # we have the full timegen
        data_diag = np.array([np.diag(d) for d in data])
    else: # already have the diagonal
        data_diag = data
    if data_std is not None:
        if data_std.ndim > 2: # we have the full timegen
            data_std_diag = np.array([np.diag(d) for d in data_std])
        else: # already have the diagonal
            data_std_diag = data_std

    for i_plot in range(n_plots):
        color = cmap(i_plot) if not len(cmap_groups) else cmap(cmap_groups[i_plot])
        if len(labels):
            ax.plot(times_train, data_diag[i_plot], c=cmap(i_plot), alpha=0.5, lw=1/(n_plots/10), label=labels[i_plot])
        else:
            ax.plot(times_train, data_diag[i_plot], c=cmap(i_plot), alpha=0.5, lw=1/(n_plots/10))
        if data_std is not None:
            ax.fill_between(times_train, data_diag[i_plot]-data_std_diag[i_plot], data_diag[i_plot]+data_std_diag[i_plot], alpha=0.2, color=cmap(i_plot))

    # plot mean
    ax.plot(times_train, np.mean(data_diag, 0), c='k', alpha=0.8, lw=1, label="Mean")

    plt.ylabel(ylabel)
    plt.xlabel("Time (s)")
    if len(labels): plt.legend()
    plt.tight_layout()
    
    cmap_groups_str = f"_{len(np.unique(cmap_groups))}groups" if len(cmap_groups) else ""
    plt.savefig(f'{out_fn}_{ylabel}_all_diags{cmap_groups_str}.png')
    plt.close()


def plot_GAT(data_mean, out_fn, train_cond, train_tmin, train_tmax, test_tmin, test_tmax, ylabel="AUC", contrast=False, gen_cond=None, gen_color='k', slices=[], version="v1", window=False):
    train_word_onsets, train_image_onset = get_onsets(train_cond, version=version)
    if window: # Decoding inside subwindow
        word_onsets, image_onset = [], []
        train_word_onsets, train_image_onset = [], []
        orientation = "vertical"
        shrink = 1.
    elif gen_cond is not None: # mean it is a generalization
        word_onsets, image_onset = get_onsets(gen_cond, version=version)
        orientation = "horizontal"
        shrink = 0.7
    else:
        word_onsets, image_onset = train_word_onsets, train_image_onset
        orientation = "vertical"
        shrink = 1.

    try:
        n_times_train = data_mean.shape[0]
        n_times_test = data_mean.shape[1]
    except:
        set_trace()
    times_train = np.linspace(train_tmin, train_tmax, n_times_train)
    times_test = np.linspace(test_tmin, test_tmax, n_times_test)

    if contrast:
        vmin = np.min(data_mean) if np.min(data_mean) < -0.001 else -0.001 # make the colormap center on the white, whatever the values
        vcenter=0.
        vmax = np.max(data_mean) if np.max(data_mean) > 0.001 else 0.001
    else:
        vcenter = 0.5
        vmin = 0.4
        vmax = 0.6
    divnorm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # FULL TIMEGEN PLOT
    fig, ax = plt.subplots()
    plt.imshow(data_mean, norm=divnorm, cmap='bwr', origin='lower', extent=[test_tmin, test_tmax, train_tmin, train_tmax])
    cbar = plt.colorbar(orientation=orientation, shrink=shrink)
    cbar.set_label(ylabel)
    plt.xlabel("Testing time (s)")
    plt.ylabel("Training time (s)")

    for w_onset in word_onsets:
        fig.axes[0].axvline(x=w_onset, color='k', linestyle='--', alpha=.5)
    for img_onset in image_onset:
        fig.axes[0].axvline(x=img_onset, color='k', linestyle='-', alpha=.5)

    if gen_cond is not None: # if generalization, then use a different color
        color = gen_color
    else:
        color = "k"
    for w_onset in train_word_onsets:
        fig.axes[0].axhline(y=w_onset, color=color, linestyle='--', alpha=.5)
    for img_onset in train_image_onset:
        fig.axes[0].axhline(y=img_onset, color=color, linestyle='-', alpha=.5)

    plt.savefig(f'{out_fn}_{ylabel}_timegen.png')


    ## ADD SLICES
    if slices and ylabel=='AUC':
        cmap = plt.cm.get_cmap('plasma', len(slices))
        # add colored line to th matrix plot
        for i_slice, sli in enumerate(slices):
            plt.axhline(y=sli, linestyle=':', alpha=.9, color=cmap(i_slice))
        plt.savefig(out_fn + '_wslices.png')
        plt.close()

        max_auc = np.max(data_mean)
        all_slices_ave = []
        fig, ax = plt.subplots()
        for w_onset in word_onsets:
             fig.axes[0].axvline(x=w_onset, linestyle='--', color='k')
        for img_onset in image_onset:
             fig.axes[0].axvline(x=img_onset, linestyle='-', color='k')

        for i_slice, sli in enumerate(slices):
            # data_mean_line = data_mean[(np.abs(times_train - sli)).argmin()]
            line_idx = (np.abs(times_train - sli)).argmin()
            data_mean_line = np.mean(data_mean[line_idx-5:line_idx+5], 0)
            
            # vertical dashed line for time reference
            if gen_cond is None:
                plt.axvline(x=sli, color=cmap(i_slice), linestyle=':', alpha=.7)

            # actual trace plot
            plt.plot(times_test, data_mean_line, label=str(sli)+'s', color=cmap(i_slice))
            # plt.fill_between(times, data_line-std, data_line+std, color=cmap(i_slice), alpha=0.2)

            # Compute statistic
            # fvalues, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(all_AUC[:,(np.abs(times_train - sli)).argmin()]-0.5, n_permutations=1000, threshold=None, tail=1, n_jobs=5, verbose=False)
            # significance hlines
            # for i_clust, cluster in enumerate(clusters):
            #     if cluster_p_values[i_clust] < 0.05:
            #         # with std
            #         # plt.plot(times[cluster], np.ones_like(times[cluster])*(max_auc+0.1+0.01*i_slice), color=cmap(i_slice))
            #         # without std, put the hlines closer to the curves
            #         plt.plot(times_test[cluster], np.ones_like(times[cluster])*(max_auc+0.02+0.01*i_slice), color=cmap(i_slice))

        plt.legend(title='training time', loc='upper right')
        plt.axhline(y=0.5, color='k', linestyle='-', alpha=.3)
        plt.ylabel(ylabel)
        plt.xlabel("Time (s)")            
        plt.savefig(out_fn + '_slices.png')
        plt.close()

    return


## OLD, WITH statannot, WHILE THE NEW ONES USES THE MORE RECENT STATANNOTATIONS
# def make_sns_barplot(df, x, y, hue=None, box_pairs=[], out_fn="tmp.png", ymin=None, hline=None, rotate_ticks=False, tight=False, ncol=1):
#     fig, ax = plt.subplots(figsize=(18,14))
#     g = sns.barplot(x=x, y=y, hue=hue, data=df, ax=ax, ci=68) # ci=68 <=> standard error
#     ax.set_xlabel(x,fontsize=25)
#     ax.set_ylabel(y, fontsize=25)
#     if box_pairs:
#         _ = add_stat_annotation(ax, plot="barplot", data=df, x=x, hue=hue, y=y, line_offset_to_box=0, \
#                                 text_format='star', test='Wilcoxon', box_pairs=box_pairs, verbose=0, use_fixed_offset=True) 
#     if hue is not None:
#          = plt.gca().getax.legend().set_visible(False)
#         leg.set_title(hue, prop={'size':25})
#     if ymin is not None:
#         ax.set_ylim(ymin)
#     if hline is not None:
#         ax.axhline(y=hline, lw=1, ls='--', c='grey', zorder=-10)
#     if rotate_ticks:
#         for tick in ax.get_xticklabels():
#             tick.set_rotation(45)
#             tick.set_ha('right')
#     if tight:
#         plt.tight_layout()
#     if ncol > 1:
#         plt.legend(ncol=ncol)
#     plt.savefig(out_fn, transparent=True, dpi=400, bbox_inches='tight')
#     plt.close()


def plot_all_props(ave_dict, std_dict, times, out_fn, labels=['S1', 'C1', 'R', 'S2', 'C2'], word_onsets=[], image_onset=[]): # 'R'
    """ To plot multiple decoders diagonals on the same plot
    all on the same plot
    """
    fig, ax = plt.subplots()
    for w_onset in word_onsets:
        fig.axes[0].axvline(x=w_onset, linestyle='--', color='k', lw=1)
    for img_onset in image_onset:
        fig.axes[0].axvline(x=img_onset, linestyle='-', color='k', lw=1)
    for i, k in enumerate(labels):
        ax.plot(times, ave_dict[k], alpha=0.8, lw=3, c=cmaptab10(i), label=back2fullname(k))
        # ax.fill_between(times, ave_dict[k]-std_dict[k], ave_dict[k]+std_dict[k], alpha=0.2, zorder=-1, lw=0, color=cmaptab10(i))
    ax.axhline(y=0.5, color='k', linestyle='-', alpha=.5, lw=0.3)
    plt.legend()
    plt.ylabel("AUC")
    plt.xlabel("Time (s)")
    plt.xlim(-.5, 6)
    plt.tight_layout()
    plt.savefig(out_fn)


def add_slice_line_1ax(ax, slice_time, times, data_line, mult_fac=3, alpha=.4, fill=False, stat_fill=False, fill_where=[], color='k', lw=2, h_lw=1, mksz=8, zorder=-10):
    # Draw the plots in a few steps
    data_line *= mult_fac
    ax.plot(times, data_line+slice_time, color=color, lw=lw, clip_on=False, zorder=zorder)
    ax.axhline(xmin=times[0], xmax=times[-1], y=slice_time, linewidth=h_lw, clip_on=True, color=color, zorder=zorder)
    if fill:
        ax.fill_between(times, data_line+slice_time, slice_time, alpha=alpha, clip_on=False, zorder=zorder, color=color)
    if stat_fill: # fill only when significant
        ax.fill_between(times, data_line+slice_time, slice_time, where=fill_where, color=color, alpha=alpha, clip_on=False, zorder=zorder)
    # Remove axes details that don't play well with overlap
    ax.set_yticks([])
#     ax.set_xticks([])    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.set_xlim(times[0], times[-1])


def joyplot_with_stats(data_dict, times, out_fn, tmin=-.5, tmax=8, labels=['S1', 'C1', 'R', 'S2', 'C2'], \
                       word_onsets=[], image_onset=[], fsz=16, title_fsz=26, stat='wilc', threshold=0.01):
    """ To plot multiple decoders diagonals on the same plot
    one per subplot, with statistics
    """
    if not np.all([lab in data_dict.keys() for lab in labels]):
        print(f"!!! Could not find key(s) {[lab for lab in labels if lab not in data_dict.keys()]} in the data_dict")
        return
    fig, axes = plt.subplots(len(labels), figsize=(18, len(labels)*3))
    for i, k in enumerate(labels):
        for w_onset in word_onsets:
            axes[i].axvline(x=w_onset, linestyle='--', color='k', ymin=0., ymax=1, lw=1, clip_on=False)
        for img_onset in image_onset:
            axes[i].axvline(x=img_onset, linestyle='-', color='k', lw=1, clip_on=False)
        axes[i].axhline(y=0.5, color='k', linestyle='-', alpha=.7, lw=0.5)

        dat, std = np.mean(data_dict[k], 0), sem(data_dict[k], 0)
        time_mask = np.logical_and(tmin<=times, times<=tmax)
        dat, std, local_times = dat[time_mask], std[time_mask], times[time_mask]
        axes[i].plot(local_times, dat, alpha=0.8, lw=3, c=cmaptab10(i), clip_on=False)
        axes[i].fill_between(local_times, dat-std, dat+std, alpha=0.2, zorder=-1, lw=0, color=cmaptab10(i), clip_on=False)

        if stat == "wilc":
            np.random.seed(seed=233423)
            signif = np.zeros_like(local_times)
            chance_array = np.ones_like(data_dict[k][:,0]) * 0.5
            for t in range(len(local_times)):
                signif[t] = wilcoxon(data_dict[k][:, time_mask][:, t], chance_array, alternative="greater")[1] #two-sided
            signif = fdr_correction(np.array(signif), alpha=0.05)[0] #.astype(bool)
            axes[i].fill_between(local_times, 0.5, dat, where=signif, alpha=0.8, zorder=-1, lw=0, color=cmaptab10(i), clip_on=False)
            # axes[i].fill_between(local_times, 0.5, signifed_dat, alpha=0.8, zorder=-1, lw=0, color=cmaptab10(i), clip_on=False)
        # fvalues, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(data_dict[k]-0.5, n_permutations=1000, threshold=1e-6, tail=1, n_jobs=-1, verbose=False, buffer_size=None)
        elif stat == "cluster_perm":
            for cluster, pval in zip(clusters, cluster_p_values):
                if pval < 0.05:
                    axes[i].fill_between(local_times[cluster], 0.5, dat[cluster], alpha=0.8, zorder=-1, lw=0, color=cmaptab10(i), clip_on=False)
        
        axes[i].text(-0.02, .55, k.split("_")[0], ha='center', va='center', transform=axes[i].transAxes, fontsize=title_fsz)
        # back2fullname(k.split("_")[0])
        # axes[i].set_title(back2fullname(k.split("_")[0]), loc='left')
        
        # from ipdb import set_trace; set_trace()

        # cosmetics
        axes[i].set_xlim(tmin, tmax)
        for spine in ['bottom', 'top', 'left', 'right']:
            axes[i].spines[spine].set_visible(False)
        axes[i].set_yticks([])
        if (i+1) < len(labels): axes[i].set_xticks([])
        # axes[i].set_ylabel("AUC")

    axes[-1].tick_params(axis='both', which='major', labelsize=fsz)
    axes[-1].set_xlabel("Time (s)")
    plt.subplots_adjust(hspace=-.1)
    # plt.tight_layout()
    plt.savefig(out_fn)


def plot_all_props_multi(ave_dict, std_dict, times, out_fn, labels=['S1', 'C1', 'R', 'S2', 'C2'], word_onsets=[], image_onset=[]): # 'R'
    """ To plot multiple decoders diagonals on the same plot
    one per subplot
    """
    fig, axes = plt.subplots(len(labels), figsize=(12, len(labels)*3))
    for i, k in enumerate(labels):
        for w_onset in word_onsets:
            axes[i].axvline(x=w_onset, linestyle='--', color='k', lw=1)
        for img_onset in image_onset:
            axes[i].axvline(x=img_onset, linestyle='-', color='k', lw=1)
        axes[i].axhline(y=0.5, color='k', linestyle='-', alpha=.7, lw=0.5)
        axes[i].plot(times, ave_dict[k], alpha=0.8, lw=3, c=cmaptab10(i))
        axes[i].fill_between(times, ave_dict[k]-std_dict[k], ave_dict[k]+std_dict[k], alpha=0.2, zorder=-1, lw=0, color=cmaptab10(i))
        axes[i].set_title(back2fullname(k))
        axes[i].set_ylabel("AUC")
        axes[i].set_xlim(-.5, 7)
    axes[i].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_fn)


def plot_single_ch_perf(scores, info, out_fn, cmap_name='bwr', vmin=.4, vmax=.6, score_label='AUC', title=None, ticksize=14):
    from .plot_channels import plot_ch_scores
    cmap = matplotlib.cm.get_cmap(cmap_name)
    ## add dummy image for the cbar
    img = plt.imshow([scores, scores], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.gca().set_visible(False)
    fig, ax = plt.subplots()
    fig = plot_ch_scores(info, scores, cmap, ch_type='all', axes=ax)
    cbar_ax = fig.add_axes([0.8, 0.1, 0.1, 0.05])
    cbar = fig.colorbar(img, orientation="horizontal", cax=cbar_ax)
    cbar_ax.set_title(score_label)
    cbar.ax.tick_params(labelsize=ticksize)
    if title is not None:
        ax.set_title(title)
    plt.savefig(out_fn, dpi=200)


"""
********   HELPER FUNCTIONS   *********
*                                     *
*  .----.                    .---.    *
* '---,  `.________________.'  _  `.  *
*      )   ________________   <_>  :  *
* .---'  .'                `.     .'  *
*  `----'                    `---'    *
*                                     *
***************************************
Called by main functions """

def get_cv(train_cond, crossval, n_folds):
    """Choose crossvalidation scheme from argparse arguments"""
    if train_cond == "two_objects":
        if crossval == 'shufflesplit':
            cv = RepeatedStratifiedGroupKFold(n_splits=n_folds, n_repeats=10)
        elif crossval == 'kfold':
            print("Using StratifiedKFold instead of StratifiedGroupKFold")
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)    
            # cv = StratifiedGroupKFold(n_splits=n_folds)
    elif crossval == 'shufflesplit':
        cv = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.5, random_state=42)
    elif crossval == 'kfold':
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        print('unknown specified cross-validation scheme ... exiting')
        raise
    return cv


def get_split_indices(split_queries, epochs):
    """ Get the indices of specific queries 
    in order to test them separately 
    after training the decoders
    """
    if not isinstance(epochs, list): epochs = [epochs]
    test_split_query_indices = []
    actual_split_queries = [] # will contain only queries for which we found trials
    for split_query in split_queries: # get the indices of each of the categories to split during test
        metadatas = [epo.metadata for epo in epochs]
        split_overall_indices = [epo[split_query].metadata.index for epo in epochs]
        split_local_indices = [np.arange(len(meta))[np.isin(meta.index, split_idx)] for meta, split_idx in zip(metadatas, split_overall_indices)]
        # split local indices contains the indices in the epochs.get_data() corresponding to the query
        # offset the second query indices by the length of the first epochs.get_data() to get final indices (X is the concatenation of both epochs data)
        split_local_indices[1] += len(metadatas[0])
        cat_split_local_indices = np.concatenate(split_local_indices)
        if not len(cat_split_local_indices):
            print(f"Did not found any trial for split query {split_query}, moving on")
            continue
            # raise RuntimeError(f"split queries did not yield any trial for query: {split_query}")
        print(f"Found {len(cat_split_local_indices)} trial for split query {split_query}")
        test_split_query_indices.append(cat_split_local_indices)
        actual_split_queries.append(split_query)
    return test_split_query_indices, actual_split_queries


def get_X_y_from_epochs_list(args, epochs, sfreq):
    n_queries = len(epochs) # 2
    nchan = {} # store the number of channels for available modalities
    ch_names = {} # store the channel names for available modalities
    data = []
    for i in range(n_queries):
        data.append([])
        if isinstance(epochs[i], mne.time_frequency.EpochsTFR):
            sz = epochs[i].data.shape
            # data[-1].append(np.mean(epochs[i].data, axis=2))
            data[-1].append(epochs[i].data.reshape(sz[0], sz[1]*sz[2], sz[3]))
        else:
            data[-1].append(epochs[i].get_data(picks='meg'))
        nchan['meg'] = epochs[i].info['nchan']
        ch_names['meg'] = epochs[i].ch_names

    y = np.concatenate([np.ones(len(data[i][0]))*i for i in range(n_queries)]).astype(float)
    X = np.concatenate([np.concatenate(data[i], axis=1) for i in range(n_queries)])
    # X = trials * channels * time

    # shuffle labels to get baseline performance
    if args.shuffle:
        np.random.suffle(y)

    for i in range(n_queries):
        print(f'Using {len(data[i][0])} examples for class {i+1}')

    return X, y, nchan, ch_names


def equalize_events_single_epo(epo, class_queries):
    """ Equalize events in a single epoch
    typically for OVR """    
    print(f"Equalizing event counts: ", end='')
    md = epo.metadata
    counts = [len(md.query(class_query)) for class_query in class_queries]
    n_trials = min(counts)
    print(f"keeping: {n_trials} events in each class")
    epos = [epo[q][np.random.choice(range(len(epo[q])), n_trials, replace=False)] for q in class_queries]
    return mne.concatenate_epochs(epos, add_offset=False)


def get_X_y_from_queries(epochs, class_queries, split_queries):
    """ get X and y for decoding based on 
    an epochs object and a list of queries (for OVR)
    also returns split queries indices
    """
    md = epochs.metadata
    X, y, groups = [], [], []
    mds_for_split = [] # store sub-mds, in the order of X and y, to get correct indices for split queries
    for i, class_query in enumerate(class_queries):
        if not len(md.query(class_query)):
            print(f"!! did not find any trial for query {class_query} !!")
        X.extend(epochs[class_query].get_data())
        y.extend([i for qwe in range(len(md.query(class_query)))])
        mds_for_split.append(md.query(class_query))
        # rely on the indices in the metadata to get groups. Only useful to split scene trials 
        groups.extend(md.query(class_query).index.values)
    md_for_split = pd.concat(mds_for_split).reset_index(drop=True) # single df, same order as X and y
    test_split_query_indices = [] # get split query indices
    for split_query in split_queries:
        test_split_query_indices.append(md_for_split.query(split_query).index.values)
    X, y = np.array(X), np.array(y)
    return X, y, groups, test_split_query_indices


class RidgeClassifierCVwithProba(RidgeClassifierCV):
    def predict_proba(self, X):
        d = self.decision_function(X)
        d_2d = np.c_[-d, d]
        return softmax(d_2d)