import mne
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from ipdb import set_trace
import argparse
from glob import glob
import os.path as op
import os
from natsort import natsorted
import pandas as pd
import pickle
import time
from copy import deepcopy
from tqdm import trange
from yellowbrick.classifier import ROCAUC
from scipy.signal import savgol_filter
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, RobustScaler, label_binarize, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV, RidgeCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, permutation_test_score, GridSearchCV, LeaveOneOut, GroupKFold
from sklearn.decomposition import PCA
from sklearn.utils.extmath import softmax        
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

# local import
from .params import *
from .commons import *
from .split import *


class RidgeClassifierCVwithProba(RidgeClassifierCV):
    def predict_proba(self, X):
        d = self.decision_function(X)
        d_2d = np.c_[-d, d] # go 2d to get proba for all classes at the same time
        return softmax(d_2d, copy=False)

cmap = plt.cm.get_cmap('tab10', 10)

short_to_long_cond = {"loc": "localizer", "one_obj": "one_object", "two_obj": "two_objects",
                      "localizer": "localizer", "one_object":"one_object"}


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


def get_paths_rsa(args, dirname='RSA'):
    subject_string = args.subject if args.subject!='grand' else ''
    in_dir = f"{args.root_path}/Data/{args.epochs_dir}/{subject_string}"
    print("\nGetting training filename:")
    if isinstance(args.train_cond, list):
        print("Getting MULTIPLE training filenameS:")
        train_fn = [natsorted(glob(in_dir + f'/{cond}*-epo.fif'))[0] for cond in args.train_cond]
    else:
        print(in_dir + f'/{args.train_cond}*-epo.fif')
        train_fn = natsorted(glob(in_dir + f'/{args.train_cond}*-epo.fif'))[0]
    print(train_fn)
    # assert len(train_fn) == 1
    print("\nGetting test filenames:")
    test_fns = [natsorted(glob(in_dir + f'/{cond}*-epo.fif')) for cond in args.test_cond]
    print(test_fns)

    ### SET UP OUTPUT DIRECTORY AND FILENAME
    out_fn = get_out_fn(args, dirname=dirname)
    xdawn_str = "dawn-" if args.xdawn else ""
    out_fn += xdawn_str

    # wait for a random time in order to avoird conflit (parallel jobs that try to construct the same directory)
    rand_time = float(str(abs(hash(str(args))))[0:8]) / 100000000
    print(f'sleeping {rand_time} seconds to desynchronize parallel scripts')
    time.sleep(rand_time)
    
    out_dir = f'{args.root_path}/Results/{dirname}_v{args.version}/{args.epochs_dir}/{args.subject}'
    if not op.exists(out_dir):
        print('Constructing output directory')
        os.makedirs(out_dir)
    else:
        print('output directory already exists')
        if args.overwrite:
            print('overwrite is set to True ... overwriting\n')
        else:
            print('overwrite is set to False ... exiting smoothly')
            exit()
    return train_fn, out_fn


def load_data_rsa(args, fn, filter=''):
    # LOAD THE DATA
    epochs = mne.read_epochs(fn, preload=True, verbose=False)
    if args.subtract_evoked:
        epochs = epochs.subtract_evoked(epochs.average())
    if args.filter: # filter metadata before anything else
        print(epochs.metadata)
        epochs = epochs[args.filter]
    epochs = epochs.pick('meg')
    print(epochs.info)
    initial_sfreq = epochs[0].info['sfreq']

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
        epochs = epochs.pick(ch_to_keep)

    if args.freq_band:
        # freq_bands = dict(delta=(1, 3.99), theta=(4, 7.99), alpha=(8, 12.99), beta=(13, 29.99), low_gamma=(30, 69.99), high_gamma=(70, 150))
        freq_bands = dict(low_high=(0.03, 80), low_low=(0.03, 40), high_low=(2, 40), high_vlow=(2, 20), 
                                      low_vlow=(0.03, 20), low_vhigh=(0.03, 160), high_vhigh=(2, 160), 
                                      vhigh_vhigh=(20, 160), vhigh_high=(20, 80), vhigh_low=(20, 40))
        fmin, fmax = freq_bands[args.freq_band]
        print("\nFILTERING WITH FREQ BAND: ", (fmin, fmax))
        # bandpass filter
        epochs = epochs.filter(fmin, fmax, n_jobs=-1)

    if args.sfreq < epochs.info['sfreq']: 
        if args.sfreq < epochs.info['lowpass']:
            print(f"Lowpass filtering the data at the final sampling rate, {args.sfreq}Hz")
            epochs.filter(None, args.sfreq, l_trans_bandwidth='auto', h_trans_bandwidth='auto', filter_length='auto', phase='zero', fir_window='hamming', fir_design='firwin')
        print(f"Resampling from {epochs.info['sfreq']} to {args.sfreq} ... ")
        epochs = epochs.resample(args.sfreq)
        print("finished resampling ... ")

    data = epochs.data if isinstance(epochs, mne.time_frequency.EpochsTFR) else epochs.get_data()

    # CLIP THE DATA
    if args.clip:
        print('clipping the data at the 5th and 95th percentile for each channel')
        for ch in range(data.shape[1]):
            # Get the 5th and 95th percentile and channel
            p5 = np.percentile(data[:, ch, :], 5)
            p95 = np.percentile(data[:, ch, :], 95)
            data[:, ch, :] = np.clip(data[:, ch, :], p5, p95)

    # # add the temporal derivative of each channel as new feature
    # X = np.concatenate([X, np.gradient(X, axis=1)], axis=1)

    if args.smooth:
        print(f"Smoothing the data with a gaussian window of size {args.smooth}")
        for i_trial in range(len(data)):
            for i_ch in range(len(data[i_trial])):
                data[i_trial, i_ch] = smooth(data[i_trial, i_ch], window_len=5, window='hanning')
        # X = gaussian_filter1d(X, sigma=25, axis=2)

    # # Concatenate/average time point if needed
    # if args.cat:
    #     X = savgol_filter(X, window_length=51, polyorder=3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

    if args.cat:
        data = win_ave_smooth(data, nb_cat=args.cat, mean=args.mean)[0]
        new_info = epochs.info.copy()
        old_ch_names = deepcopy(new_info['ch_names'])
        for i in range(args.cat-1): new_info['ch_names'] += [f"{ch}_{i}" for ch in old_ch_names]
        new_info['nchan'] = new_info['nchan'] * args.cat
        old_info_chs = deepcopy(new_info['chs'])
        for i in range(args.cat-1): new_info['chs'] += deepcopy(old_info_chs)
        for i in range(new_info['nchan']): new_info['chs'][i]['ch_name'] = new_info['ch_names'][i] 
    else:
        new_info = epochs.info

    # print('zscoring each epoch')
    # for idx in range(X.shape[0]):
    #     for ch in range(X.shape[1]):
    #         X[idx, ch, :] = (X[idx, ch] - np.mean(X[idx, ch])) / np.std(X[idx, ch])
    epochs = mne.EpochsArray(data, new_info, metadata=epochs.metadata, tmin=epochs.times[0], verbose="warning")


    # crop after getting high gammas and smoothing to avoid border issues
    block_type = op.basename(fn).split("-epo.fif")[0]
    tmin, tmax = tmin_tmax_dict[block_type]
    print('initial tmin and tmax: ', epochs.times[0], epochs.times[-1])
    print('cropping to final tmin and tmax: ', tmin, tmax)
    epochs = epochs.crop(tmin, tmax)

    ### BASELINING
    if args.baseline:
        print('baselining...')
        epochs = epochs.apply_baseline((tmin, 0))

    return epochs #, test_split_query_indices #, pd.concat(metadatas)


def plot_rsa(results, factors, out_fn, times, cond, data_std=None, ylabel="Spearman r", version="v1", dpi=50):
    """ Plot traces of RSA correlation between empirical and a bunch of theoretical matrices
    results = n_factors * n_times
    """
    word_onsets, image_onset = get_onsets(cond, version=version)
    n_times = len(times)

    # FACTORS PLOT
    fig, ax = plt.subplots()
    if data_std is not None:
        for factor, fac_results, std_results in zip(factors, results, data_std):
            plot = plt.plot(times, fac_results, label=factor)
            ax.fill_between(times, fac_results-std_results, fac_results+std_results, alpha=0.2)
    else:
        for factor, fac_results in zip(factors, results):
            plot = plt.plot(times, fac_results, label=factor)
        
    plt.ylabel(ylabel)
    plt.xlabel("Time (s)")
    plt.legend()
    for w_onset in word_onsets:
         fig.axes[0].axvline(x=w_onset, linestyle='--', color='k')
    for img_onset in image_onset:
         fig.axes[0].axvline(x=img_onset, linestyle='-', color='k')
    plt.tight_layout()
    plt.savefig(f'{out_fn}_rsa.png')
    plt.close()



def multi_plot_rsa(results, factors, out_fn, times, cond, data_std=None, ylabel="Spearman", version="v1", dpi=50):
    """ Plot traces of RSA correlation between empirical and a bunch of theoretical matrices
    one per subplot
    results = n_factors * n_times
    """
    word_onsets, image_onset = get_onsets(cond, version=version)
    n_times = len(times)
    fig, axes = plt.subplots(len(factors), 1, figsize=(8, 3*len(factors)), sharex=True)
    if data_std is not None:
        for i, (factor, fac_results, std_results) in enumerate(zip(factors, results, data_std)):
            print(i)
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(factor)
            for w_onset in word_onsets:
                 axes[i].axvline(x=w_onset, linestyle='--', color='k')
            for img_onset in image_onset:
                 axes[i].axvline(x=img_onset, linestyle='-', color='k')
            axes[i].axhline(y=0, linestyle='-', color='grey')
            plot = axes[i].plot(times, fac_results, label=factor, c=cmap(i))
            axes[i].fill_between(times, fac_results-std_results, fac_results+std_results, alpha=0.2, color=cmap(i))
    else:
        for i, (factor, fac_results) in enumerate(zip(factors, results)):
            axes[i].set_ylabel(ylabel)
            axes[i].set_xlabel("Time (s)")
            axes[i].set_title(factor)
            for w_onset in word_onsets:
                 axes[i].axvline(x=w_onset, linestyle='--', color='k')
            for img_onset in image_onset:
                 axes[i].axvline(x=img_onset, linestyle='-', color='k')
            plot = axes[i].plot(times, fac_results, label=factor, c=cmap(i))
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f'{out_fn}_rsa_multiplot.png')
    plt.close()


def decoder_confusion(args, epochs, class_queries, n_times):
    """ X: n_trials, n_sensors, n_times
        y: n_trials
        class_queries: list of strings, pandas queries to get each class
    """
    md = epochs.metadata
    X, y, groups = [], [], []
    for i, class_query in enumerate(class_queries):
        X.extend(epochs[class_query].get_data())
        y.extend([i for qwe in range(len(md.query(class_query)))])
        # rely on the indices in the metadata to get groups. Only useful to split scene trials 
        groups.extend(md.query(class_query).index.values)

    X, y = np.array(X), np.array(y)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    print(f"n_classes: {n_classes}")

    # if not np.all(counts >= args.n_folds): # can't do usual crossval with only a few example of each class... so do LOO. Happens with single scene classification
    #     # cv = LeaveOneOut() # waaaaay too long
    #     print(f"Using only {args.min_nb_trial} folds because we don;t have enough trials")
    #     cv = StratifiedKFold(n_splits=args.min_nb_trial, shuffle=True, random_state=42)

    # if "Right_obj" in class_queries[0] or "Left_obj" in class_queries[0]: # special case, need groupedKFold
    if args.train_cond == "two_objects":
        if args.crossval == 'shufflesplit':
            cv = RepeatedStratifiedGroupKFold(n_splits=args.n_folds, n_repeats=10)
        elif args.crossval == 'kfold':
            cv = StratifiedGroupKFold(n_splits=args.n_folds)
    elif args.crossval == 'shufflesplit':
        cv = StratifiedShuffleSplit(n_splits=args.n_folds, test_size=0.5, random_state=42)
    elif args.crossval == 'kfold':
        cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    else:
        print('unknown specified cross-validation scheme ... exiting')
        raise
    
    clf = LogisticRegressionCV(Cs=10, class_weight='balanced', solver='liblinear', multi_class='auto', n_jobs=-1, cv=5, max_iter=10000)
    # clf = LogisticRegressionCV(Cs=10, class_weight='balanced', solver='lbfgs', max_iter=10000, verbose=False, cv=5, n_jobs=1)
    # clf = LogisticRegression(C=1, class_weight='balanced', solver='liblinear', multi_class='auto')
    # clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 9), cv=cv, class_weight='balanced')
    # clf = RidgeClassifierCVwithProba(alphas=np.logspace(-4, 4, 9), cv=5, class_weight='balanced')
    # clf = RidgeClassifier(class_weight='balanced')
    clf = OneVsRestClassifier(clf, n_jobs=1)
    # clf = OneVsOneClassifier(clf, n_jobs=1)
    onehotenc = OneHotEncoder(sparse=False, categories='auto')
    onehotenc = onehotenc.fit(np.arange(n_classes).reshape(-1,1))

    if args.reduc_dim:
        pipeline = make_pipeline(RobustScaler(), PCA(args.reduc_dim), clf)
    else:
        pipeline = make_pipeline(RobustScaler(), clf)


    all_models = []
    AUC = np.zeros(n_times)
    all_confusions = np.zeros((n_times, len(classes), len(classes))) # full confusion matrix
    for t in trange(n_times):
        # all_models.append([])
        for train, test in cv.split(X, y, groups=groups): # groups is ignored for non-groupedKFold
            pipeline.fit(X[train, :, t], y[train])
            # all_models[-1].append(deepcopy(pipeline))
            preds = predict(pipeline, X[test, :, t], multiclass=True)
            if preds.ndim == 2:
                preds = preds
            else:
                preds = onehotenc.transform(preds.reshape((-1,1)))

            all_confusions[t] += confusion_matrix(y[test], preds.argmax(1), normalize='all')
            AUC[t] += roc_auc_score(y_true=y[test], y_score=preds, multi_class='ovr', average='weighted') / args.n_folds

    return AUC, all_confusions



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


def save_rsa_results(args, out_fn, results, factors):
    print('Saving RSA results')
    pickle.dump(dict(zip(factors, results)), open(out_fn + '_all_results.p', 'wb'))



def get_class_queries_rsa(cond, order_or_side='order'):
    """ get the queries for each class
    depending on the condition
    6 classes for localizer
    9 for objects
    18 for scenes
    """
    colors = ["vert", "bleu", "rouge"]
    shapes = ["triangle", "cercle", "carre"]
    if cond == "localizer":
        class_queries = colors + shapes 
        print("TODO: Implement decoding of image localier, only using words for now")
    elif cond == "one_object":
        class_queries = [f"Shape1=='{s}' and Colour1=='{c}'" for s in shapes for c in colors]
    elif cond == "two_objects":
        if order_or_side == "order": # order of presentation
            class_queries = [f"Shape1=='{s}' and Colour1=='{c}'" for s in shapes for c in colors] + \
                            [f"Shape2=='{s}' and Colour2=='{c}'" for s in shapes for c in colors]
        elif order_or_side == "side": # actual side that the object is in the corresponding image (with a matching trial)
            class_queries = [f"Left_obj=='{s}_{c}'" for s in shapes for c in colors] + \
                            [f"Right_obj=='{s}_{c}'" for s in shapes for c in colors]
        else:
            raise RuntimeError(f"Wrong query for scenes: {order_or_side}")
    return class_queries



def get_dsm_from_queries(factor, class_queries):
    """ get the distance matrix for a given factor 
    for a set of queries that will be used to get trials for each class
    """
    if "+" in factor: # joint factor, get a matrix for each subfactor
        subfac_dsms = []
        for subfac in factor.split("+"): # recursive call
            subfac_dsms.append(get_dsm_from_queries(subfac, class_queries))
        if factor == "Right_obj+Left_obj": # here we need a least one to be equal
            dsm = np.sum(subfac_dsms, 0) > 0
        else: # if any property is different, then the distance is 1, else 0
            try:
                dsm = np.any(subfac_dsms, 0)
                # dsm = np.logical_or(*subfac_dsms)
            except:
                set_trace()
        return dsm

    # marginal factor or recursive calls
    dsm = np.ones((len(class_queries), len(class_queries)))

    # special case, we don't have (sub) factor names in the queries
    if "Right_obj" in class_queries[0] or "Left_obj" in class_queries[0]:
        for ii, class_query1 in enumerate(class_queries):
            if factor == "Shape": 
                val1 = class_query1.split("'")[1].split("_")[0]
            elif factor == "Colour":
                val1 = class_query1.split("'")[1].split("_")[1]
            else: # Right_obj and Left_obj factors
                val1 = class_query1 #.split("'")[1]
            # print("\n", val1)
            for jj, class_query2 in enumerate(class_queries):
                if factor == "Shape": 
                    val2 = class_query2.split("'")[1].split("_")[0]
                elif factor == "Colour":
                    val2 = class_query2.split("'")[1].split("_")[1]
                else: # Right_obj and Left_obj factors
                    val2 = class_query2 #.split("'")[1]
                # print(val2)
                if val1 == val2:
                    dsm[ii, jj] = 0
    
    else: # usual case, Shapes and Colours only
        for ii, class_query1 in enumerate(class_queries):
            if factor not in class_query1: # query irrelevant of the factor, let distance=1
                # print("m1")
                continue
            try:
                val1 = class_query1.split(factor)[1].split("'")[1]
            except:
                set_trace()
            # print("\n", val1)
            for jj, class_query2 in enumerate(class_queries):
                if factor not in class_query2: # query irrelevant of the factor, let distance=1
                    # print("m1")
                    continue
                try:
                    val2 = class_query2.split(factor)[1].split("'")[1]
                except:
                    set_trace()
                # print(val2)
                if val1 == val2:
                    dsm[ii, jj] = 0
    return dsm


def regression_score(data_dsm, model_dsms):
    """ Compute regression coefficients 
    for each input matrices
    """
    reg = RidgeCV()
    scaler = MinMaxScaler() # StandardScaler()
    reg = make_pipeline(scaler, reg)
    reg.fit(model_dsms, data_dsm)
    coefs = reg[1].coef_
    return coefs


def random_forest_regression_score(data_dsm, model_dsms):
    """ Compute feature importance 
    for each input matrices using random forest
    """
    # use permutation_importance instead? Better but need to do train-test split and refit the estimator many times.
    # do some crazy gridsearch over parameters?
    reg = RandomForestRegressor()
    scaler = MinMaxScaler() # StandardScaler()
    reg = make_pipeline(scaler, reg)
    reg.fit(model_dsms, data_dsm)
    feat_imp = reg[1].feature_importances_
    return feat_imp

