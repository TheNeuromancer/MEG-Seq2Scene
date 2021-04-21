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
from sklearn.preprocessing import StandardScaler, RobustScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, permutation_test_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from scipy.signal import savgol_filter
from scipy.stats import ttest_1samp
from scipy.ndimage import gaussian_filter1d

# local import
from .params import *


short_to_long_cond = {"loc": "localizer", "one_obj": "one_object", "two_obj": "two_objects",
                      "localizer": "localizer", "one_object":"one_object"}


def get_onsets(cond, version="v1"):
    """ get the word and image onsets depending on the condition
    """
    if version == "v1": # first version, longger SOA
        SOA_dict = {"localizer": .9, "one_object": .65, "two_objects": .65}
    else: # second version for subject 9 and up
        SOA_dict = {"localizer": .9, "one_object": .6, "two_objects": .6}
    delay_dict = {"localizer": None, "one_object": 1., "two_objects": 2.}
    nwords_dict = {"localizer": 1, "one_object": 2, "two_objects": 5}

    SOA = SOA_dict[cond]
    delay = delay_dict[cond]
    nwords = nwords_dict[cond]

    word_onsets = []
    image_onset = []
    for i_w in range(nwords):
        word_onsets.append(i_w * SOA)

    if delay: # for one_object and two_objects conditions
        image_onset.append((i_w+1) * SOA + delay)

    return word_onsets, image_onset



def get_out_fn(args, dirname='Decoding'):
    out_dir = f'{args.root_path}/Results/{dirname}_v{args.version}/{args.epochs_dir}/{args.subject}'

    # C_string = f'_{args.C}C'
    # cat_string = '_' + str(args.cat) + "cat" if not args.mean else '_' + str(args.cat) + "mean"
    reduc_dim_str = f"_reduc{args.reduc_dim}comp" if args.reduc_dim else "" # %50 
    baseline_str = "_baseline" if args.baseline else ""
    # smooth_str = f"_{args.smooth}smooth" if args.smooth else ""
    shuffle_str = '_shuffled' if args.shuffle else ''
    fband_str = f'_{args.freq_band}' if args.freq_band else ''
    cond_str = f"_cond-{args.train_cond}-"
    if hasattr(args, 'train_query_1'):
        if args.train_query_1 and args.train_query_2:
            train_query_1 = '_'.join(args.train_query_1.split())
            train_query_2 = '_'.join(args.train_query_2.split())
        else:
            train_query_1 = ''
            train_query_2 = ''
        out_fn = f'{out_dir}/{args.label}-{train_query_1}_vs_{train_query_2}{reduc_dim_str}{shuffle_str}{fband_str}{cond_str}'
    else: # RSA
        out_fn = f'{out_dir}/{args.label}-{reduc_dim_str}{shuffle_str}{fband_str}{cond_str}'
    
    out_fn = shorten_filename(out_fn)
    print('\noutput file will be in: ' + out_fn)
    print('eg:' + out_fn + '_AUC_diag.npy\n')
    return out_fn


def shorten_filename(fn):
    # shorten the output fn because we sometimes go over the 255-characters limit imposed by ubuntu
    fn = fn.replace("'", "")
    fn = fn.replace('"', '')
    fn = fn.replace('[', '')
    fn = fn.replace('(', '')
    fn = fn.replace(')', '')
    fn = fn.replace(']', '')
    fn = fn.replace(',', '')
    fn = fn.replace('Colour', 'C')
    fn = fn.replace('Shape', 'S')
    fn = fn.replace('Binding', 'Bd')
    fn = fn.replace('cercle', 'cl')
    fn = fn.replace('carre', 'ca')
    fn = fn.replace('triangle', 'tr')
    fn = fn.replace('bleu', 'bl')
    fn = fn.replace('vert', 'vr')
    fn = fn.replace('rouge', 'rg')
    fn = fn.replace('==', '=')
    fn = fn.replace('Matching=match', 'match')
    
    # if fn is still too long, make some ugly changes
    # if len(os.path.basename(fn)) > 255:
    if len(fn) > 255:
        fn = fn.replace('reduc', '')
        fn = fn.replace('baseline', 'bl')
        fn = fn.replace('and_', '')
        fn = fn.replace('or_', '_')
        fn = fn.replace('cond', 'cd')

    # if fn is still too long, make some even uglier changes
    # if len(os.path.basename(fn)) > 255:
    if len(fn) > 255: # remove underscores but only in the fn, not in the path
        fn = fn.replace(op.basename(fn), op.basename(fn).replace('_', ''))

    # if fn is still too long, throw an error
    # if len(os.path.basename(fn)) > 255:
    if len(fn) > 255:    
        print(f'\n\nOutput fn is too long.\n\n{fn}\nis longer than UNIX limit which is 255 characters...cropping brutally')
        fn = fn[0:255]
        # exit()

    return fn


def get_paths(args, dirname='Decoding'):

    subject_string = args.subject if args.subject!='grand' else ''

    in_dir = f"{args.root_path}/Data/{args.epochs_dir}/{subject_string}"
    print("\nGetting training filename:")
    print(in_dir + f'/{args.train_cond}*-epo.fif')
    train_fn = natsorted(glob(in_dir + f'/{args.train_cond}*-epo.fif'))
    print(train_fn)
    assert len(train_fn) == 1
    print("\nGetting test filenames:")
    test_fns = [natsorted(glob(in_dir + f'/{cond}*-epo.fif')) for cond in args.test_cond]
    print(test_fns)

    ### SET UP OUTPUT DIRECTORY AND FILENAME
    out_fn = get_out_fn(args, dirname=dirname)
    test_out_fns = []
    for test_cond in args.test_cond: 
        test_query_str = f"{'_'.join(args.test_query_1.split())}_vs_{'_'.join(args.test_query_2.split())}"
        test_out_fns.append(shorten_filename(f"{out_fn}_tested_on_{test_cond}_{test_query_str}"))

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
    return train_fn, test_fns, out_fn, test_out_fns


def load_data(args, fn, query_1, query_2):

    # LOAD THE DATA
    epochs = mne.read_epochs(fn[0], preload=True, verbose=False)
    if args.subtract_evoked:
        epochs = epochs.subtract_evoked(epochs.average())
    epochs = [epochs[query_1], epochs[query_2]]
    epochs = [epo.pick('meg') for epo in epochs]
    print(epochs[0].info)
    print(epochs[1].info)
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
        print(f"starting resampling from {epochs[0].info['sfreq']} to {args.sfreq} ... ")
        epochs = [epo.resample(args.sfreq) for epo in epochs]
        print("finished resampling ... ")

    data = [epo.data if isinstance(epo, mne.time_frequency.EpochsTFR) else epo.get_data() for epo in epochs]

    metadatas = [epo.metadata for epo in epochs]

    # CLIP THE DATA
    if args.clip:
        print('clipping the data at the 5th and 95th percentile for each channel')
        for query_data in data:
            for ch in range(query_data.shape[1]):
                # Get the 5th and 95th percentile and channel
                p5 = np.percentile(query_data[:, ch, :], 5)
                p95 = np.percentile(query_data[:, ch, :], 95)
                query_data[:, ch, :] = np.clip(query_data[:, ch, :], p5, p95)


    # # add the temporal derivative of each channel as new feature
    # X = np.concatenate([X, np.gradient(X, axis=1)], axis=1)

    if args.smooth:
        print(f"Smoothing the data with a gaussian window of size {args.smooth}")
        for query_data in data:
            for i_trial in range(len(query_data)):
                for i_ch in range(len(query_data[i_trial])):
                    query_data[i_trial, i_ch] = smooth(query_data[i_trial, i_ch], window_len=5, window='hanning')
        # X = gaussian_filter1d(X, sigma=25, axis=2)


    # # Concatenate/average time point if needed
    # if args.cat:
    #     X = savgol_filter(X, window_length=51, polyorder=3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

    if args.cat: # old way to smooth, moving average window
        for query_data in data:
            sz = query_data.shape
            if args.mean:
                new_data = np.zeros_like(query_data)
            else:
                new_data = np.zeros((sz[0], sz[1]*args.cat, sz[2]))

            for t in range(sz[2]):
                nb_to_cat = args.cat if t>args.cat else t
                if args.mean: # average consecutive timepoints
                    new_data[:,:,t] = query_data[:,:,t-nb_to_cat:t+1].mean(axis=2)
                else: # concatenate
                    if nb_to_cat < args.cat: # we miss some data points before tmin 
                        # just take the first timesteps and copy them
                        dat = query_data[:,:,t-nb_to_cat:t+1]
                        # dat = dat.reshape(sz[0], sz[1] * dat.shape[2])
                        while dat.shape[2] < args.cat:
                            idx = np.random.choice(nb_to_cat+1) # take a random number below the current timepoint
                            dat = np.concatenate((dat, dat[:,:,idx,np.newaxis]), axis=2) # add it to the data
                        new_data[:,:,t] = dat.reshape(sz[0], sz[1] * args.cat)
                    else:
                        new_data[:,:,t] = query_data[:,:,t-nb_to_cat:t].reshape(sz[0], sz[1] * nb_to_cat)
            query_data = new_data


    # print('zscoring each epoch')
    # for idx in range(X.shape[0]):
    #     for ch in range(X.shape[1]):
    #         X[idx, ch, :] = (X[idx, ch] - np.mean(X[idx, ch])) / np.std(X[idx, ch])
    epochs = [mne.EpochsArray(data, old_epo.info, metadata=meta, tmin=old_epo.times[0], verbose="warning") for data, meta, old_epo in zip(data, metadatas, epochs)]


    # crop after getting high gammas and smoothing to avoid border issues
    block_type = op.basename(fn[0]).split("-epo.fif")[0]
    tmin, tmax = tmin_tmax_dict[block_type]
    print('initial tmin and tmax: ', [(epo.times[0], epo.times[-1]) for epo in epochs])
    print('cropping to final tmin and tmax: ', tmin, tmax)
    epochs = [epo.crop(tmin, tmax) for epo in epochs]
    

    ### BASELINING
    if args.baseline:
        print('baselining...')
        epochs = [epo.apply_baseline((tmin, 0)) for epo in epochs]


    test_split_query_indices = []
    for split_query in args.split_queries: # get the indices of each of the categories to split during test
        metadatas = [epo.metadata for epo in epochs]
        split_overall_indices = [epo[split_query].metadata.index for epo in epochs]
        split_local_indices = [np.arange(len(meta))[np.isin(meta.index, split_idx)] for meta, split_idx in zip(metadatas, split_overall_indices)]
        # split local indices contains the indices in the epochs.get_data() corresponding to the query
        # offset the second query indices by the length of the first epochs.get_data() to get final indices (X is the concatenation of both epochs data)
        split_local_indices[1] += len(metadatas[0])
        test_split_query_indices.append(np.concatenate(split_local_indices))

    return epochs, test_split_query_indices #, pd.concat(metadatas)



def get_data(args, epochs, sfreq):

    n_queries = 2

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

    print(f'Using {len(data[0][0])} examples for class {1}')
    print(f'Using {len(data[1][0])} examples for class {2}')

    return X, y, nchan, ch_names


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    
    see also: 
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')

    return y[(window_len//2):-(window_len//2)]
    # return y[(window_len//2-1):-(window_len//2)]
    # return y



def predict(clf, data):
    """
    wrapper for predicting from any sklearn classifier
    """
    try:
        pred = clf.predict_proba(data)[:,1] # no multiclass so just keep the proba for class 1
    except AttributeError: # no predict_proba method
        pred = clf.predict(data)
    return pred


def decode(args, X, y, clf, n_times, test_split_query_indices):

    if args.crossval == 'shufflesplit':
        cv = StratifiedShuffleSplit(n_splits=args.n_folds, test_size=0.5)
    elif args.crossval == 'kfold':
        cv = StratifiedKFold(n_splits=args.n_folds, shuffle=False) # do not shuffle here!! 
        # that is because we stored the indices of the queries we want to split at test time.
        # plus we shuffle during the data loading
    else:
        print('unknown specified cross-validation scheme ... exiting')
        raise
    
    if args.reduc_dim:
        pipeline = make_pipeline(RobustScaler(), PCA(args.reduc_dim), clf)
    else:
        pipeline = make_pipeline(RobustScaler(), clf)

    all_models = []
    if args.timegen:
        AUC = np.zeros((n_times, n_times))
        mean_preds = np.zeros((n_times, n_times))
        if test_split_query_indices: # split the test indices according to the query
            AUC_test_query_split = np.zeros((n_times, n_times, len(test_split_query_indices)))
            mean_preds_test_query_split = np.zeros((n_times, n_times, len(test_split_query_indices)))
        else:
            AUC_test_query_split = None
            mean_preds_test_query_split = None

        for train, test in cv.split(X, y):
            all_models.append([])
            for t in trange(n_times):
                pipeline.fit(X[train, :, t], y[train])
                all_models[-1].append(deepcopy(pipeline))
                for tgen in range(n_times):
                    if test_split_query_indices: # split the test indices according to the query
                        for i_query, split_indices in enumerate(test_split_query_indices):
                            test_query = test[np.isin(test, split_indices)]
                            pred = predict(pipeline, X[test_query, :, tgen])
                            AUC_test_query_split[t, tgen, i_query] += roc_auc_score(y_true=y[test_query], y_score=pred) / args.n_folds
                            mean_preds_test_query_split[t, tgen, i_query] += np.mean(pred) / args.n_folds
                        
                    # normal test
                    pred = predict(pipeline, X[test, :, tgen])
                    AUC[t, tgen] += roc_auc_score(y_true=y[test], y_score=pred) / args.n_folds
                    mean_preds[t, tgen] += np.mean(pred) / args.n_folds

    # TODO: implement generalization over conditions for the non-timegen case
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

    print('mean AUC: ', AUC.mean())
    print('max AUC: ', AUC.max())

    if args.avg_clf: # average classifier parameters over all crossval splits
        if args.reduc_dim:
            print('cannot average classifier weights when using PCA because there can \
                be a different nb of selected components for each fold .. .exiting')
            raise
        all_models = get_averaged_clf(args, all_models, n_times)

    return all_models, AUC, AUC_test_query_split


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

        # # PCA in exists
        # if args.reduc_dim:
        #     all_models[t][0][1].components_ = np.mean([all_models[t][i][1].components_ for i in range(args.n_folds)], 0) 
    
    # discard the model for each split
    # averaged_models = all_models[:,0]
    return avg_models

    # # clf = SVC(C=args.C, kernel='linear', probability=True, class_weight='balanced', verbose=False)
    # clf.coef_ = np.mean([clf.coef_ for clf in all_models[t]], axis=0)
    # clf.intercept_ = np.mean([clf.intercept_ for clf in all_models[t]], axis=0)
    # pred = predict(clf, t_data)
    # print(roc_auc_score(y_true=y, y_score=pred))
    # print('\n')
            

def test_decode(args, X, y, all_models):
    n_folds = len(all_models)
    n_times_test = X.shape[2]
    n_times_train = len(all_models[0])
    
    if args.timegen:
        AUC = np.zeros((n_times_train, n_times_test))
        mean_preds = np.zeros((n_times_train, n_times_test))
        for tgen in trange(n_times_test):
            t_data = X[:, :, tgen]
            for t in range(n_times_train):
                all_folds_preds = []
                for i_fold in range(n_folds):
                    pipeline = all_models[i_fold][t]
                    all_folds_preds.append(predict(pipeline, t_data))
                AUC[t, tgen] = roc_auc_score(y_true=y, y_score=np.mean(all_folds_preds, 0))
                mean_preds[t, tgen] = np.mean(all_folds_preds)

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

    print('mean AUC: ', AUC.mean())
    print('max AUC: ', AUC.max())

    return AUC, mean_preds


def test_decode_all_sentence(args, X, y, all_models):
    X = get_sentence_representation(args, X)
    
    all_folds_preds = []
    for i_fold in range(args.n_folds):
        pipeline = all_models[i_fold]
        all_folds_preds.append(predict(pipeline, X))
    AUC = roc_auc_score(y_true=y, y_score=np.mean(all_folds_preds, 0))
    mean_preds = np.mean(all_folds_preds)

    print('test AUC: ', AUC)
    return AUC, mean_preds


def decode_single_chan(args, X, y, clf):
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=False) # do not shuffle here!! 
    # that is because we stored the indices of the queries we want to split at test time.
    # plus we shuffle during the data loading
    if args.reduc_dim:
        pipeline = make_pipeline(RobustScaler(), PCA(args.reduc_dim), clf)
    else:
        pipeline = make_pipeline(RobustScaler(), clf)

    nchan = X.shape[1]
    all_models = []
    AUC = np.zeros((nchan))
    for train, test in cv.split(X, y):
        all_models.append([])
        for ch in trange(nchan):
            ch_data = X[train, ch, :]
            pipeline.fit(ch_data, y[train])
            all_models[-1].append(deepcopy(pipeline))
            pred = predict(pipeline, X[test, ch, :])
            AUC[ch] += roc_auc_score(y_true=y[test], y_score=pred) / args.n_folds

    return all_models, AUC

def test_decode_single_chan(args, X, y, all_models):    
    nchan = X.shape[1]
    AUC = np.zeros(nchan)
    for ch in range(nchan):
        ch_data = X[:, ch, :]
        all_folds_preds = []
        for i_fold in range(args.n_folds):
            pipeline = all_models[i_fold][ch]
            all_folds_preds.append(predict(pipeline, ch_data))
        AUC[ch] = roc_auc_score(y_true=y, y_score=np.mean(all_folds_preds, 0))

    return AUC


def permutation_test_single_chan(args, X, y, clf):
    """
    Only returns the score and pval, for now no way to get the trained models and generalize them to new conditions
    """
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=False) # do not shuffle here!! 
    # that is because we stored the indices of the queries we want to split at test time.
    # plus we shuffle during the data loading
    if args.reduc_dim:
        pipeline = make_pipeline(RobustScaler(), PCA(args.reduc_dim), clf)
    else:
        pipeline = make_pipeline(RobustScaler(), clf)

    nchan = X.shape[1]
    pvals = np.zeros((nchan))
    AUC = np.zeros((nchan))
    for ch in trange(nchan):
        score, perm_scores, pvalue = permutation_test_score(pipeline, X[:, ch, :], y, scoring="roc_auc", 
                                                        cv=cv, n_permutations=args.n_perm, n_jobs=-2)
        AUC[ch] = score
        pvals[ch] = pvalue

    return pvals, AUC


def gridsearch_decode(args, X, y, clf, n_times):
    '''
    get best parameters over a family of classifiers. 
    through nested crossval. 
    typically called by test_classifiers_decoding.py
    '''
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=False, random_state=42)
    # cv = StratifiedShuffleSplit(n_splits=50, test_size=0.2)

    if args.reduc_dim:
        pipeline = make_pipeline(RobustScaler(), PCA(args.reduc_dim), clf)
    else:
        pipeline = make_pipeline(RobustScaler(), clf)

    all_models = []
    if args.timegen:
        AUC = np.zeros((n_times, n_times))
        for train, test in cv.split(X, y):
            all_models.append([])
            for t in trange(n_times):
                pipeline.fit(X[train, :, t], y[train])
                all_models[-1].append(deepcopy(pipeline))
                for tgen in range(n_times):                    
                    # normal test
                    pred = predict(pipeline, X[test, :, tgen])
                    AUC[t, tgen] += roc_auc_score(y_true=y[test], y_score=pred) / args.n_folds
    else:
        AUC = np.zeros(n_times)                
        for train, test in cv.split(X, y):
            all_models.append([])
            for t in trange(n_times):
                pipeline.fit(X[train, :, t], y[train])
                all_models[-1].append(deepcopy(pipeline))
                pred = predict(pipeline, X[test, :, t])
                AUC[t] += roc_auc_score(y_true=y[test], y_score=pred) / args.n_folds

    print('mean AUC: ', AUC.mean())
    print('max AUC: ', AUC.max())

    return all_models, AUC


# ///////////////////////////////////////////////////////// #
#################### SAVING AND PLOTTING ####################
# ///////////////////////////////////////////////////////// #

def save_results(args, out_fn, AUC, all_models=None):
    print('Saving results')
    if args.timegen:
        AUC_diag = np.diag(AUC)
        np.save(out_fn + '_AUC.npy', AUC)
    else:
        AUC_diag = AUC
        np.save(out_fn + '_AUC_diag.npy', AUC_diag)
    if all_models:
        pickle.dump(all_models, open(out_fn + '_all_models.p', 'wb'))
    return


def save_preds(args, out_fn, preds):
    print('Saving predictions')
    if args.timegen:
        preds_diag = np.diag(preds)
        np.save(out_fn + '_preds.npy', preds)
    else:
        preds_diag = preds
        np.save(out_fn + '_preds_diag.npy', preds_diag)
    return


def save_patterns(args, out_fn, all_models):
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


def plot_perf(args, out_fn, data_mean, train_cond, train_tmin, train_tmax, test_tmin, test_tmax, ylabel="AUC", contrast=False, gen_cond=None):
    """ plot performance of individual subject,
    called during training by decoding.py script
    """
    if gen_cond is None:
        plot_diag(data_mean=data_mean, data_std=None, out_fn=out_fn, train_cond=train_cond, 
            train_tmin=train_tmin, train_tmax=train_tmax, ylabel=ylabel, contrast=contrast)

    plot_GAT(data_mean=data_mean, out_fn=out_fn, train_cond=train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, 
             test_tmax=test_tmax, ylabel=ylabel, contrast=contrast, gen_cond=gen_cond, slices=[])
    return


def plot_diag(data_mean, out_fn, train_cond, train_tmin, train_tmax, data_std=None, ylabel="AUC", contrast=False, version="v1"):
    word_onsets, image_onset = get_onsets(train_cond, version=version)
    n_times_train = data_mean.shape[0]
    times_train = np.linspace(train_tmin, train_tmax, n_times_train)

    # DIAGONAL PLOT
    fig, ax = plt.subplots()
    data_mean_diag = np.diag(data_mean)
    plot = plt.plot(times_train, data_mean_diag)
    if data_std is not None:
        data_std_diag = np.diag(data_std)
        ax.fill_between(times_train, data_mean_diag-data_std_diag, data_mean_diag+data_std_diag, alpha=0.2)
    plt.ylabel(ylabel)
    plt.xlabel("Time (s)")

    for w_onset in word_onsets:
         fig.axes[0].axvline(x=w_onset, linestyle='--', color='k')
    for img_onset in image_onset:
         fig.axes[0].axvline(x=img_onset, linestyle='-', color='k')

    plt.savefig(f'{out_fn}_{ylabel}_diag.png')
    plt.close()


def plot_GAT(data_mean, out_fn, train_cond, train_tmin, train_tmax, test_tmin, test_tmax, ylabel="AUC", contrast=False, gen_cond=None, gen_color='k', slices=[], version="v1"):
    train_word_onsets, train_image_onset = get_onsets(train_cond, version=version)
    if gen_cond is not None: # mean it is a generalization
        word_onsets, image_onset = get_onsets(gen_cond, version=version)
        orientation = "horizontal"
        shrink = 0.7
    else:
        word_onsets, image_onset = train_word_onsets, train_image_onset
        orientation = "vertical"
        shrink = 1.

    n_times_train = data_mean.shape[0]
    n_times_test = data_mean.shape[1]
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


def get_ylabel_from_fn(fn):
    # acc or AUC
    if fn[-7:-4] == 'acc' or fn[-12:-9] == 'acc':
        ylabel = 'Accuracy'
    elif fn[-7:-4] == 'AUC' or fn[-12:-9] == 'AUC':
        ylabel = 'AUC'
    elif fn[-9:-4] == 'preds': # or fn[-12:-9] == 'AUC':
        ylabel = 'prediction'
    else:
        print('\n\nDid not find a correct label in the filename')
        set_trace()
    return ylabel


# def plot_all_sentence_decoding(args, out_fn, data):
#     t_stat, pval = ttest_1samp(data, 0.5)
#     plt.figure()
#     plt.bar(0, np.mean(data), yerr=np.std(data))
#     plt.savefig(f'{out_fn}_bar.png')
#     # print('save pval??')

# def plot_all_sentence_decoding_test(args, out_fn, data):
#     data = np.array(data) # n_subjects * (1 + splits-queries + generalisation)
#     stars = []
#     for dat in data.T:
#         t_stat, pval = ttest_1samp(dat, 0.5)
#         if pval < 0.001:
#             stars.append('***')
#         elif pval < 0.01:
#             stars.append('**')
#         elif pval < 0.05:
#             stars.append('*')
#         else:
#             stars.append('')
#     cmap = plt.cm.get_cmap('cividis', len(data[0]))
#     colors = [cmap(i) for i in range(len(data[0]))]
#     plt.figure()
#     plt.bar(range(len(data[0])), np.mean(data, 0), yerr=np.std(data, 0), color=colors)
#     frame = plt.gca()
#     # frame.axes.get_xaxis().set_visible(False)
#     frame.axes.get_xaxis().set_ticks(range(len(data[0])))
#     frame.axes.get_xaxis().set_ticklabels(stars)    
#     plt.ylabel('AUC')
#     plt.savefig(f'{out_fn}_bar_test.png')

# def save_single_chan_results(args, out_fn, AUC, pvals, ch_names, all_models=None):
#     print('Saving Single channel results')
#     AUC = dict(zip(ch_names, AUC))
#     pickle.dump(AUC, open(out_fn + '_AUC.p', 'wb'))
#     pvals = dict(zip(ch_names, pvals))
#     pickle.dump(pvals, open(out_fn + f'_{args.n_perm}p_pvals.p', 'wb'))
#     if all_models:
#         pickle.dump(all_models, open(out_fn + '_all_models.p', 'wb'))
#     return


# def plot_single_chan_perf(args, out_fn, data, pvals, ylabel="AUC", contrast=False):
#     print('Plotting single channel performance')
#     nchan = len(data)
#     channels = np.arange(nchan)

#     reject, corrected_pvals = mne.stats.fdr_correction(pvals, alpha=0.05)

#     fig, ax = plt.subplots()
#     plot = plt.plot(channels, data)
#     plt.ylabel(ylabel)
#     plt.xlabel("channels")
#     if contrast:
#         plt.axhline(y=0, color='k', linestyle='--')
#     else:
#         plt.axhline(y=0.5, color='k', linestyle='--')
#     for idx in np.where(reject)[0]: plt.plot(idx, np.max(data)+0.1, 'ro')
#     plt.savefig(f'{out_fn}_{ylabel}.png')
#     plt.close()

#     fig, ax = plt.subplots()
#     plot = plt.plot(channels, np.sort(data))
#     plt.ylabel(ylabel)
#     plt.xlabel("channels")
#     if contrast:
#         plt.axhline(y=0, color='k', linestyle='--')
#     else:
#         plt.axhline(y=0.5, color='k', linestyle='--')
#     for idx in np.where(np.sort(reject))[0]: plt.plot(idx, np.max(data)+0.1, 'ro')
#     plt.savefig(f'{out_fn}_{ylabel}_sorted.png')
#     plt.close()

#     return



def plot_gridseach_parameters(args, out_fn, params, times, AUC):
    '''
    plotting the best parameters for each time point after a gridsearch, 
    canonically called in test_classifiers_decoding.py
    '''
    n_colors = 100
    cmap = plt.cm.get_cmap('hsv', n_colors)

    for param in params:
        fig, ax1 = plt.subplots()
        color = cmap(np.random.randint(n_colors))
        # if param in ['alpha', 'C', 'learning_rate']: # logspaced values
        #     ax1.plot(times, params[param], color=color)
        #     ax1.set_yscale('log')
        # else: # plot dummy values so as to have constant spacing in the yaxis

        # orig_entries = []
        # for i_fold in range(len(params[param])): 
        #     orig_entries.append([str(val) for val in params[param][i_fold]])
        # labels = sorted(np.unique(orig_entries))

        labels = sorted(np.unique(params[param])) # all entries are already string
        
        values = [] # get the corresponding indices
        for i_fold in range(args.n_folds):
            values.append([labels.index(params[param][i_fold, t]) for t in range(len(times))])
        
        mean_values = np.mean(values, axis=0)
        std_values = np.std(values, axis=0)

        ax1.plot(times, mean_values, color=color)
        ax1.fill_between(times, mean_values-std_values, mean_values+std_values, color=color, alpha=0.2)

        ax1.set_yticks(ticks=range(len(labels)))
        ax1.set_yticklabels(labels=[l.decode('UTF-8') for l in labels])
        ax1.set_ylabel(param, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(times, AUC, color='k')
        ax2.set_ylabel('AUC', color='k')
        ax2.tick_params(axis='y', labelcolor='k')

        ax1.set_xlabel('time (s)')
        plt.tight_layout()
        plt.savefig(f'{out_fn}_best_{param}.png')



def get_decod_onset(AUC, th=0.6, n_min=5):
    ''' Get the onset of significant decoding performance,
    ie at least n_min consecutive time points above threshold th.
    returns 0 if there is no significant cluster.
    '''
    if len(AUC.shape) == 2:
        AUC = np.diag(AUC)

    above_th = AUC > th
    consec_tp = 0
    signif = False
    for t in range(len(AUC)):
        if above_th[t]:
            consec_tp += 1
        else:
            consec_tp = 0
        if consec_tp == n_min:
            signif = True
            break

    if signif:
        return t - n_min
    else:
        return None

def get_decod_spread(AUC, onset, th=0.6):
    '''get the spread of the decoding performance,
    ie the normalized area ofthe square starting at the onset of significant decoding
    '''
    square = AUC[onset::, onset::]
    above_th_square = square > th
    spread = np.sum(above_th_square) / np.prod(above_th_square.shape)
    return spread

    # consecutive_timepoints = np.diff(np.where(np.concatenate(([above_th[0]], above_th[:-1] != above_th[1:], [True])))[0])[::2]

    # if np.any(consecutive_timepoints > n_min):

    # chance = np.ones_like(AUC) * 0.5
    # # Compute statistic
    # fvalues, clusters, cluster_p_values, H0 = \
    #     permutation_cluster_test([AUC, chance], n_permutations=1000, tail=1, n_jobs=-1, verbose=False)
