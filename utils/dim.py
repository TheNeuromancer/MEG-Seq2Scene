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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
from scipy.signal import savgol_filter
from scipy.stats import ttest_1samp
from autoreject import AutoReject
from statannotations.Annotator import Annotator
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from mne.decoding import UnsupervisedSpatialFilter
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace


# local import
from .commons import *
from .params import *
from utils.decod import get_X_y_from_epochs_list, save_results


def load_data(args, fn, queries):
    ''' Similar to utils.decod's load_data 
    but takes arbitrary number of queries 
    and returns as many epochs objects '''
    epochs = mne.read_epochs(fn, preload=True, verbose=False)
    if "two_objects-epo.fif" in fn:
        epochs.metadata = complement_md(epochs.metadata)
        epochs.metadata['Complexity'] = epochs.metadata.apply(add_complexity_to_md, axis=1)
    if "Flash" not in epochs.metadata.keys():
        epochs.metadata['Flash'] = 0 # old subject did not have flashes
    if args.filter: # filter metadata before anything else
        print(epochs.metadata)
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
    if queries: # load sub-epochs, one for each query
        epochs = [epochs[query] for query in queries]
    else: # load the whole epochs (typically for RSA)
        epochs = [epochs]
    epochs = [epo.pick('meg') for epo in epochs]
    initial_sfreq = epochs[0].info['sfreq']

    if args.equalize_events:
        print(f"Equalizing event counts: ", end='')
        n_trials = min([len(epo) for epo in epochs])
        print(f"keeping: {n_trials} events in each class")
        epochs = [epo[np.random.choice(range(len(epo)), n_trials, replace=False)] for epo in epochs]


    ### SELECT ONLY CH THAT HAD AN EFFECT IN THE LOCALIZER
    if args.localizer:
        auc_loc = pickle.load(open(path2loc, 'rb'))[args.subject[0:2]]
        print(f'Keeping {sum(auc_loc>args.auc_thresh)} MEG channels out of {len(auc_loc)} based on localizer results\n')
        ch_to_keep = np.where(auc_loc > args.auc_thresh)[0]
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


    # # Concatenate/average time point if needed
    # if args.cat:
    #     X = savgol_filter(X, window_length=51, polyorder=3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

    # if args.cat:
    #     data = win_ave_smooth(data, nb_cat=args.cat, mean=args.mean)
    #     new_info = epochs[0].info.copy()
    #     old_ch_names = deepcopy(new_info['ch_names'])
    #     for i in range(args.cat-1): new_info['ch_names'] += [f"{ch}_{i}" for ch in old_ch_names]
    #     new_info['nchan'] = new_info['nchan'] * args.cat
    #     old_info_chs = deepcopy(new_info['chs'])
    #     for i in range(args.cat-1): new_info['chs'] += deepcopy(old_info_chs)
    #     for i in range(new_info['nchan']): new_info['chs'][i]['ch_name'] = new_info['ch_names'][i] 
    # else:
    #     new_info = epochs[0].info
    
    new_info = epochs[0].info.copy() # default, overwritten if cat is not none
    if args.cat:
        data = win_ave_smooth(data, nb_cat=args.cat, mean=args.mean)
        if not args.mean: # create new channel names because concatenation of timepoints equate adding channels
            print(f"Concatenating {args.cat} consecutive timepoints")
            old_ch_names = deepcopy(epochs[0].info['ch_names'])
            new_ch_names = [f"{ch}_{i}" for i in range(args.cat) for ch in old_ch_names]
            new_nchan = epochs[0].info['nchan'] * args.cat
            new_ch_types = [typ for i in range(args.cat) for typ in epochs[0].info.get_channel_types('meg')]
            new_info = mne.create_info(ch_names=new_ch_names, sfreq=epochs[0].info['sfreq'], ch_types=new_ch_types)
        else:
            print(f"Averaging {args.cat} consecutive timepoints")

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


## Old way
# def split_for_pca(args, X):
#     # returns all splits
#     cv = KFold(args.n_folds, shuffle=True, random_state=42)
#     return ([[X[train].mean(0), X[test].mean(0)] for train, test in cv.split(X)])


# def PCA_dim(args, X_train, X_test):
#     ''' dimensionality analysis 
#     using PCA and reconstruction '''
#     pca = PCA(args.n_comp)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train, 0) # fit scaler on the overall evoked
#     pca.fit(X_train)
#     X_test = scaler.transform(X_test)
#     PCAed = pca.transform(X_test)
    
#     components = pca.components_ # n_comp, nchan
#     exvar, MSE, l2 = [], [], [] # score per component
#     for i_comp in range(args.n_comp):
#         reconstruction = np.dot(PCAed[:, 0:i_comp+1], components[0:i_comp+1])

#         exvar.append(explained_variance_score(X_test, reconstruction))
#         MSE.append(np.mean((X_test - reconstruction)**2))
#         l2.append(np.linalg.norm(X_test - reconstruction))

#     return exvar, MSE, l2


## NEW WAY
def participation_ratio(eigenvalues): # as defined in Gao 2017
    return eigenvalues.sum()**2 / np.sum([eig**2 for eig in eigenvalues])


# def catplot():
#     fig, ax = plt.subplots()
#     annotator = Annotator(ax, box_pairs, plot='boxplot', data=local_df, x='cond', y='mean l2', text_format='star')
#     sns.boxplot(data=local_df, x='cond', y='mean l2')
#     annotator.configure(test='Mann-Whitney', verbose=False, loc="inside", fontsize=12, use_fixed_offset=True).apply_and_annotate() # , comparisons_correction="bonferroni"
#     plt.ylabel(f"Full sentence reconstruction score (L2)")
#     plt.xlabel("")
#     plt.savefig(output_fn+f"_full_sentence_score_L2_box.png", dpi=300)
#     plt.close('all')


# def plot_average_over_time()

