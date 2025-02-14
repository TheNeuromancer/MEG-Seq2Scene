import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import matplotlib.pyplot as plt

import mne
import numpy as np
from ipdb import set_trace
import pandas as pd
import argparse
import pickle
import time
import os.path as op
import os
import importlib
from glob import glob
from natsort import natsorted
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import sem, spearmanr
from prettytable import PrettyTable
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import warnings
# warnings.filterwarnings('ignore', '.*Provided stat_fun.*', )
warnings.filterwarnings('ignore', '.*No clusters found.*', )

from utils.decod import *
from utils.params import *

matplotlib.rcParams.update({'font.size': 19})
matplotlib.rcParams.update({'lines.linewidth': 2})
plt.rcParams['figure.figsize'] = [12., 8.]
plt.rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser(description='MEG ans SEEG plotting of decoding results')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='all',help='subject name')
parser.add_argument('-o', '--out-dir', default='agg', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
# parser.add_argument('--ovr', action='store_true',  default=False, help='Whether to get the one versus rest directory or classic decoding')
parser.add_argument('-v', '--verbose', action='store_true',  default=False, help='Print more stuff')
parser.add_argument('-a', '--already_saved', action='store_true',  default=False, help='if the output file was already created, just load it instead of recomputing everything. Use with caution, if you changed a parameter then you need to run the pipeline again.')
# parser.add_argument('--smooth_plot', default=0, type=int, help='Smoothing preds before plotting')
args = parser.parse_args()

config = importlib.import_module(f"configs.{args.config}", "Config").Config() # import config parameters
for arg in vars(config): setattr(args, arg, getattr(config, arg)) # update argparse with arguments from the config
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

print('This script lists all the patterns.npy files in all the subjects decoding output directories, plots the patterns and compute the correlations between them')
v = args.version
decoding_dir = f"Decoding_multi_v{v}" 
if args.subject in ["all", "v1", "v2",  "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"
print('\noutput files will be in: ' + out_dir)
create_folder(out_dir, args.overwrite)

# list all preds.npy files in the directory
all_fns = natsorted(glob(in_dir + f'/*patterns.npy'))
if not all_fns:
    raise RuntimeError(f"Did not find any patterns files in {in_dir}/*patterns.npy ... Did you pass the right config?")

mag_idx, grad_idx, all_idx = [pickle.load(open(f"{args.root_path}/Data/{s}_indices.p", "rb")) for s in ['mag', 'grad', 'all']]
mag_info, grad_info, all_info = [pickle.load(open(f"{args.root_path}/Data/{s}_info.p", "rb")) for s in ['mag', 'grad', 'all']]
indices = {'mag': mag_idx, 'grad': grad_idx, 'all': all_idx}
infos = {'mag': mag_info, 'grad': grad_info, 'all': all_info}

def plot_patterns(pattern, out_fn, mag_info, mag_idx, grad_info, grad_idx):
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(pattern[mag_idx], mag_info, axes=ax, contours=0)
    plt.savefig(f'{out_fn}_pattern_mag.png')
    plt.close()
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(pattern[grad_idx], grad_info, axes=ax, contours=0)
    plt.savefig(f'{out_fn}_pattern_grad.png')
    plt.close()

    # vmin, vcenter, vmax = np.min(pattern), 0, np.max(pattern)
    # plot_single_ch_perf(pattern[mag_idx], mag_info, f"{out_fn}_pattern_mag_.png", cmap_name='bwr', vmin=vmin, vcenter=vcenter, vmax=vmax, score_label='Weight', title=None, ticksize=14)
    # plot_single_ch_perf(pattern[grad_idx], grad_info, f"{out_fn}_pattern_grad_.png", cmap_name='bwr', vmin=vmin, vcenter=vcenter, vmax=vmax, score_label='Weight', title=None, ticksize=14)

# def get_correlation_concat_for_all_subjects(all_patterns, n_classes):
# Correlates the concatenated data of all subjects. Use the other function "get_correlation_across_subjects"
#     correlation_matrix = np.zeros((n_classes, n_classes))
#     # Compute the correlations across subjects for each pair of classes
#     for i in range(n_classes):
#         for j in range(n_classes):
#             cls1_data = all_patterns[:, i, :]  # shape: (n_subjects, n_sensors)
#             cls2_data = all_patterns[:, j, :]  # shape: (n_subjects, n_sensors)
#             # Flatten sensors across subjects for correlation calculation
#             corr = np.corrcoef(cls1_data.flatten(), cls2_data.flatten())[0, 1] # 2 by 2 
#             correlation_matrix[i, j] = corr
#     return correlation_matrix

def get_correlation_across_subjects(all_patterns):
    # compute the cross-correlation between patterns across participants 
    # (for each pair of participants, then averaged)
    n_subs, n_classes, n_sensors = all_patterns.shape
    correlation_matrix = np.zeros((n_classes, n_classes)) # Initialize a 6x6 matrix to store correlations
    for i in range(n_classes): # Loop over all pairs of categories
        for j in range(n_classes):
            correlations = [] # List to store correlations between participants for the category pair (i, j)
            for p1 in range(n_subs): # Loop over all pairs of participants
                for p2 in range(p1 + 1, n_subs):  # Only compute for unique pairs
                    # Get sensor weights for the two participants for categories i and j
                    weights_p1_i = all_patterns[p1, i, :]
                    weights_p2_j = all_patterns[p2, j, :]

                    # Compute correlation between participant p1 for category i and participant p2 for category j
                    corr = np.corrcoef(weights_p1_i, weights_p2_j)[0, 1]
                    correlations.append(corr) # Store the correlation

                    # Ensure symmetry by also correlating p2's category i with p1's category j
                    if i != j:
                        weights_p2_i = all_patterns[p2, i, :]
                        weights_p1_j = all_patterns[p1, j, :]
                        corr_symmetric = np.corrcoef(weights_p2_i, weights_p1_j)[0, 1]
                        correlations.append(corr_symmetric)
            
            # Take the mean correlation for category pair (i, j)
            correlation_matrix[i, j] = np.mean(correlations)
    
    return correlation_matrix    


def plot_correlation(correlations, out_fn, labels, vmin=None, vmax=None):
    fig, ax = plt.subplots()
    if vmin is None: vmin = np.min(correlations)
    if vmax is None: vmax = np.max(correlations)
    im = ax.imshow(correlations, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(correlations)))
    ax.set_yticks(np.arange(len(correlations)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.colorbar(im, label="Correlation")
    plt.savefig(f'{out_fn}_correlations.png')

def get_labels(label):
    if "Prop" in label:
        return shapes + colors + relations 
    else:
        raise ValueError(f"Unknown label {label}, should be Property")
    # if "S" in label:
    #     return ["carre", "cercle", "triangle"]
    # elif "C" in label:
    #     return ["rouge", "bleu", "vert"]
    # elif "R" in label:
    #     return ["left", "right"]
    # else:
    #     from ipdb import set_trace; set_trace()

## All possible training time (depends on the property that is decoded) (relative to relevant word onset)
train_times = ["0.17", "0.2", "0.3", "0.4", "0.5", "0.6", "0.8"]

patterns_fn = f"{op.dirname(op.dirname(out_dir))}/all_patterns.p"
all_labels = np.unique([op.basename(fn).split('-')[0] for fn in all_fns]) # Shoud be a single one: Property
if not args.already_saved:
    all_df = []
    # report = mne.Report()
    for label in all_labels:
        if args.verbose: print(f"Doing {label}")
        for train_cond in ["localizer_one_object_two_objects"]:
            for train_time in train_times:
                if args.verbose: print(train_time)
                all_patterns = []
                future_df = {}
                for fn in all_fns: # maybe change this loop? Loop only once for every file and use string comprehension to get the file parameters. 
                    if op.basename(fn)[0:len(label)+1] != f"{label}-": continue 
                    if f"cond-{train_cond}-" not in fn: continue
                    # if "tested_on" in fn: continue # ensure we don't have generalization results (shouldn't be usefull after the following line)
                    if train_time not in fn:
                        continue
                    if f"#{train_time},{train_time}#{train_time},{train_time}#" not in fn:
                        continue

                    if args.verbose: print('loading file ', fn)
                    pattern = np.load(fn)
                    all_patterns.append(pattern)
                    
                    future_df['pattern'] = [pattern]
                    future_df['subject'] = [op.basename(op.dirname(fn))[0:2]]
                    future_df['train_cond'] = [train_cond]
                    future_df['train_time'] = [train_time]
                    future_df['label'] = [label]
                    all_df.append(pd.DataFrame(future_df))


                n_subs = len(all_patterns)
                if not n_subs: 
                    if args.verbose: print(f"Not a single pattern found for {label} trained on {train_cond} at {train_time} ...") 
                    continue
                else:
                    median_pattern = np.median(all_patterns, 0) # median over subjects
                    all_patterns = np.array(all_patterns) # shape: (n_subjects, n_classes, n_sensors)
                out_fn = f"{out_dir}/{label}_trained_on_{train_cond}_at_{train_time}_{n_subs}ave"

                if median_pattern.ndim == 2: # OVR, one additional dimension n_classes  * n_sensors
                    labels = get_labels(label)
                    n_classes = len(labels)
                    for ch_type in ['all']: # 'mag', 'grad', 
                        # average correlation plot between the patterns averaged over subjects (not so interesting, diag is ones)
                        correlations = np.corrcoef(median_pattern[:,indices[ch_type]])
                        plot_correlation(correlations, f"{out_fn}_averaged_{ch_type}", labels, vmin=-1, vmax=1)
                        ## correlation over subjects
                        corr_mat = get_correlation_across_subjects(all_patterns[:,:,indices[ch_type]])
                        plot_correlation(corr_mat, f"{out_fn}_over_subjects_{ch_type}", labels)

                    # report.add_figs_to_section(f'{label} trained on {train_cond} at {train_time}', [f'{out_fn}_correlations.png'], section=f'{label} trained on {train_cond} at {train_time}')

                    for patt in median_pattern:
                        plot_patterns(patt, out_fn, mag_info, mag_idx, grad_info, grad_idx)

                else:
                    plot_patterns(median_pattern, out_fn, mag_info, mag_idx, grad_info, grad_idx)

                if args.verbose: print(f"Finished {label} trained on {train_cond} at {train_time}\n")
                plt.close('all')

                    # pattern_all_labels[f"{label}_{train_cond}"] = pattern # store values for all labels for multi plot

    df = pd.concat(all_df)
    df.to_csv(f"{out_dir}/all_patterns.csv") #, index=False)

else: # if already saved, just load the data
    # df = pd.read_csv(f"{out_dir}/all_patterns.csv")
    raise NotImplementedError("Not implemented yet -- saving np.arrays in the df will not work, it fails at loading time")


for t in ["0.2", "0.3", "0.4", "0.6"]:

    # colors, shapes, 1 and 2 
    grouped_patterns = np.stack(df.query(f"label=='Prop{t}'")['pattern'].values) # already of shape n_subs * total n_classes * n_sensors
    # grouped_patterns = []
    # grouped_patterns.append(np.stack(df.query(f"label=='S1' & train_cond=='scenes' & train_time=='{t}'")['pattern'].values))
    # grouped_patterns.append(np.stack(df.query(f"label=='C1' & train_cond=='scenes' & train_time=='{float(t)+.6:.1f}'")['pattern'].values)) #[:len(grouped_patterns[0])])
    # grouped_patterns.append(np.stack(df.query(f"label=='R' & train_cond=='scenes' & train_time=='{float(t)+1.2:.1f}'")['pattern'].values)[:,np.newaxis,:])
    # grouped_patterns.append(np.stack(df.query(f"label=='S2' & train_cond=='scenes' & train_time=='{float(t)+1.8:.1f}'")['pattern'].values))
    # grouped_patterns.append(np.stack(df.query(f"label=='C2' & train_cond=='scenes' & train_time=='{float(t)+2.4:.1f}'")['pattern'].values)) # {str(float(t)+2.4)}
    # concat_patterns = np.concatenate(grouped_patterns, 1) # n_subs * total n_classes (3+3+2or1?+3+3) * n_sensors
    concat_patterns = grouped_patterns
    n_subs = len(concat_patterns)
    labels = shapes + colors + relations #+ ["rel"] + shapes + colors
    corr_mat_all = get_correlation_across_subjects(concat_patterns[:,:,indices['all']])
    out_fn_all = f"{out_dir}/{n_subs}ave_over_subjects_All_Features_t{t}_all_ch"
    plot_correlation(corr_mat_all, out_fn_all, labels)
    corr_mat_mag = get_correlation_across_subjects(concat_patterns[:,:,indices['mag']])
    out_fn_mag = f"{out_dir}/{n_subs}ave_over_subjects_All_Features_t{t}_mag"
    plot_correlation(corr_mat_mag, out_fn_mag, labels)
    corr_mat_grad = get_correlation_across_subjects(concat_patterns[:,:,indices['grad']])
    out_fn_grad = f"{out_dir}/{n_subs}ave_over_subjects_All_Features_t{t}_grad"
    plot_correlation(corr_mat_grad, out_fn_grad, labels)
            

print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")
