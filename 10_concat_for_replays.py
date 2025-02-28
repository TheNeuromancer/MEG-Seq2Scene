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
from scipy.stats import sem, ttest_ind
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler
import warnings
# warnings.filterwarnings('ignore', '.*Provided stat_fun.*', )
# warnings.filterwarnings('ignore', '.*No clusters found.*', )

from utils.decod import *
from utils.params import *
from utils.replays import plot_reactivations, plot_word_position_reactivations

matplotlib.rcParams.update({'font.size': 19})
matplotlib.rcParams.update({'lines.linewidth': 2})
plt.rcParams['figure.figsize'] = [12., 8.]
plt.rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser(description='MEG ans SEEG plotting of decoding results')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='all',help='subject name')
parser.add_argument('-i', '--in-dir', default='multi', help='input directory (ovr or multi)')
parser.add_argument('-o', '--out-dir', default='agg', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('-d', '--dont_recompute', action='store_true',  default=False, help='Whether to skip the aggregation phase, only works if we already saved the preds')
parser.add_argument('-v', '--verbose', action='store_true',  default=False, help='Print more stuff')
parser.add_argument('-p', '--plot', action='store_true',  default=False, help='Plot reactivations (takes longer)')
parser.add_argument('--smooth_plot', default=0, type=int, help='Smoothing preds before plotting')
args = parser.parse_args()

config = importlib.import_module(f"configs.{args.config}", "Config").Config() # import config parameters
for arg in vars(config): setattr(args, arg, getattr(config, arg)) # update argparse with arguments from the config
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

print('This script lists all the .npy files in all the subjects decoding output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')
v = args.version
decoding_dir = f"Decoding_{args.in_dir}_v{v}" # if args.ovr else f"Decoding_v{v}"
if args.subject in ["all", "v1", "v2",  "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"
out_dir_plots = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/plots/"
print('\noutput files will be in: ' + out_dir)
create_folder(out_dir, args.overwrite)
create_folder(out_dir_plots, args.overwrite)

dummy_class_enc = LabelEncoder()
dummy_labbin = LabelBinarizer()
# mag_idx, grad_idx = [pickle.load(open(f"{args.root_path}/Data/{s}_indices.p", "rb")) for s in ['mag', 'grad']]
# mag_info, grad_info = [pickle.load(open(f"{args.root_path}/Data/{s}_info.p", "rb")) for s in ['mag', 'grad']]
minmaxScaler = MinMaxScaler()

## All possible training time (depends on the property that is decoded).
# train_times = ['0.8', '2.6', '0.2', '1.4', '2.0']
train_times = ["0.17", "0.2", "0.3", "0.4", "0.5", "0.6", "0.8"] + ["0.77", "0.9", "1.0", "1.1", "1.2", "1.4"] + ["1.37", "1.5", "1.6", "1.7", "1.8", "2.0"]
train_times = train_times + ["1.97", "2.1", "2.2", "2.3", "2.4", "2.6"] + ["2.57", "2.7", "2.8", "2.9", "3.0", "3.2"]
## Generalization window for objects and scenes
gen_windows = [(3, 5)] #, (1.5, 2.2)]


# list all preds.npy files in the directory
all_fns = natsorted(glob(in_dir + f'/*preds.npy'))
if not all_fns:
    raise RuntimeError(f"Did not find any preds files in {in_dir}/*preds.npy ... Did you pass the right config?")
print(f"Found {len(all_fns)} preds files")

preds_fn = f"{op.dirname(op.dirname(out_dir))}/all_preds.p"
metadata_fn = f"{op.dirname(op.dirname(out_dir))}/all_metadata.p"
all_labels = np.unique([op.basename(fn).split('-')[0] for fn in all_fns])
print(f"Found labels {all_labels}")
all_preds_data = []
all_df = []
for label in all_labels:
    if args.verbose: print(f"Doing {label}")
    for train_cond in ["localizer_one_object_two_objects", "two_objects"]: # "two_objects_localizer", "localizer", "obj", "scenes", , "localizer_two_objects"
        for split_query in [False]: # no split query in replay decoding so far (but migh wanna include it later)
            for gen_cond in ["obj", "scenes"]: # "localizer", 
                for train_time in train_times:
                    for gen_window in gen_windows:
                        if args.verbose: print(train_time)
                        mds_this_cond = []
                        for fn in all_fns:
                            if op.basename(fn)[0:len(label)+1] != f"{label}-": continue 
                            if f"cond-{train_cond}-" not in fn: continue
                            # if any([str(sub) in fn for sub in [25, 27, 11, 20]]): 
                            #     print(f"rejecting fn {fn}")
                            #     continue # bad subjects
                            if gen_cond is not None:
                                if f"#{train_time},{train_time}#{gen_window[0]},{gen_window[1]}#" not in fn: 
                                    continue
                                if f"tested_on_{gen_cond}" not in fn: continue # only generalization results
                            else: # gen_cond is None 
                                if "tested_on" in fn: continue # ensure we don't have generalization results (shouldn't be usefull after the preceeding line)
                                print(train_time, fn)
                                if f"#{train_time},{train_time}#{train_time},{train_time}#" not in fn:
                                    continue
                            # do not load the full mnius splits
                            if "full_minus_split" in fn: continue

                            if args.verbose: print('loading file ', fn)
                            preds = np.load(fn) # times * trials * classes 
                            preds = preds.squeeze() # trials * classes (single time point for replay decoding)
                            # all_preds.append(preds) # len(n_subs) of array of inhomogeneous shape n_trials * n_classes

                            metadata_fn = fn.replace('preds.npy', 'metadata.csv')
                            md = pd.read_csv(metadata_fn) # times * trials * classes 
                            n_times, n_trials, n_classes = preds.shape

                            md["train_time"] = [train_time] * n_trials
                            md["gen_window"] = [gen_window] * n_trials
                            md["train_cond"] = [train_cond] * n_trials
                            md["gen_cond"] = [gen_cond] * n_trials
                            md["split_query"] = [split_query] * n_trials
                            md["label"] = [label] * n_trials
                            sub = op.basename(op.dirname(fn))[0:2]
                            md["sub"] = [sub] * n_trials
                            # md["preds"] = preds.transpose(1,0,2).tolist()
                            all_preds_data.extend(preds.transpose(1,0,2))

                            all_df.append(md)

                            
                            np.random.seed(42) # same state to get the same trial number, that way we have the same Prop and WordPos 
                            if args.plot: # plot random activations, last second only
                                n_sec = 2 # number of second to plot (usually only the last one)
                                times = np.arange(0, n_sec + 0.0001, 1/100)
                                n_times_delay = 200 - (n_sec * 100) # when to start. 0 or 100 normally. 
                                if "PropAll" in label:
                                    preds = preds[:,:,0:7]
                                # do_rel = True if "PropAll" in label else False
                                # do_rel = True if preds.shape[2] > 6 else False
                                do_rel = False
                                for i_trial in range(n_trials):
                                    if np.random.rand() < 0.05:
                                        if md.loc[i_trial, "Perf"] == 0: continue # skip if there was an error 
                                        if "Prop" in md.loc[i_trial, "label"]:
                                            # if np.any(preds[100::, i_trial] > threshold_per_subject[int(sub)]):
                                            title = ' '.join(md.loc[i_trial, ['Shape1', 'Colour1', 'Shape2', 'Colour2']].values)
                                            out_fn = f"{out_dir_plots}/{label}-{train_cond}-{train_time.replace('.', '')}-{gen_cond}-sub-{sub}-trial-{i_trial}_activations.png"
                                            plot_reactivations(times, preds[n_times_delay::, i_trial].T, out_fn, threshold=np.mean(threshold_per_subject[int(sub)]), title=title, markevery=5, do_rel=do_rel)
                                                
                                        elif "WordPos" in label:
                                            out_fn = f"{out_dir_plots}/{label}-{train_cond}-{train_time.replace('.', '')}-{gen_cond}-sub-{sub}-trial-{i_trial}_activations_WordPos.png"
                                            plot_word_position_reactivations(times, preds[n_times_delay::, i_trial].T, out_fn, threshold=0.4, markevery=5, title='')
                                        else:
                                            raise pwet



                        if not len(all_preds_data): 
                            if args.verbose: print(f"found no file for {label} trained on {train_cond} with generalization to {gen_cond} for  split query {split_query}, train time {train_time}, gen window {gen_window}, continuing")
                            continue
                        n_subs = md["sub"].nunique()
                        
                        if args.verbose: print(f"Finished {label} trained on {train_cond} with generalization {gen_cond}  for  split query {split_query}\n")
                        plt.close('all')


df = pd.concat(all_df)
df.drop(columns=["Unnamed: 0", "Loc_word", "Word_position", "Mapping", "Matching", "Error_type", "Violated_position", "split_query", "Mismatch_side"], inplace=True)
df.reset_index(inplace=True)
df.drop(columns="index", inplace=True)
df.to_csv(f"{out_dir}/all_preds_data.csv", index=False)
# cannot save a numpy array as an entry in the df. So saving data separately, but the order matches. 

# cannot save as np array because for the heterogeneity of the dimensions. so list of trials, for each category, for all subjects
# items in the last are of shape n_times * n_classes
pickle.dump(all_preds_data, open(f"{out_dir}/all_preds_data.pkl", 'wb'))

print(f"Used {df['sub'].nunique()} subjects)")
print(f"Found {len(all_labels)} labels. Successfully loaded {df.label.nunique()} labels: {df.label.unique()}")

print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")