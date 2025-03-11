import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import matplotlib.pyplot as plt

import mne
import numpy as np
from ipdb import set_trace
import argparse
import pickle
import time
import os.path as op
import os
import importlib
from glob import glob
from natsort import natsorted
from itertools import permutations, combinations
import pandas as pd
from copy import deepcopy

from utils.decod import *
from utils.params import *

matplotlib.rcParams.update({'font.size': 19})
matplotlib.rcParams.update({'lines.linewidth': 2})
plt.rcParams['figure.figsize'] = [12., 8.]
plt.rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser(description='MEG plotting of same vs different trial analysis results')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='all',help='subject name')
parser.add_argument('-o', '--out-dir', default='agg', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('-k', '--kind',  default='corr', help='corr of decod or ccaR or ccaR2')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

assert args.kind in ["corr", "decod", "ccaR", "ccaR2"], f"args.kind should be one of ['corr', 'decod', 'ccaR', 'ccaR2'], but is {args.kind}"

start_time = time.time()

feat2feats = {"Shape": ['carre', 'cercle', 'triangle', 'ca', 'cl', 'tr'], "Colour": ['rouge', 'bleu', 'vert', 'vr', 'bl', 'rg']}

print('This script lists all the decoding window results files in all the subjects decoding output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')
_dir = f"Correlation_v{args.version}" if args.kind=='corr' else f"Same_Diff_Decod_v{args.version}" if args.kind=='decod' else f"Same_Diff_CCA_v{args.version}"
if args.subject in ["all", "v1", "v2",  "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/{_dir}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/{_dir}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/{_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"

print('\noutput files will be in: ' + out_dir)

if op.exists(out_dir): # warn and stop if args.overwrite is set to False
    print('output file already exists...')
    if args.overwrite:
        print('overwrite is set to True ... overwriting')
    else:
        print('overwrite is set to False ... exiting')
        exit()
else:
    print('Constructing output dirtectory: ', out_dir)
    os.makedirs(out_dir)

# list all .npy files in the directory
all_fns = natsorted(glob(in_dir + '/*.npy'))
if args.kind == "ccaR":
    print("keeping only R files, rejecting R2")
    all_fns = [fn for fn in all_fns if "R.npy" in fn]
elif args.kind == "ccaR2":
    print("keeping only R2 files, rejecting R")
    all_fns = [fn for fn in all_fns if "R2.npy" in fn]
print(f"Found {len(all_fns)} files")

# keep the first 8 subjects for the 1st version, all the remaining for v2
if args.subject == "v1":
    all_fns = [fn for fn in all_fns if int(op.basename(op.dirname(fn))[0:2]) < 9]
    version = "v1"
elif args.subject == "v2":
    all_fns = [fn for fn in all_fns if int(op.basename(op.dirname(fn))[0:2]) > 8]
    version = "v2"
elif args.subject == "all":
    version = "v2"
elif args.subject == "goods":
    all_fns = [fn for fn in all_fns if not op.basename(op.dirname(fn))[0:2] in bad_subjects]
    version = "v2"
elif int(args.subject[0:2]) < 9:
    version = "v1"
elif int(args.subject[0:2]) > 8:
    version = "v2"
else:
    qwe


data_dict = {"Mirror": [], "Window": [], "Subject": [], "Matching": []}
if args.kind == "ccaR": Rs = {}
for mirror in (True, False):
    for window in (True, False):
        for reducdim in (True, False): # for decod only
            for matching in (True, False): # for CCA only
                data, all_subs = [], []
                for fn in all_fns:
                    if mirror and not ("mirror" in fn): continue
                    if not mirror and "mirror" in fn: continue
                    if window and not "#" in fn: continue
                    if not window and "#" in fn: continue
                    if reducdim and not "reducdim" in fn: continue
                    if not reducdim and "reducdim" in fn: continue
                    if matching and not "_match_" in fn: continue
                    if not matching and "_match_" in fn: continue
                    # print('loading file ', fn)
                    data.append(np.load(fn))
                    all_subs.append(op.basename(op.dirname(fn))[0:2])

                    data_dict["Mirror"] = "Yes" if mirror else "No"
                    data_dict["Window"] = "Yes" if window else "No"
                    data_dict["Matching"] = "Yes" if matching else "No"
                    data_dict["Subject"] = op.basename(op.dirname(fn))[0:2]

                if not data:
                    print(f"did not find any data for mirror={mirror} and window={window}, and reducdim={reducdim} and matching={matching}, moving on")
                    continue
                print(f"Doing  mirror={mirror} and window={window}, and reducdim={reducdim} and matching={matching}, found {len(all_subs)} subjects")
                
                # set_trace()
                data = np.array(data)
                print(data.shape)
                mirror_str = "_mirror" if mirror else ""
                window_str = "_window" if window else ""
                matching_str = "_matching" if matching else ""
                # print(f"\nDoing {mirror_str} and ")
                out_fn = f"{out_dir}/{len(data)}ave{mirror_str}{window_str}{matching_str}"
                n_subs = len(all_subs)

                if args.kind == "corr": # n_subs*nchan
                    # data = data.mean(1)
                    # pval = wilcoxon(data[:,0], data[:,1], alternative="two-sided")[1] #two-sided
                    # print(f"Correlation for mirror={mirror} and window={window}: R = {np.mean(data[:,0]):.5f} for match and R = {np.mean(data[:,1]):.5f} for mismatch -- pval = {pval:3f}")
                    set_trace()
                    pvals = []
                    for ch in range(data.shape[1]): # test each channel
                        ch_data = data[:,ch,:]
                        pval = wilcoxon(ch_data[:,0], ch_data[:,1], alternative="two-sided")[1] #two-sided
                        pvals.append(pval)
                    corrected_pvals = fdr_correction(np.array(pvals), alpha=0.5)[1]
                    for ch, pval in zip(range(data.shape[1]), corrected_pvals):
                        if pval < 0.1:
                            ch_data = data[:,ch,:]
                            print(f"Correlation for ch {ch}: R = {np.mean(ch_data[:,0]):.5f} for match and R = {np.mean(ch_data[:,1]):.5f} for mismatch -- pval = {pval:3f}")

                elif args.kind == "decod":
                    pval = wilcoxon(data, np.ones_like(data)*0.5, alternative="greater")[1] #two-sided
                    print(f"Decoding perf: AUC = {np.mean(data):.3f}, pval={pval}")

                elif args.kind == "ccaR2": # CCA, data is an array of R of size n_subs*n_times
                    # set_trace()
                    chance = np.zeros_like(data[:,0])
                    for t in range(data.shape[1]):
                        pval = wilcoxon(data[:,t], chance, alternative="greater")[1] #two-sided
                        # if pval < 0.05: 
                        print(f"R2 after CCA at time {t}: R2 = {np.mean(data[:,t]):.3f}, pval={pval}")

                elif args.kind == "ccaR":  # CCA, data is an array of R of size n_subs*n_times*nchan
                    Rs[f"{mirror_str}-{matching_str}"] = data

                    plt.imshow(data.mean(0))
                    plt.savefig(out_fn)
                    plt.close()

                    tmin, tmax = tmin_tmax_dict["scenes"]
                    plot_GAT(data_mean=data.mean(0), out_fn=out_fn, train_cond='scenes', train_tmin=tmin, train_tmax=tmax, test_tmin=tmin, ybar=0,
                             test_tmax=tmax, ylabel="R", contrast=True, resplock=False, gen_cond=None, slices=None, version=version, window=None)
                    plt.close() 

                    # pass # just do aggregate analyses
                    # _, n_times, nchan = data.shape
                    # pvals = np.zeros((n_times, nchan))
                    # chance = np.zeros_like(data[:,0,0])
                    # for t in range(n_times):
                    #     for ch in range(nchan):
                    #         pval = wilcoxon(data[:,t, ch], chance, alternative="greater")[1] #two-sided
                    #         pvals[t, ch] = pval
                    #         # if pval < 0.05: 
                    #         #     print(f"Correlation after CCA at time {t}, channel {ch}: R = {np.mean(data[:,t,ch]):.3f}, pval={pval}")
                    # corrected_pvals = fdr_correction(pvals.ravel(), alpha=0.05)[1].reshape((n_times, nchan))
                    # for t in range(n_times):
                    #     for ch in range(nchan):
                    #         if pvals[t, ch] < 0.05:
                    #             print(f"Correlation after CCA at time {t}, channel {ch}: R = {np.mean(data[:,t,ch]):.3f}, pval={pvals[t, ch]}")





if args.kind == "ccaR": 
    _, n_times, nchan = Rs["_mirror-_matching"].shape
    # pvals = np.zeros((n_times, nchan))
    # for t in range(n_times):
    #     for ch in range(nchan):
    #         pval = wilcoxon(Rs["_mirror-_matching"][:,t, ch], Rs["_mirror-"][:,t, ch], alternative="greater")[1] #two-sided
    #         pvals[t, ch] = pval
    #         # if pval < 0.05: 
    #         #     print(f"Correlation after CCA (mirror) at time {t}, channel {ch}: R = {Rs['_mirror-_matching'][:,t,ch].mean():.3f} vs {Rs['_mirror-'][:,t,ch].mean():.3f}, pval={pval}")
    # corrected_pvals = fdr_correction(pvals.ravel(), alpha=0.01)[1].reshape((n_times, nchan))
    # for t in range(n_times):
    #     for ch in range(nchan):
    #         if pvals[t, ch] < 0.01:
    #             print(f"Correlation after CCA (mirror) at time {t}, channel {ch}: R = {Rs['_mirror-_matching'][:,t,ch].mean():.3f} vs {Rs['_mirror-'][:,t,ch].mean():.3f}, pval={pvals[t, ch]}")

    # for t in range(n_times):
    #     for ch in range(nchan):
    #         pval = wilcoxon(Rs["-_matching"][:,t,ch], Rs["-"][:,t,ch], alternative="greater")[1] #two-sided
    #         pvals[t, ch] = pval
    #         # if pval < 0.01: 
    #         #     print(f"Correlation after CCA (nonmirror) at time {t}, channel {ch}: R = {Rs['-_matching'][:,t,ch].mean():.3f} vs {Rs['-'][:,t,ch].mean():.3f}, pval={pval}")
    # corrected_pvals = fdr_correction(pvals.ravel(), alpha=0.01)[1].reshape((n_times, nchan))
    # for t in range(n_times):
    #     for ch in range(nchan):
    #         if pvals[t, ch] < 0.01:
    #             print(f"Correlation after CCA (nonmirror) at time {t}, channel {ch}: R = {Rs['-_matching'][:,t,ch].mean():.3f} vs {Rs['-'][:,t,ch].mean():.3f}, pval={pvals[t, ch]}")


    fig, ax = plt.subplots()
    times = np.linspace(tmin, tmax, n_times)
    ax.plot(times, np.median(np.mean(Rs["_mirror-_matching"], 0), 1), label="Mirror Matching")
    ax.plot(times, np.median(np.mean(Rs["_mirror-"], 0), 1), label="Mirror NOT Matching")
    ax.plot(times, np.median(np.mean(Rs["-_matching"], 0), 1), label="NOT Mirror Matching")
    ax.plot(times, np.median(np.mean(Rs["-"], 0), 1), label="NOT Mirror NOT Matching")
    plt.legend()
    fig.savefig(f"{out_dir}/{len(data)}_median_ch_R.png")

    set_trace()
    # # for mirror in (True, False):
    # #     for matching in (True, False): # for CCA only
    #         mirror_str = "_mirror" if mirror else ""
    #         matching_str = "_matching" if matching else ""
    #         dat = Rs[f"{mirror_str}-{matching_str}"]

    
# df = pd.DataFrame.from_dict(data_dict)
# # df_all_labels.append(deepcopy(df))

    # print(df["Trained on"].unique())
    # print(df["Tested on"].unique())
    # if "Cfull" in label or "Cdelay" in label or "SideC_delay" in label:
    #     cond_str1 = "Colour1"
    #     cond_str2 = "Colour2"
    # elif "Sfull" in label or "Sdelay" in label or "SideS_delay" in label:
    #     cond_str1 = "Shape1"
    #     cond_str2 = "Shape2"
    # else:
    #     cond_str1 = "Shape1+Colour1"
    #     cond_str2 = "Shape2+Colour2"
    
    # box_pairs = []
    # # if "Side" in label:
    # #     box_pairs += [(("Shape1+Colour1-one_object", "Shape1+Colour1-one_object"), ("Shape1+Colour1-one_object", "Right_obj-two_objects"))]
    # #     box_pairs += [(("Shape1+Colour1-one_object", "Shape1+Colour1-one_object"), ("Shape1+Colour1-one_object", "Left_obj-two_objects"))]
    # #     # box_pairs += [(("Shape1+Colour1-one_object", "Shape1+Colour1-one_object"), ("Shape1+Colour1-one_object", "Left_obj-two_objects"))]
    # # else:
    # if "SideC" in label:
    #     continue
    #     box_pairs += [(("Left_color-two_objects", "Left_color-two_objects"), ("Left_color-two_objects", "Right_color-two_objects"))]
    # elif "SideS" in label:
    #     box_pairs += [(("Left_shape-two_objects", "Left_shape-two_objects"), ("Left_shape-two_objects", "Right_shape-two_objects"))]
    # else:
    #     box_pairs += [((f"{cond_str1}-two_objects", f"{cond_str1}-two_objects"), (f"{cond_str1}-two_objects", f"{cond_str1}-one_object"))]
    #     box_pairs += [((f"{cond_str1}-two_objects", f"{cond_str1}-two_objects"), (f"{cond_str1}-two_objects", f"{cond_str2}-two_objects"))]
    #     box_pairs += [((f"{cond_str1}-one_object", f"{cond_str1}-one_object"), (f"{cond_str1}-one_object", f"{cond_str1}-two_objects"))]
    #     box_pairs += [((f"{cond_str1}-one_object", f"{cond_str1}-one_object"), (f"{cond_str1}-one_object", f"{cond_str2}-two_objects"))]
    #     box_pairs += [((f"{cond_str2}-two_objects", f"{cond_str2}-two_objects"), (f"{cond_str2}-two_objects", f"{cond_str1}-two_objects"))]
    #     box_pairs += [((f"{cond_str2}-two_objects", f"{cond_str2}-two_objects"), (f"{cond_str2}-two_objects", f"{cond_str1}-one_object"))]
    
    # make_sns_barplot(df, x='Tested on', y='AUC', hue='Trained on', box_pairs=box_pairs, out_fn=f'{out_fn}_AUC.png', hline=0.5, ymin=0.45)
    
print("ALL DONE")
