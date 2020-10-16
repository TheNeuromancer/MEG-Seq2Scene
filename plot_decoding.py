import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import matplotlib.pyplot as plt

import mne
import numpy as np
from ipdb import set_trace
import argparse
import pickle
import time
import os.path as op
import os
from glob import glob
from mne.stats import permutation_cluster_1samp_test

from utils.decod import *

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'lines.linewidth': 4})
plt.rcParams['figure.figsize'] = [12., 8.]
plt.rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser(description='MEG ans SEEG plotting of decoding results')
parser.add_argument('-r', '--root-path', default='/neurospin/unicog/protocols/MEG/Seq2Scene/', help='root path')
parser.add_argument('-i', '--in-dir', default='Results/Decoding_v6/Epochs/', help='input directory')
parser.add_argument('-s', '--subject', default='theo',help='subject name')
parser.add_argument('-o', '--out-dir', default='/agg/', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--slices', action='store_true',  default=False, help='Whether to make horizontal slice plots of single decoder')
parser.add_argument('--skip_for', action='store_true',  default=False, help='Whether to skip the split queries plot (eg: for_struct=2-6)')

print(mne.__version__)
args = parser.parse_args()
print(args)

start_time = time.time()

tmin_tmax_dict = {"localizer": [-.2, 1.], "imgloc": [-.2, 1.], "one_object": [-.5, 4.], "two_objects": [-.5, 6.]}
## THIS IS PRETTY BAD, WE SHOULD STORE TMIN AND TMAX SOMEWHERE ELSE

feat2feats = {"Shape": ['carre', 'cercle', 'triangle', 'ca', 'cl', 'tr'], "Colour": ['rouge', 'bleu', 'vert', 'vr', 'bl', 'rg']}

if args.slices:
    slices = [0.18, 0.3, 0.5, 0.7, 0.9]
else:
    slices = []


print('This script lists all the .npy files in all the subjects decoding output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')

in_dir = op.join(args.root_path, args.in_dir, args.subject)
out_dir = in_dir + args.out_dir

print('\noutput files will be in: ' + out_dir)

if op.exists(out_dir): # warn and stop if args.overwrite is set to False
    print('output file already exists...')
    if args.overwrite:
        print('overwrite is set to True ... overwriting')
    else:
        print('overwrite is set to False ... exiting')
        exit()
else:
    print('Constructing output dirtectory: ', out_dir)
    os.makedirs(out_dir)

# list all .npy files in the directory
all_filenames = glob(in_dir + '/*AUC*.npy')

report = mne.report.Report()

## Trained on localizer and tested on objects and scenes
for train_cond in ["localizer", "imgloc", "one_object"]:
    for match in [True, False]:
        for feature in ["Shape", "Colour"]:
            for gen_cond in [None, "one_object", "two_objects"]:
                if train_cond == gen_cond: continue # skip when we both have one object, it is not generalization
                print(f"\nDoing training on {train_cond} with feature {feature} for generalization {gen_cond} {'matching trials only' if match else ''}")

                all_AUC = []
                for fn in all_filenames:
                    if f"-{train_cond}-" not in fn: continue
                    if match: # keep only the files where we tested on matching trials
                        if "match" not in fn: continue
                        match_str = "_match"
                    else: # do not keep these trials
                        if "match" in fn: continue
                        match_str = "_match"
                    if not np.any([feat in fn for feat in feat2feats[feature]]): continue # check that any of the feature is present 
                    if gen_cond is not None:
                        if gen_cond not in fn: continue
                    else: # ensure we don't have generalization results
                        if "tested_on" in fn: continue

                    print('loading file ', fn)
                    all_AUC.append(np.load(fn))

                if not all_AUC: 
                    print(f"found no file for feature {feature} trained on {train_cond} with generalization to {gen_cond} {'matching trials only' if match else ''}")
                    continue
                gen_str = f"_tested_on_{gen_cond}" if gen_cond is not None else ""
                out_fn = f"{out_dir}/{feature}_trained_on_{train_cond}{gen_str}{match_str}"
                all_AUC = np.array(all_AUC)
                AUC_mean = np.mean(all_AUC, 0)
                AUC_std = np.std(all_AUC, 0)

                train_tmin, train_tmax = tmin_tmax_dict[train_cond]
                if gen_cond is not None:
                    test_tmin, test_tmax = tmin_tmax_dict[gen_cond]
                else:
                    test_tmin, test_tmax = train_tmin, train_tmax
                
                n_times_train = AUC_mean.shape[0]
                n_times_test = AUC_mean.shape[1]
                times_train = np.linspace(train_tmin, train_tmax, n_times_train)
                times_test = np.linspace(test_tmin, test_tmax, n_times_test)
                
                ylabel = get_ylabel_from_fn(fn)
                is_contrast = True if (np.min(np.array(all_AUC)) < 0) or (np.max(np.array(all_AUC)) < .4) else False

                if gen_cond is None:
                    plot_diag(data_mean=AUC_mean, data_std=AUC_std, out_fn=out_fn, train_cond=train_cond, 
                        train_tmin=train_tmin, train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast)


                plot_GAT(data_mean=AUC_mean, out_fn=out_fn, train_cond=train_cond, train_tmin=train_tmin, train_tmax=train_tmax, test_tmin=test_tmin, 
                         test_tmax=test_tmax, ylabel=ylabel, contrast=is_contrast, gen_cond=gen_cond, slices=slices)

                print(f"Finished {train_cond} with feature {feature} for generalization {gen_cond} {'matching trials only' if match else ''}\n")


##########################################################################################################################################
## Trained on objects and tested on scenes
all_AUC = []
for match in [True, False]:
    for gen_cond in [None, "two_objects"]:
        print(f"\nDoing training on individual objects with generalization to {gen_cond} {'matching trials only' if match else ''}")
        all_AUC = []
        for fn in all_filenames:
            if f"-one_object-" not in fn: continue
            if "S" not in fn or "C"not in fn: continue
            if match: # keep only the files where we tested on matching trials
                if "match" not in fn: continue
                match_str = "_match"
            else: # do not keep these trials
                if "match" in fn: continue
                match_str = "_match"
            if gen_cond is not None:
                if gen_cond not in fn: continue
            else: # ensure we don't have generalization results
                if "tested_on" in fn: continue

            print('doing file ', fn)
            all_AUC.append(np.load(fn))
        
        if not all_AUC: 
            print(f"found no file for individual objects with generalization to {gen_cond} {'matching trials only' if match else ''}")
            continue
        gen_str = f"_tested_on_{gen_cond}" if gen_cond is not None else ""
        out_fn = f"{out_dir}/individual_objects{gen_str}{match_str}"
        all_AUC = np.array(all_AUC)
        AUC_mean = np.mean(all_AUC, 0)
        AUC_std = np.std(all_AUC, 0)

        train_tmin, train_tmax = tmin_tmax_dict["one_object"]
        if gen_cond is not None:
            test_tmin, test_tmax = tmin_tmax_dict[gen_cond]
        else:
            test_tmin, test_tmax = train_tmin, train_tmax

        n_times_train = AUC_mean.shape[0]
        n_times_test = AUC_mean.shape[1]
        times_train = np.linspace(train_tmin, train_tmax, n_times_train)
        times_test = np.linspace(test_tmin, test_tmax, n_times_test)

        ylabel = get_ylabel_from_fn(fn)
        is_contrast = True if (np.min(np.array(all_AUC)) < 0) or (np.max(np.array(all_AUC)) < .4) else False


        if gen_cond is None:
            plot_diag(data_mean=AUC_mean, data_std=AUC_std, out_fn=out_fn, train_cond="one_object", 
                train_tmin=train_tmin, train_tmax=train_tmax, ylabel=ylabel, contrast=is_contrast)


        plot_GAT(data_mean=AUC_mean, out_fn=out_fn, train_cond="one_object", train_tmin=train_tmin, train_tmax=train_tmax, 
            test_tmin=test_tmin, test_tmax=test_tmax, ylabel=ylabel, contrast=is_contrast, gen_cond=gen_cond, slices=slices)

        print(f"\nFiished individual objects with generalization to {gen_cond} {'matching trials only' if match else ''}\n")

        # evo_report.add_figs_to_section(fig, captions=cond, section=cond)
        
        plt.close('all')

# report.save(out_dir + 'epochs_report.html', open_browser=False, overwrite=True)