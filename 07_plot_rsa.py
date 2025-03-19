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
import importlib
from glob import glob
from mne.stats import permutation_cluster_1samp_test

from utils.RSA import *
from utils.params import *
from utils.decod import plot_diag


matplotlib.rcParams.update({'font.size': 19})
matplotlib.rcParams.update({'lines.linewidth': 2})
plt.rcParams['figure.figsize'] = [12., 8.]
plt.rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser(description='MEG plotting of RSA results')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='all',help='subject name')
parser.add_argument('-o', '--out-dir', default='agg', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--slices', action='store_true',  default=False, help='Whether to make horizontal slice plots of single decoder')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

# feat2feats = {"Shape": ['carre', 'cercle', 'triangle', 'ca', 'cl', 'tr'], "Colour": ['rouge', 'bleu', 'vert', 'vr', 'bl', 'rg']}

if args.slices:
    slices_loc = [0.17, 0.3, 0.43, 0.5, 0.72, 0.85]
    slices_one_obj = [0.2, 0.85, 1.1, 2.5, 2.75]
    slices_two_obj = [0.2, 0.85, 1.1, 2.5, 2.75, 3.5, 4., 5.]
else:
    slices = []


print('This script lists all the .npy files in all the subjects RSA output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')

if args.subject in ["all", "v1", "v2", "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/RSA_v{args.version}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/RSA_v{args.version}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/RSA_v{args.version}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"

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
all_fns = glob(in_dir + '/*all_results.p')

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

report = mne.report.Report()

print(all_fns)
all_labels = np.unique([op.basename(fn).split('_all_results.p')[0] for fn in all_fns])
print(all_labels)
for label in all_labels:
    if args.slices:
        if "localizer" in label: slices = slices_loc
        elif "one_object" in label: slices = slices_one_obj
        elif "two_objects" in label: slices = slices_two_obj
    all_subs_results = []
    all_subs_AUCs = []
    for fn in all_fns:
        # if op.basename(fn)[0:len(label)+1] != f"{label}-": continue 
        if label not in fn: continue
        print('loading file ', fn)
        try:
            all_subs_results.append(pickle.load(open(fn, "rb")))
            if "confusion" in label:
                AUC_fn = '-'.join(fn.split('-')[0:-1]) + "-_AUC.npy"
                all_subs_AUCs.append(np.load(AUC_fn))
        except EOFError:
            print("nope")
            continue
        RF_reg = True if "RF_regression" in fn else False

    if len(all_subs_results) < 2: 
        # print(f"found no file for {label} trained on {train_cond} with generalization to {gen_cond} {'matching trials only' if match else ''}")
        continue
    # set_trace()
    # continue
    print(f"\nDoing {label}")
    # average results from all subjects
    keys = all_subs_results[0].keys()
    # drop matching, its too big
    # print("\nDROPPING MATCHING\n")
    # keys = [k for k in keys if k != "Matching"]

    ave_results = {key: np.mean([res[key] for res in all_subs_results], 0) for key in keys}
    std_results = {key: np.std([res[key] for res in all_subs_results], 0) for key in keys}

    out_fn = f"{out_dir}/{label}"
    cond = label.split('-')[2]
    ylabel = label.split('_')[-1]
    factors = ave_results.keys()
    rsa_results = np.array([val for val in ave_results.values()])

    if RF_reg:
        rsa_results = np.array([x - np.median(x) for x in rsa_results])

    tmin, tmax = tmin_tmax_dict[cond]
    n_times = rsa_results.shape[1]
    print(n_times)
    times = np.linspace(tmin, tmax, n_times)
    version = "v1" if args.subject=="v1" else "v2"
    dpi = 500 if cond == "two_objects" else 200
    plot_rsa(rsa_results, factors, out_fn, times, cond=cond, data_std=std_results.values(), ylabel=ylabel, version=version, dpi=dpi)
    if len(factors) > 1:
        multi_plot_rsa(rsa_results, factors, out_fn, times, cond=cond, data_std=std_results.values(), ylabel=ylabel, version=version, dpi=dpi)

    ## AUC
    if "confusion" in label:
        plot_diag(data_mean=np.mean(all_subs_AUCs, 0), out_fn=f"{out_fn}_mean_AUC_{len(all_subs_AUCs)}subs.png", \
            data_std=np.std(all_subs_AUCs, 0), train_cond=cond, train_tmin=tmin, train_tmax=tmax, ylabel='AUC', version=version)


    print(f"Finished {label}\n")
    plt.close('all')
