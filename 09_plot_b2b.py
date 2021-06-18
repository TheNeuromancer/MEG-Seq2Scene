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

from utils.b2b import *
from utils.params import *


matplotlib.rcParams.update({'font.size': 19})
matplotlib.rcParams.update({'lines.linewidth': 2})
plt.rcParams['figure.figsize'] = [12., 8.]
plt.rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser(description='MEG plotting of B2B results')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='all',help='subject name')
parser.add_argument('-o', '--out-dir', default='agg', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
# parser.add_argument('--slices', action='store_true',  default=False, help='Whether to make horizontal slice plots of single decoder')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

print('This script lists all the .npy files in all the subjects B2B output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')

if args.subject in ["all", "v1", "v2", "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/B2B_v{args.version}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/B2B_v{args.version}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/B2B_v{args.version}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"

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
all_fns = glob(in_dir + '/*betas*.npy')

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

# report = mne.report.Report()

print(all_fns)
# all_labels = np.unique([op.basename(fn).split('-')[0] for fn in all_fns])
unique_fns = np.unique([op.basename(fn) for fn in all_fns])
print(unique_fns)
for unique_fn in unique_fns:
    all_subs_betas = []
    for fn in all_fns:
        if unique_fn not in fn: continue
        print('loading file ', fn)
        all_subs_betas.append(np.load(fn))
    
    label = unique_fn.split('-')[0]    
    if len(all_subs_betas) < 2: 
        print(f"found no file for {label}")
        continue
    
    print(f"\nDoing {unique_fn}")
    out_fn = f"{out_dir}/{unique_fn.replace('.npy', '').replace('.', '')}"

    # label_fn = f'{fn.split("_all_betas.npy")[0]}_labels.p'
    label_fn = unique_fn.replace("betas", "labels").replace(".npy", ".p") 
    legend_labels = pickle.load(open(f"{op.dirname(fn)}/{label_fn}", "rb"))

    # average results from all subjects
    try:
        ave_betas = np.mean(all_subs_betas, 0)
    except:
        set_trace()
    std_betas = np.std(all_subs_betas, 0)

    tmin, tmax = tmin_tmax_dict[label.lower()]
    n_times = len(ave_betas)
    times = np.linspace(tmin, tmax, n_times)
    version = "v1" if args.subject=="v1" else "v2"

    plot_betas(betas=ave_betas, std=std_betas, times=times, labels=legend_labels, out_fn=out_fn)

    print(f"Finished {label}\n")
    plt.close('all')
