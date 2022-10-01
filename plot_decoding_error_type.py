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
from natsort import natsorted
from scipy.stats import sem

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
parser.add_argument('--ovr', action='store_true',  default=False, help='Whether to get the one versus rest directory or classic decoding')
parser.add_argument('-r', '--response_lock', action='store_true',  default=None, help='Whether to Use response locked epochs or classical stim-locked')

args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()


decoding_dir = f"Decoding_ovr_v{args.version}" if args.ovr else f"Decoding_v{args.version}"
if args.subject in ["all", "v1", "v2",  "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"

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
all_fns = natsorted(glob(in_dir + '/*AUC*.npy'))
if args.response_lock: # keep only response lock files
    all_fns = [fn for fn in all_fns if "Resp" in fn]
    resp_str = "Resp_"
else: # keep non-response locked files
    all_fns = [fn for fn in all_fns if "Resp" not in fn]
    resp_str = ""

if not all_fns:
    raise RuntimeError(f"Did not find any AUC files in {in_dir}/*AUC*.npy ... Did you pass the right config?")

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


## ONE OBJECT
try:
    train_tmin, train_tmax = tmin_tmax_dict["response_locked" if args.response_lock else "obj"]
    word_onsets, image_onset = get_onsets("response_locked" if args.response_lock else "obj", version=version)
    cmis_fns = [fn for fn in all_fns if "CMismatch" in fn]
    if not cmis_fns: print("did not found any color mismatch file ...")
    aucs = [np.load(fn) for fn in cmis_fns]
    # print([x.shape for x in aucs])
    cmis_auc = np.diag(np.mean(aucs, 0))

    smis_fns = [fn for fn in all_fns if "SMismatch" in fn]
    smis_auc = np.diag(np.mean([np.load(fn) for fn in smis_fns], 0))

    n_times = smis_auc.shape[0]
    times = np.linspace(train_tmin, train_tmax, n_times)

    fig, ax = plt.subplots()
    lines = ax.plot(times, cmis_auc, label='Mismatch on Color')
    # ax.fill_between(times, cmis_auc-data_std_diag, data_mean_diag+data_std_diag, alpha=0.2)
    ax.plot(times, smis_auc, label='Mismatch on Shape')
    plt.legend()
    if not args.response_lock:
        ax.set_xlim(image_onset[0]-.2)
        ax.axvline(x=image_onset[0], linestyle='-', color='k')
    ax.axhline(y=0.5, color='k', linestyle='-', alpha=.5)
    plt.ylabel("AUC")
    plt.xlabel("Time (s)")

    out_fn = f"{out_dir}/{resp_str}OneObjMismatches_{len(cmis_fns)}ave"
    plt.savefig(f'{out_fn}_AUC_diag.png')
    plt.close()
except:
    pass


## TWO OBJECTS - separate decoders
# try:   
if True: 
    if args.ovr:
        pmis_fns = [fn for fn in all_fns if "PropMismatch" in fn and not "Resp" in fn and not "_for_" in fn]
        pmis_auc = np.diag(np.mean([np.load(fn) for fn in pmis_fns], 0))
        pmis_auc_sem = np.diag(sem([np.load(fn) for fn in pmis_fns], 0))

        bmis_fns = [fn for fn in all_fns if "BindMismatch" in fn and not "Resp" in fn and not "_for_" in fn]
        bmis_auc = np.diag(np.mean([np.load(fn) for fn in bmis_fns], 0))
        bmis_auc_sem = np.diag(sem([np.load(fn) for fn in bmis_fns], 0))

        rmis_fns = [fn for fn in all_fns if "RelMismatch" in fn and not "Resp" in fn and not "_for_" in fn]
        rmis_auc = np.diag(np.mean([np.load(fn) for fn in rmis_fns], 0))
        rmis_auc_sem = np.diag(sem([np.load(fn) for fn in rmis_fns], 0))

    else: # classical decoding
        pmis_fns = [fn for fn in all_fns if "PropMismatch" in fn]
        pmis_auc = np.diag(np.mean([np.load(fn) for fn in pmis_fns], 0))
        pmis_auc_sem = np.diag(sem([np.load(fn) for fn in pmis_fns], 0))

        bmis_fns = [fn for fn in all_fns if "BindMismatch" in fn]
        bmis_auc = np.diag(np.mean([np.load(fn) for fn in bmis_fns], 0))
        bmis_auc_sem = np.diag(sem([np.load(fn) for fn in bmis_fns], 0))

        rmis_fns = [fn for fn in all_fns if "RelMismatch" in fn]
        rmis_auc = np.diag(np.mean([np.load(fn) for fn in rmis_fns], 0))
        rmis_auc_sem = np.diag(sem([np.load(fn) for fn in rmis_fns], 0))

    train_tmin, train_tmax = tmin_tmax_dict["response_locked" if args.response_lock else "scenes"]
    word_onsets, image_onset = get_onsets("response_locked" if args.response_lock else "scenes", version=version)
    n_times = pmis_auc.shape[0]
    times = np.linspace(train_tmin, train_tmax, n_times)

    fig, ax = plt.subplots()
    ax.plot(times, pmis_auc, label='Mismatch on Property')
    ax.plot(times, bmis_auc, label='Mismatch on Binding')
    ax.plot(times, rmis_auc, label='Mismatch on Relation')
    plt.legend()
    if not args.response_lock:
        ax.set_xlim(image_onset[0]-.2, 7.)
        ax.axvline(x=image_onset[0], linestyle='-', color='k')
    ax.axhline(y=0.5, color='k', linestyle='-', alpha=.5)
    plt.ylabel("AUC")
    plt.xlabel("Time (s)")

    out_fn = f"{out_dir}/{resp_str}TwoObjMismatches_{len(pmis_fns)}ave"
    plt.savefig(f'{out_fn}_AUC_diag.png')
# except:
#     print(f"\n\nCould not find files PropMismatch, BindMismatch, RelMismatch, moving on\n\n")


# if args.ovr: ## actually no single deecodier and split queries for OVR
#     pmis_fns = [fn for fn in all_fns if "PropMismatch"]
#     pmis_auc = np.diag(np.mean([np.load(fn) for fn in pmis_fns], 0))
#     pmis_auc_sem = np.diag(sem([np.load(fn) for fn in pmis_fns], 0))

#     bmis_fns = [fn for fn in all_fns if "BindMismatch"]
#     bmis_auc = np.diag(np.mean([np.load(fn) for fn in bmis_fns], 0))
#     bmis_auc_sem = np.diag(sem([np.load(fn) for fn in bmis_fns], 0))

#     rmis_fns = [fn for fn in all_fns if "RelMismatch"]
#     rmis_auc = np.diag(np.mean([np.load(fn) for fn in rmis_fns], 0))
#     rmis_auc_sem = np.diag(sem([np.load(fn) for fn in rmis_fns], 0))

# else: # classical decoding
## TWO OBJECTS - single decoder and slit queries for each mismatch
try:
    pmis_fns = [fn for fn in all_fns if "Matching" in fn and "Error_type='l0'" in fn]
    pmis_auc = np.diag(np.mean([np.load(fn) for fn in pmis_fns], 0))
    pmis_auc_sem = np.diag(sem([np.load(fn) for fn in pmis_fns], 0))

    bmis_fns = [fn for fn in all_fns if "Matching" in fn and "Error_type='l1'" in fn]
    bmis_auc = np.diag(np.mean([np.load(fn) for fn in bmis_fns], 0))
    bmis_auc_sem = np.diag(sem([np.load(fn) for fn in bmis_fns], 0))

    rmis_fns = [fn for fn in all_fns if "Matching" in fn and "Error_type='l2'" in fn]
    rmis_auc = np.diag(np.mean([np.load(fn) for fn in rmis_fns], 0))
    rmis_auc_sem = np.diag(sem([np.load(fn) for fn in rmis_fns], 0))

    train_tmin, train_tmax = tmin_tmax_dict["response_locked" if args.response_lock else "scenes"]
    word_onsets, image_onset = get_onsets("response_locked" if args.response_lock else "scenes", version=version)
    n_times = pmis_auc.shape[0]
    times = np.linspace(train_tmin, train_tmax, n_times)

    fig, ax = plt.subplots()
    ax.plot(times, pmis_auc, label='Mismatch on Property')
    ax.plot(times, bmis_auc, label='Mismatch on Binding')
    ax.plot(times, rmis_auc, label='Mismatch on Relation')
    plt.legend()
    if not args.response_lock:
        ax.set_xlim(image_onset[0]-.2, 7.)
        ax.axvline(x=image_onset[0], linestyle='-', color='k')
    ax.axhline(y=0.5, color='k', linestyle='-', alpha=.5)
    plt.ylabel("AUC")
    plt.xlabel("Time (s)")

    out_fn = f"{out_dir}/{resp_str}TwoObjMismatches_split_query_{len(pmis_fns)}ave"
    plt.savefig(f'{out_fn}_AUC_diag.png')
except:
    pass




## TWO OBJECTS - single decoder and slit queries for complexity
train_tmin, train_tmax = tmin_tmax_dict["response_locked" if args.response_lock else "scenes"]
word_onsets, image_onset = get_onsets("response_locked" if args.response_lock else "scenes", version=version)
pmis_fns = [fn for fn in all_fns if "Matching" in fn and "Complexity=0" in fn]
pmis_auc = np.diag(np.mean([np.load(fn) for fn in pmis_fns], 0))
pmis_auc_sem = np.diag(sem([np.load(fn) for fn in pmis_fns], 0))

bmis_fns = [fn for fn in all_fns if "Matching" in fn and "Complexity=1" in fn]
bmis_auc = np.diag(np.mean([np.load(fn) for fn in bmis_fns], 0))
bmis_auc_sem = np.diag(sem([np.load(fn) for fn in bmis_fns], 0))

rmis_fns = [fn for fn in all_fns if "Matching" in fn and "Complexity=2" in fn]
rmis_auc = np.diag(np.mean([np.load(fn) for fn in rmis_fns], 0))
rmis_auc_sem = np.diag(sem([np.load(fn) for fn in rmis_fns], 0))

n_times = pmis_auc.shape[0]
times = np.linspace(train_tmin, train_tmax, n_times)

fig, ax = plt.subplots()
ax.plot(times, pmis_auc, label='Complexity = 0')
ax.plot(times, bmis_auc, label='Complexity = 1')
ax.plot(times, rmis_auc, label='Complexity = 2')
plt.legend()
if not args.response_lock:
    ax.set_xlim(image_onset[0]-.2, 7.)
    ax.axvline(x=image_onset[0], linestyle='-', color='k')
ax.axhline(y=0.5, color='k', linestyle='-', alpha=.5)
plt.ylabel("AUC")
plt.xlabel("Time (s)")

out_fn = f"{out_dir}/{resp_str}TwoObjMismatches_split_query_complexity_{len(pmis_fns)}ave"
plt.savefig(f'{out_fn}_AUC_diag.png')


print("ALL DONE")