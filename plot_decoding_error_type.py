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
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()


decoding_dir = f"Decoding_v{args.version}"
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


# ## ONE OBJECT
# train_tmin, train_tmax = tmin_tmax_dict["obj"]
# word_onsets, image_onset = get_onsets("obj", version=version)
# cmis_fns = [fn for fn in all_fns if "CMismatch" in fn]
# if not cmis_fns: print("did not found any color mismatch file ...")
# cmis_auc = np.diag(np.mean([np.load(fn) for fn in cmis_fns], 0))

# smis_fns = [fn for fn in all_fns if "SMismatch" in fn]
# smis_auc = np.diag(np.mean([np.load(fn) for fn in smis_fns], 0))

# n_times = smis_auc.shape[0]
# times = np.linspace(train_tmin, train_tmax, n_times)

# fig, ax = plt.subplots()
# lines = ax.plot(times, cmis_auc, label='Mismatch on Color')
# # ax.fill_between(times, cmis_auc-data_std_diag, data_mean_diag+data_std_diag, alpha=0.2)
# ax.plot(times, smis_auc, label='Mismatch on Shape')
# plt.legend()
# ax.set_xlim(image_onset[0]-.2)
# ax.axvline(x=image_onset[0], linestyle='-', color='k')
# ax.axhline(y=0.5, color='k', linestyle='-', alpha=.5)
# plt.ylabel("AUC")
# plt.xlabel("Time (s)")

# out_fn = f"{out_dir}/OneObjMismatches_{len(cmis_fns)}ave"
# plt.savefig(f'{out_fn}_AUC_diag.png')
# plt.close()


## TWO OBJECTS
train_tmin, train_tmax = tmin_tmax_dict["scenes"]
word_onsets, image_onset = get_onsets("scenes", version=version)
pmis_fns = [fn for fn in all_fns if "PropMismatch" in fn]
pmis_auc = np.diag(np.mean([np.load(fn) for fn in pmis_fns], 0))
pmis_auc_sem = np.diag(sem([np.load(fn) for fn in pmis_fns], 0))

bmis_fns = [fn for fn in all_fns if "BindMismatch" in fn]
bmis_auc = np.diag(np.mean([np.load(fn) for fn in bmis_fns], 0))
bmis_auc_sem = np.diag(sem([np.load(fn) for fn in bmis_fns], 0))

rmis_fns = [fn for fn in all_fns if "RelMismatch" in fn]
rmis_auc = np.diag(np.mean([np.load(fn) for fn in rmis_fns], 0))
rmis_auc_sem = np.diag(sem([np.load(fn) for fn in rmis_fns], 0))

n_times = pmis_auc.shape[0]
times = np.linspace(train_tmin, train_tmax, n_times)

fig, ax = plt.subplots()
ax.plot(times, pmis_auc, label='Mismatch on Property')
ax.plot(times, bmis_auc, label='Mismatch on Binding')
ax.plot(times, rmis_auc, label='Mismatch on Relation')
plt.legend()
ax.set_xlim(image_onset[0]-.2)
ax.axvline(x=image_onset[0], linestyle='-', color='k')
ax.axhline(y=0.5, color='k', linestyle='-', alpha=.5)
plt.ylabel("AUC")
plt.xlabel("Time (s)")

out_fn = f"{out_dir}/TwoObjMismatches_{len(pmis_fns)}ave"
plt.savefig(f'{out_fn}_AUC_diag.png')

print("ALL DONE")