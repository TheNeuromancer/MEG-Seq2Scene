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
from mne.stats import permutation_cluster_1samp_test
from prettytable import PrettyTable
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from itertools import permutations, combinations
import pandas as pd
from copy import deepcopy

from utils.decod import *
from utils.params import *

matplotlib.rcParams.update({'font.size': 19})
matplotlib.rcParams.update({'lines.linewidth': 2})
plt.rcParams['figure.figsize'] = [12., 8.]
plt.rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser(description='MEG ans SEEG plotting of decoding results')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='01_js180232',help='subject name')
parser.add_argument('-o', '--out-dir', default='agg', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--ovr', action='store_true',  default=False, help='Whether to get the one versus rest directory or classic decoding')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

feat2feats = {"Shape": ['carre', 'cercle', 'triangle', 'ca', 'cl', 'tr'], "Colour": ['rouge', 'bleu', 'vert', 'vr', 'bl', 'rg']}

print('This script lists all the decoding window results files in all the subjects decoding output directories, takes the set of this and the averages all unique filenames to get on plot for all subjects per condition')
decoding_dir = f"Decoding_window_v{args.version}"
if args.subject in ["all", "v1", "v2",  "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"

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
all_fns = natsorted(glob(in_dir + '/*AUC.p'))

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

dummy_class_enc = LabelEncoder()
dummy_labbin = LabelBinarizer()
cmap10 = plt.cm.get_cmap('tab10', 10)

all_labels = np.unique([op.basename(fn).split('-')[0] for fn in all_fns])
df_all_labels = []

for label in all_labels:
    all_AUC = []
    all_AUC_allt = []
    all_accuracy = []
    all_cosines = []
    all_subs = []
    for fn in all_fns:
        if op.basename(fn)[0:len(label)+1] != f"{label}-": continue 
        print('loading file ', fn)
        all_AUC.append(pickle.load(open(fn, "rb")))
        all_accuracy.append(pickle.load(open(f"{fn[0:-5]}accuracy.p", "rb")))
        if op.exists(f"{fn[0:-5]}cosine.p"):
            all_cosines.append(pickle.load(open(f"{fn[0:-5]}cosine.p", "rb")))
        if op.exists(f"{fn[0:-2]}_allt.p"):
            all_AUC_allt.append(pickle.load(open(f"{fn[0:-2]}_allt.p", "rb")))
        all_subs.append(op.basename(op.dirname(fn))[0:2])

    print(f"\nDoing {label}")
    out_fn = f"{out_dir}/{label}_{len(all_AUC)}ave"

    all_subs_labels = np.unique(all_subs)
    all_subs = dummy_class_enc.fit_transform(all_subs)
    n_subs = len(all_subs)

    conds = natsorted([x for x in all_AUC[0].keys()])
    n_conds = len(conds)

    auc_dict = {"Trained on": [], "Tested on": [], "AUC": [], "subject": [], "label":[]}
    for train_cond in conds:
        for test_cond in conds:
            for i_sub in range(n_subs):
                auc_dict["Trained on"].append(train_cond)
                auc_dict["Tested on"].append(test_cond)
                auc_dict["AUC"].append(all_AUC[i_sub][train_cond][test_cond])
                auc_dict["subject"].append(i_sub+1)
                auc_dict["label"].append(label)

    df = pd.DataFrame.from_dict(auc_dict)
    df_all_labels.append(deepcopy(df))

    print(df["Trained on"].unique())
    print(df["Tested on"].unique())
    if "Cfull" in label or "Cdelay" in label or "SideC_delay" in label:
        cond_str1 = "Colour1"
        cond_str2 = "Colour2"
    elif "Sfull" in label or "Sdelay" in label or "SideS_delay" in label:
        cond_str1 = "Shape1"
        cond_str2 = "Shape2"
    else:
        cond_str1 = "Shape1+Colour1"
        cond_str2 = "Shape2+Colour2"
    
    box_pairs = []
    # if "Side" in label:
    #     box_pairs += [(("Shape1+Colour1-one_object", "Shape1+Colour1-one_object"), ("Shape1+Colour1-one_object", "Right_obj-two_objects"))]
    #     box_pairs += [(("Shape1+Colour1-one_object", "Shape1+Colour1-one_object"), ("Shape1+Colour1-one_object", "Left_obj-two_objects"))]
    #     # box_pairs += [(("Shape1+Colour1-one_object", "Shape1+Colour1-one_object"), ("Shape1+Colour1-one_object", "Left_obj-two_objects"))]
    # else:
    if "SideC" in label:
        continue
        box_pairs += [(("Left_color-two_objects", "Left_color-two_objects"), ("Left_color-two_objects", "Right_color-two_objects"))]
    elif "SideS" in label:
        box_pairs += [(("Left_shape-two_objects", "Left_shape-two_objects"), ("Left_shape-two_objects", "Right_shape-two_objects"))]
    else:
        box_pairs += [((f"{cond_str1}-two_objects", f"{cond_str1}-two_objects"), (f"{cond_str1}-two_objects", f"{cond_str1}-one_object"))]
        box_pairs += [((f"{cond_str1}-two_objects", f"{cond_str1}-two_objects"), (f"{cond_str1}-two_objects", f"{cond_str2}-two_objects"))]
        box_pairs += [((f"{cond_str1}-one_object", f"{cond_str1}-one_object"), (f"{cond_str1}-one_object", f"{cond_str1}-two_objects"))]
        box_pairs += [((f"{cond_str1}-one_object", f"{cond_str1}-one_object"), (f"{cond_str1}-one_object", f"{cond_str2}-two_objects"))]
        box_pairs += [((f"{cond_str2}-two_objects", f"{cond_str2}-two_objects"), (f"{cond_str2}-two_objects", f"{cond_str1}-two_objects"))]
        box_pairs += [((f"{cond_str2}-two_objects", f"{cond_str2}-two_objects"), (f"{cond_str2}-two_objects", f"{cond_str1}-one_object"))]
    
    make_sns_barplot(df, x='Tested on', y='AUC', hue='Trained on', box_pairs=box_pairs, out_fn=f'{out_fn}_AUC.png', hline=0.5, ymin=0.45)
    

    ## COSINE SIMILARITY
    if all_cosines:
        cos_dict = {"Conditions": [], "subject": [], "cosine similarity": []}
        for cond1 in conds:
            for cond2 in conds:
                if cond1==cond2: continue # same hyperplane
                for i_sub in range(n_subs):
                    try:
                        cos_dict["cosine similarity"].append(all_cosines[i_sub][f"{cond1}--vs--{cond2}"])
                        cos_dict["Conditions"].append(shorten_filename(f"{cond1} vs {cond2}"))
                        cos_dict["subject"].append(i_sub+1)
                    except: # that's ok, we only have one-way comparison (they are symmetrical)
                        continue
        df = pd.DataFrame.from_dict(cos_dict)
        

        cos_conds = [x for x in df.Conditions.unique()]
        # box_pairs = [((c1, c1), (c1, c2)) for c1 in cos_conds for c2 i]
        box_pairs = [x for x in combinations(cos_conds, 2)]
        make_sns_barplot(df, x='Conditions', y='cosine similarity', box_pairs=box_pairs, out_fn=f'{out_fn}_cos.png')
    else:
        print(f"did not find any file for cosines for label {label}")


    ## AUC gen to all time points
    if all_AUC_allt and "windows" in all_AUC_allt[0].keys():
        n_subs = len(all_AUC_allt)
        print(f"found {len(all_AUC_allt)} files with all timepoints tested")
        print(all_AUC_allt[0].keys())
        if len(np.atleast_1d(all_AUC_allt[0][conds[0]][conds[0]])) == 1:
            print("Problem loading all timepoints AUC, single timepoint found ... moving to next condition but look it up!")
            continue
        # remove non-matched times
        print("\nRemoving subjects with a number of timepoints that do not match the first one")
        to_del = []
        for i1, train_cond in enumerate(conds):
            for i2, test_cond in enumerate(conds):
                if train_cond!=test_cond:
                    n_times = len(all_AUC_allt[0][train_cond][test_cond])
                    for i_sub in range(n_subs):
                        try:
                            if len(all_AUC_allt[i_sub][train_cond][test_cond]) != n_times:
                                to_del.append(i_sub)
                                print(f"Dropping subject {i_sub+1}")
                        except TypeError:
                            to_del.append(i_sub)
                            print(f"Dropping subject because it does not have multiple generalization timepoints {i_sub+1}")
        all_AUC_allt = [all_AUC_allt[i_sub] for i_sub in range(n_subs) if i_sub not in to_del]
        n_subs = len(all_AUC_allt)

        clr_ctr = 0
        fig, axes = plt.subplots(len(conds))
        for i1, train_cond in enumerate(conds):
            for i2, test_cond in enumerate(conds):
                clr = cmap10(clr_ctr)
                clr_ctr += 1
                if train_cond==test_cond: # same training and testing, plot cval score in a hline
                    win = all_AUC_allt[0]["windows"][train_cond]
                    ave = np.mean([all_AUC[i_sub][train_cond][test_cond] for i_sub in range(n_subs)], 0)
                    std = np.std([all_AUC[i_sub][train_cond][test_cond] for i_sub in range(n_subs)], 0)
                    axes[i1].axhline(y=ave, xmin=win[0], xmax=win[1], color=clr, linestyle='-', alpha=.3)
                    # axes[i1].fill_between(np.linspace(win[0], win[1], 2), ave-std, ave+std, alpha=0.2, zorder=-1, lw=0, color=clr)

                else:
                    n_times = len(all_AUC_allt[0][train_cond][test_cond])
                    tmin, tmax = tmin_tmax_dict[test_cond.split("-")[1]]
                    times = np.linspace(tmin, tmax, n_times)
                    try:
                        # print([all_AUC_allt[i_sub][train_cond][test_cond].shape for i_sub in range(n_subs)])
                        ave = np.mean([all_AUC_allt[i_sub][train_cond][test_cond] for i_sub in range(n_subs)], 0)
                    except:
                        set_trace()
                    std = np.std([all_AUC_allt[i_sub][train_cond][test_cond] for i_sub in range(n_subs)], 0)

                    plot = axes[i1].plot(times, ave, alpha=0.8, lw=1, label=test_cond, c=clr)
                    axes[i1].fill_between(times, ave-std, ave+std, alpha=0.2, zorder=-1, lw=0, color=plot[0].get_color())

            axes[i1].axhline(y=0.5, color='k', linestyle='-', alpha=.3, lw=.1)
            axes[i1].set_title(f"Trained on {train_cond}")
            axes[i1].legend(title='Tested on', fontsize=8) #, fancybox=True)
            axes[i1].set_ylabel("AUC")
        axes[i1].set_xlabel("Times (s)")
        plt.tight_layout()
        plt.savefig(f"{out_fn}_AUC_allt.png")

    print(f"Finished {label}\n")
    plt.close('all')

# stim vs delay plot
df_all_labels = pd.concat(df_all_labels)
df_all_labels['Trained during'] = df_all_labels.apply(lambda x: ("Stimulus" if 'full' in x.label else "Delay" if 'delay' in x.label else "Other"), axis=1)
df_loc = deepcopy(df_all_labels)
df_loc = df_loc.query("`Trained during` != 'Other'")
df_loc = df_loc.query("`Trained on` in ['Shape1-one_object', 'Colour1-one_object', 'Shape1-two_objects', 'Colour1-two_objects', 'Shape2-two_objects', 'Colour2-two_objects']") 
#['Colour1-one object', 'Shape1-one object', 'Shape1-two objects', 'Shape2-two objects', 'Colour1-two objects', 'Colour2-two obiects']")
df_loc = df_loc.query("`Trained on` == `Tested on`")

box_pairs = []
# box_pairs += [((f"{cond_str1}-two_objects", f"{cond_str1}-two_objects"), (f"{cond_str1}-two_objects", f"{cond_str1}-one_object"))]
# box_pairs += [((f"{cond_str1}-two_objects", f"{cond_str1}-two_objects"), (f"{cond_str1}-two_objects", f"{cond_str2}-two_objects"))]
# box_pairs += [((f"{cond_str1}-one_object", f"{cond_str1}-one_object"), (f"{cond_str1}-one_object", f"{cond_str1}-two_objects"))]
# box_pairs += [((f"{cond_str1}-one_object", f"{cond_str1}-one_object"), (f"{cond_str1}-one_object", f"{cond_str2}-two_objects"))]
# box_pairs += [((f"{cond_str2}-two_objects", f"{cond_str2}-two_objects"), (f"{cond_str2}-two_objects", f"{cond_str1}-two_objects"))]
# box_pairs += [((f"{cond_str2}-two_objects", f"{cond_str2}-two_objects"), (f"{cond_str2}-two_objects", f"{cond_str1}-one_object"))]

order = ['Shape1-one_object', 'Colour1-one_object', 'Shape1-two_objects', 'Colour1-two_objects', 'Shape2-two_objects', 'Colour2-two_objects']
make_sns_barplot(df_loc, x='Trained during', y='AUC', hue='Trained on', box_pairs=box_pairs, out_fn=f'{out_fn}_AUC_trainon.png', hline=0.5, ymin=0.45, order=order)

make_sns_barplot(df_loc, x='AUC', y='Trained during', hue='Trained on', box_pairs=box_pairs, out_fn=f'{out_fn}_AUC_trainon_horizontal.png', vline=0.5, xmin=0.45, order=order)

make_sns_barplot(df_loc, x='AUC', y='Trained on', hue='Trained during', box_pairs=box_pairs, out_fn=f'{out_fn}_AUC_traindur_horizontal.png', vline=0.5, xmin=0.45, order=order)

