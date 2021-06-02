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
from mne.stats import permutation_cluster_1samp_test
from prettytable import PrettyTable
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from itertools import permutations, combinations
import pandas as pd

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

all_labels = np.unique([op.basename(fn).split('-')[0] for fn in all_fns])
max_auc_all_facconds = []
all_faccond_names = []

for label in all_labels:
    all_AUC = []
    all_accuracy = []
    all_cosines = []
    all_subs = []
    for fn in all_fns:
        if op.basename(fn)[0:len(label)+1] != f"{label}-": continue 
        print('loading file ', fn)
        all_AUC.append(pickle.load(open(fn, "rb")))
        all_accuracy.append(pickle.load(open(f"{fn[0:-5]}accuracy.p", "rb")))
        all_cosines.append(pickle.load(open(f"{fn[0:-5]}cosine.p", "rb")))
        all_subs.append(op.basename(op.dirname(fn))[0:2])

    print(f"\nDoing {label}")
    out_fn = f"{out_dir}/{label}_{len(all_AUC)}ave"

    all_subs_labels = np.unique(all_subs)
    all_subs = dummy_class_enc.fit_transform(all_subs)
    n_subs = len(all_subs)


    conds = [x for x in all_AUC[0].keys()]
    n_conds = len(conds)

    auc_dict = {"Trained on": [], "Tested on": [], "AUC": [], "subject": []}
    for train_cond in conds:
        for test_cond in conds:
            for i_sub in range(n_subs):
                auc_dict["Trained on"].append(train_cond)
                auc_dict["Tested on"].append(test_cond)
                auc_dict["AUC"].append(all_AUC[i_sub][train_cond][test_cond])
                auc_dict["subject"].append(i_sub+1)

    df = pd.DataFrame.from_dict(auc_dict)

    if label == "AllColors":
        cond_str1 = "Colour1"
        cond_str2 = "Colour2"
    elif label == "AllSs":
        cond_str1 = "Shape1"
        cond_str2 = "Shape2"
    else:
        cond_str1 = "Shape1+Colour1"
        cond_str2 = "Shape2+Colour2"
    
    box_pairs = []
    box_pairs += [((f"{cond_str1}-two_objects", f"{cond_str1}-two_objects"), (f"{cond_str1}-two_objects", f"{cond_str1}-one_object"))]
    box_pairs += [((f"{cond_str1}-two_objects", f"{cond_str1}-two_objects"), (f"{cond_str1}-two_objects", f"{cond_str2}-two_objects"))]
    box_pairs += [((f"{cond_str1}-one_object", f"{cond_str1}-one_object"), (f"{cond_str1}-one_object", f"{cond_str1}-two_objects"))]
    box_pairs += [((f"{cond_str1}-one_object", f"{cond_str1}-one_object"), (f"{cond_str1}-one_object", f"{cond_str2}-two_objects"))]
    box_pairs += [((f"{cond_str2}-two_objects", f"{cond_str2}-two_objects"), (f"{cond_str2}-two_objects", f"{cond_str1}-two_objects"))]
    box_pairs += [((f"{cond_str2}-two_objects", f"{cond_str2}-two_objects"), (f"{cond_str2}-two_objects", f"{cond_str1}-one_object"))]

    make_sns_barplot(df, x='Tested on', y='AUC', hue='Trained on', box_pairs=box_pairs, out_fn=f'{out_fn}_AUC.png', hline=0.5, ymin=0.45)
    


    ## COSINE SIMILARITY
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

    print(f"Finished {label}\n")
    plt.close('all')