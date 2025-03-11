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
from scipy.stats import sem
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
parser.add_argument('-s', '--subject', default='01_js180232',help='subject name')
parser.add_argument('-o', '--out-dir', default='agg', help='output directory')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('-r', '--remake', action='store_true',  default=False, help='recompute average again, even if plotting only aggregates')
parser.add_argument('-v', '--verbose', action='store_true',  default=False, help='Print more stuff')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

start_time = time.time()

feat2feats = {"Shape": ['carre', 'cercle', 'triangle', 'ca', 'cl', 'tr'], "Colour": ['rouge', 'bleu', 'vert', 'vr', 'bl', 'rg']}

if args.slices:
    slices_loc = [0.17, 0.3, 0.43, 0.5, 0.72, 0.85]
    slices_one_obj = [0.2, 0.85, 1.1, 2.5, 2.75]
else:
    slices = []

decoding_dir = f"Decoding_ovr_v12"
single_ch_decoding_dir = f"Decoding_single_ch_v12"
# decoding_dir = f"Decoding_ovr_v{args.version}" if args.ovr else f"Decoding_v{args.version}"
if args.subject in ["all", "v1", "v2",  "goods"]: # for v1 and v2 we filter later
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/*/"
else:
    in_dir = f"{args.root_path}/Results/{decoding_dir}/{args.epochs_dir}/{args.subject}/"
out_dir = f"{args.root_path}/Results/Final_figures/{decoding_dir}/{args.epochs_dir}/{args.subject}/{args.out_dir}/"

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

data_fn = f"{op.dirname(op.dirname(out_dir))}/all_data.p"
diags_fn = f"{op.dirname(op.dirname(out_dir))}/all_diags.p"
patterns_fn = f"{op.dirname(op.dirname(out_dir))}/all_patterns.p"
print(f"loading each results and directly loading diag aggregates from {diags_fn}")
diag_auc_all_labels = pickle.load(open(diags_fn, "rb"))
pattern_all_labels = pickle.load(open(patterns_fn, "rb"))
# auc_all_labels = pickle.load(open(data_fn, "rb"))


## Fig 1 - One object


## all basic decoders diagonals on the same plot
tmin, tmax = tmin_tmax_dict["scenes"]
times = np.arange(tmin, tmax+1e-10, 1./args.sfreq)
word_onsets, image_onset = get_onsets("scenes", version=version)

# for label, pattern in pattern_all_labels.items():
#     fig = mne.viz.plot_topomap(pattern[mag_idx], mag_info, contours=0)
#     plt.savefig(f'{out_dir}/{label}_pattern_mag.png')
#     plt.close('all')
#     fig = mne.viz.plot_topomap(pattern[grad_idx], grad_info, contours=0)
#     plt.savefig(f'{out_dir}/{label}_pattern_grad.png')
#     plt.close('all')
# # from ipdb import set_trace; set_trace()   

print(diag_auc_all_labels.keys())

if args.ovr:
    # mismatch type
    labels = ['Matching_scenes_None_', 'PropMismatch_scenes_None_', 'BindMismatch_scenes_None_', 'RelMismatch_scenes_None_'] #, 'Button_scenes_None_', 'Perf_scenes_None_']
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_mismatches.png', tmin=4.5, word_onsets=word_onsets, image_onset=image_onset)

    # only basics
    labels = ['S1_scenes_None_', 'C1_scenes_None_', 'R_scenes_None_', 'S2_scenes_None_', 'C2_scenes_None_']
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_basic.png', word_onsets=word_onsets, image_onset=image_onset)
    labels = ['Flash_scenes_None_', 'Matching_scenes_None_', 'Button_scenes_None_', 'Perf_scenes_None_'] # 'SameObj_scenes_None_', 
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_basic2.png', word_onsets=word_onsets, image_onset=image_onset)

    # # Shape gen
    # labels = ['S1_scenes_None_', 'S2_scenes_None_', 'S1_scenes_scenes_', 'S2_scenes_scenes_']
    # joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_shape_gen.png', word_onsets=word_onsets, image_onset=image_onset)

    # extended
    labels = ['S1_scenes_None_', 'C1_scenes_None_', 'R_scenes_None_', 'S2_scenes_None_', 'C2_scenes_None_', 'Flash_scenes_None_', 'Matching_scenes_None_', 'Button_scenes_None_', 'Perf_scenes_None_']
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_extended.png', word_onsets=word_onsets, image_onset=image_onset)



    # object properties and gen
    tmin, tmax = tmin_tmax_dict["obj"]
    times = np.arange(tmin, tmax+1e-10, 1./args.sfreq)
    word_onsets, image_onset = get_onsets("obj", version=version)
    labels = ['S_obj_None_', 'C_obj_None_'] #, 'AllObj_obj_None_', 'CMismatch_obj_None_', 'SMismatch_obj_None_']
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, tmax=4, times=times, out_fn=f'{out_dir}/scenes_joyplot_obj_props.png', word_onsets=word_onsets, image_onset=image_onset)
    # labels = ['S_obj_None_', 'S_0_obj_scenes_', 'S_1_obj_scenes_', 'C_obj_None_', 'C_0_obj_scenes_', 'C_1_obj_scenes_']
    # joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, tmax=4, times=times, out_fn=f'{out_dir}/scenes_joyplot_obj_prop_and_gen.png', word_onsets=word_onsets, image_onset=image_onset)

    # object properties gen to scenes
    labels = ['S_0_obj_scenes_', 'S_1_obj_scenes_', 'C_0_obj_scenes_', 'C_1_obj_scenes_']
    # set_trace()
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_obj_gen.png', word_onsets=word_onsets, image_onset=image_onset)

else:
    labels = ['SameC_scenes_None_', 'SameS_scenes_None_', 'SameC_0_scenes_scenes_', 'SameS_0_scenes_scenes_', 'SameObj_scenes_None_']
    # labels = ['SameC_scenes_None_', 'SameS_scenes_None_', 'SameObj_scenes_None_']
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, times=times, out_fn=f'{out_dir}/scenes_joyplot_complexity.png', word_onsets=word_onsets, image_onset=image_onset)

    labels = ['PropMismatch_scenes_None_', 'BindMismatch_scenes_None_', 'RelMismatch_scenes_None_']
    joyplot_with_stats(data_dict=diag_auc_all_labels, labels=labels, tmin=4.5, times=times, out_fn=f'{out_dir}/scenes_joyplot_mismatches.png', word_onsets=word_onsets, image_onset=image_onset)





# joyplot_with_stats(data_dict=diag_auc_all_labels, labels=diag_auc_all_labels.keys(), times=times, out_fn=f'{out_dir}/scenes_joyplot_all.png', word_onsets=word_onsets, image_onset=image_onset) # ['S1','C1','R','S2','C2','All1stObj','All2ndObj']

# # all properties
# plot_all_props_multi(ave_dict=ave_auc_all_labels, std_dict=std_auc_all_labels, labels=ave_auc_all_labels.keys(), times=times, out_fn=f'{out_dir}/scenes_props_multi_all.png', word_onsets=word_onsets, image_onset=image_onset) # ['S1','C1','R','S2','C2','All1stObj','All2ndObj']
# plot_all_props(ave_dict=ave_auc_all_labels, std_dict=std_auc_all_labels, labels=ave_auc_all_labels.keys(), times=times, out_fn=f'{out_dir}/scenes_props_all.png', word_onsets=word_onsets, image_onset=image_onset) # ['S1','C1','R','S2','C2','All1stObj','All2ndObj']

# # default ones
# plot_all_props_multi(ave_dict=ave_auc_all_labels, std_dict=std_auc_all_labels, times=times, out_fn=f'{out_dir}/scenes_props_multi.png', word_onsets=word_onsets, image_onset=image_onset)
# plot_all_props(ave_dict=ave_auc_all_labels, std_dict=std_auc_all_labels, times=times, out_fn=f'{out_dir}/scenes_props.png', word_onsets=word_onsets, image_onset=image_onset)

# # extended ones
# plot_all_props_multi(ave_dict=ave_auc_all_labels, std_dict=std_auc_all_labels, labels=['S1','C1','R','S2','C2','All1stObj','All2ndObj'], times=times, out_fn=f'{out_dir}/scenes_props_multi_all.png', word_onsets=word_onsets, image_onset=image_onset)
# plot_all_props(ave_dict=ave_auc_all_labels, std_dict=std_auc_all_labels, times=times, out_fn=f'{out_dir}/scenes_props_all.png', word_onsets=word_onsets, image_onset=image_onset)


# x = PrettyTable()
# x.field_names = ["Factor", "max AUC"]
# order = np.argsort(max_auc_all_facconds)
# with open(f'{out_fn}_max_AUC.csv', 'w') as f:
#     for i in order:
#         x.add_row([all_faccond_names[i], f"{max_auc_all_facconds[i]:.3f}"])
#         f.write(f"{all_faccond_names[i]}, {max_auc_all_facconds[i]:.3f}")
# # print(x)
# # set_trace()

print(f"ALL FINISHED, elpased time: {(time.time()-start_time)/60:.2f}min")
