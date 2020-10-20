import os.path as op
import os
from glob import glob
from ipdb import set_trace
import mne
import argparse
import pandas as pd
import numpy as np
import pickle

from utils.params import *
from utils.reject import *

parser = argparse.ArgumentParser(description='MEG basic preprocessing')
parser.add_argument('-r', '--root-path', default='/neurospin/unicog/protocols/MEG/Seq2Scene/', help='Path to parent project folder')
parser.add_argument('-o', '--out-dir', default='/Data/Epochs', help='output directory')
parser.add_argument('-p', '--subject', default='theo',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--l_freq', default=1., type=float, help='Low-pass filter frequency (0 for only high pass')
parser.add_argument('--h_freq', default=60, type=float, help='High_pass filter frequency(0 for only low pass)')
parser.add_argument('--notch', default=50, type=int, help='frequency of the notch filter (0 = no notch filtering)')
parser.add_argument('--resample_sfreq', default=0, type=int, help='resampling frequency')
parser.add_argument('--plot', default=False, action='store_true', help='Whether to plot some channels and power spectrum')
parser.add_argument('--show', default=False, action='store_true', help='Whether to show some channels and power spectrum ("need to be locally or ssh -X"')
parser.add_argument('--ch-var-reject', default=0, type=int, help='whether to reject channels based on temporal variance')
parser.add_argument('--epo-var-reject', default=0, type=int, help='whether to reject epochs based on temporal variance')
parser.add_argument('--ref-run', default=8, type=int, help='reference run for head position for maxwell filter')

parser.add_argument('--pass-fosca', default=False, action='store_true', help='exceptions for the first pilot"')

### TODO: OPTIONALLY PASS A LIST OF BAD MEG SENSORS AS ARGUMENT? OR A PATH TO A TXT OR CSV FILE CONTAINING THE BAD SENSORS

print(mne.__version__)
args = parser.parse_args()
print(args)

import matplotlib
if args.show:
    matplotlib.use('Qt5Agg') # output to screen (locally or through ssh -X)
else:
    matplotlib.use('Agg') # no output to screen.
import matplotlib.pyplot as plt


if args.subject == "fosca":
    TRIG_DICT = {"localizer_block_start": 50,
                 "one_object_block_start": 150,
                 "two_objects_block_start": 250, 
                 "localizer_trial_start": 55,
                 "one_object_trial_start": 155,
                 "two_objects_trial_start": 255,
                 "new_word": 10,
                 "image": 20}

elif args.subject == "theo":
    TRIG_DICT = {"imglocalizer_block_start": 70,
                 "localizer_block_start": 50,
                 "one_object_block_start": 100,
                 "two_objects_block_start": 200, 
                 "imgloc_trial_start": 75,
                 "localizer_trial_start": 55,
                 "one_object_trial_start": 105,
                 "two_objects_trial_start": 205,
                 "imglocalizer_pause_start": 80,
                 "localizer_pause_start": 60,
                 "one_object_pause_start": 110,
                 "two_objects_pause_start": 210,
                 "new_word": 10, "image": 20, 
                 "correct": 30, "wrong": 40}

in_dir = op.join(args.root_path + '/Data', 'orig', args.subject)
all_runs_fns = glob(in_dir + '/*run*.fif')
all_runs_fns = sorted(all_runs_fns)

all_md_fns = glob(in_dir + '/*run*.csv')
all_md_fns = sorted(all_md_fns)

all_bads_fns = glob(in_dir + '/*run*_bads.p')
all_bads_fns = sorted(all_bads_fns)
if len(all_bads_fns) == 0:
    print(f"Did not find any file containg bad channels at path {in_dir}, you should have detected the bads channels before doing the maxwell filtering using the detect_bad_channels.py script")

if args.subject=="theo": # skip the 5th blocks that have a problem (split in two)
    all_runs_fns = [run for run in all_runs_fns if "run05_1obj_first" not in run] # _1obj_first
    all_md_fns = [md for md in all_md_fns if "run05_1obj_first" not in md]

assert len(all_runs_fns) == len(all_md_fns)
print(all_runs_fns)
print(all_md_fns)
n_runs = len(all_runs_fns)

# Make output directory
out_dir = op.join(args.root_path + args.out_dir, args.subject)
out_dir_plots = op.join(out_dir, 'plots')
if not op.exists(out_dir_plots):
    os.makedirs(out_dir_plots)

# get head position of the reference run
ref_info = mne.io.read_info(all_runs_fns[args.ref_run], verbose='warning')
ref_run_head_pos = ref_info['dev_head_t']
# compute head origin from digitization points (same for each run)
_, head_origin, _ = mne.bem.fit_sphere_to_headshape(ref_info, dig_kinds=['hpi', 'cardinal', 'extra'], units='m')
# calibration files
ct_sparse_fn = f"{args.root_path}/Data/SSS/ct_sparse_nspn.fif"
sss_cal_fn = f"{args.root_path}/Data/SSS/sss_cal_nspn.dat"

raw_loc = []
raw_imgloc = []
raw_1obj = []
raw_2obj = []

events_loc = []
events_imgloc = []
events_1obj = []
events_2obj = []

md_loc = []
md_imgloc = []
md_1obj = []
md_2obj = []

# change to dict for cleaner use?
# LOC = {"raw": [], "events": [], "md": []}

for i_run, raw_fn_in in enumerate(all_runs_fns):
    print("doing file ", raw_fn_in)

    if i_run==0 and args.pass_fosca: # do not do the first block as we do not have the labels 
        continue

    # Load data
    raw = mne.io.Raw(raw_fn_in, preload=True, verbose='error', allow_maxshield=True)
    try:
        md = pd.read_csv(all_md_fns[i_run])
    except: # weird hack - the 2obj md is with a different encoding ... 
        md = pd.read_csv(all_md_fns[i_run], encoding='ISO-8859-1')
    try: # read bads from file, you should have detected the bads channels before doing the maxwell filtering using the detect_bad_channels.py script
        bads = pickle.load(open(all_bads_fns[i_run], "rb"))
    except: 
        print("did not find the bads file ...")
        set_trace()

    md["run_nb"] = str(i_run + 1)

    if "run05" in raw_fn_in: # hack to get the hpi registration
        qwe = mne.io.read_info("../Data/orig/theo/run05_1obj_first.fif", verbose='warning')
        raw.info['dev_head_t'] = qwe['dev_head_t']

    # MAXWELL FILTERING
    mne.channels.fix_mag_coil_types(raw.info)
    # noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(raw, origin=head_origin, coord_frame='head', calibration=sss_cal_fn, cross_talk=ct_sparse_fn, verbose="warning")
    # if args.plot: plot_deviant_maxwell(scores, f"{out_dir_plots}/run_{i_run+1}")
    raw.info['bads'] = list(bads) #+ noisy_chs + flat_chs
    print(f"bad channels automatically detected and interpolated based on Maxwell filtering: {raw.info['bads']}")
    raw = mne.preprocessing.maxwell_filter(raw, origin=head_origin, coord_frame='head', calibration=sss_cal_fn, 
                                    cross_talk=ct_sparse_fn, destination=ref_run_head_pos, verbose='warning')
    
    raw = raw.del_proj("all")
    
    if args.plot:
        # plot power spectral densitiy
        fig = raw.plot_psd(area_mode='range', fmin=0., fmax=200., average=True)
        plt.savefig(f'{out_dir_plots}/run_{i_run+1}_psd_before_anything.png')
        plt.close()


    # Band-pass the data channels
    if args.l_freq or args.h_freq:
        if args.l_freq == 0: args.l_freq = None
        if args.h_freq == 0: args.h_freq = None
        print("Filtering data between %s and %s (Hz)" %(args.l_freq, args.h_freq))
        raw.filter(
            args.l_freq, args.h_freq,
            l_trans_bandwidth='auto',
            h_trans_bandwidth='auto',
            filter_length='auto', phase='zero', fir_window='hamming',
            fir_design='firwin')


    if args.plot:
        # plot power spectral densitiy
        fig = raw.plot_psd(area_mode='range', fmin=0., fmax=200., average=True)
        plt.savefig(f'{out_dir_plots}/run_{i_run+1}_psd_after_bandpass_{args.l_freq}_{args.h_freq}hz.png')
        plt.close()


    if args.notch:
        print(f"applying notch filter at {args.notch} and 3 harmonics")
        notch_freqs = [args.notch, args.notch*2, args.notch*3, args.notch*4]
        raw = raw.notch_filter(notch_freqs)

    if args.plot:
        # plot power spectral densitiy
        fig = raw.plot_psd(area_mode='range', fmin=0., fmax=200., average=True)
        plt.savefig(f'{out_dir_plots}/run_{i_run+1}_psd_after_bandpass_{args.l_freq}_{args.h_freq}hz_and_notch_{args.notch}hz.png')
        plt.close()

    
    if args.resample_sfreq:
        print("Resampling data to %.1f Hz" % args.resample_sfreq)
        raw.resample(args.resample_sfreq, npad='auto')
        resample_str = f'_and_resampling_{args.resample_sfreq}hz'
    else:
        resample_str = ''

    if args.plot: 
        # plot power spectral densitiy
        fig = raw.plot_psd(area_mode='range', fmin=0., fmax=args.resample_sfreq/2, average=True)
        plt.savefig(f'{out_dir_plots}/run_{i_run+1}_psd_after_bandpass_{args.l_freq}_{args.h_freq}hz_and_notch_{args.notch}hz{resample_str}.png')
        plt.close()

    # print('final raw info')
    # print(raw.info)
    # print('\n\n')

    try:
        events = mne.find_events(raw, min_duration=0.1)
    except:
        set_trace()

    # "start" and "end" triggers
    events = events[np.where(events[:,2] != 16434)]
    events = events[np.where(events[:,2] != 16384)]
    print(f"found {len(events)} triggers")
    
    # fn based block type detection
    if 'imgloc' in op.basename(raw_fn_in):
        block_type = 'imgloc'
        raw_imgloc.append(raw)
        events = events[np.where(events==TRIG_DICT['imgloc_trial_start'])[0]]
        events_imgloc.append(events)
        md_imgloc.append(md)
    elif 'loc' in op.basename(raw_fn_in):
        block_type = 'localizer'
        raw_loc.append(raw)
        events = events[np.where(events==TRIG_DICT['localizer_trial_start'])[0]]
        events_loc.append(events)
        if i_run == 5 and args.pass_fosca: # add the missing information
            with open(f'{in_dir}/additional_info_for_block_6loc.txt') as f:
                f.read().splitlines()
                set_trace()
            md['Loc_word'] = words
        md_loc.append(md)
    elif '1obj' in op.basename(raw_fn_in):
        block_type = 'one_object'
        raw_1obj.append(raw)
        events = events[np.where(events==TRIG_DICT['one_object_trial_start'])[0]]

#         ## HACK FOR FOSCA's PILOT WHERE WE MISSED 3 TRIALS
#         if len(events) < len(md):
#             print("WARNING: incorrect number of trials or metadata entries ... maybe the recording started after \
# the beginning of the block, as in fosca's pilot ... setting a trace... just continue to remove the metadata entries from the beginning of the run")
#             if not args.pass_fosca:
#                 set_trace()
#             md = md.iloc[len(md)-len(events)::]

        ## HACK FOR THEO'S PILOT: THE 5TH BLOCK IS SPLIT IN TWO AND WE MISSED A FEW TRIALS ...
        if "run05" in raw_fn_in:
            md = md.iloc[len(md)-len(events)::] # just skip the first few trials

        events_1obj.append(events)
        md_1obj.append(md)
    elif '2obj' in op.basename(raw_fn_in):
        block_type = 'two_objects'
        raw_2obj.append(raw)
        events = events[np.where(events==TRIG_DICT['two_objects_trial_start'])[0]]

        ## HACK FOR FOSCA's PILOT WHERE WE MISSED 3 TRIALS
        if len(md) < len(events):
            if not args.pass_fosca:
                print("WARNING: missing metadata value compared to nubmer of events ... happenened in fosca's pilot (missing one value). Removing the last one (no particular reason, we are not really interested in the 2 objects block anyway as there are only 20 trials.")
                set_trace()
            events = events[1::]
        events_2obj.append(events)
        md_2obj.append(md)


    ## epochs for this block plot
    if args.plot:
        tmin, tmax = tmin_tmax_dict[block_type]
        epo = mne.Epochs(raw, events, event_id=TRIG_DICT[f'{block_type}_trial_start'], tmin=tmin, tmax=tmax, metadata=md, baseline=(None, 0))
        evo = epo.average()
        evo.plot(spatial_colors=True, show=False)
        plt.savefig(f'{out_dir_plots}/run_{i_run+1}_{block_type}_epochs.png')
plt.close()

    # raw.save(raw_fn_out, overwrite=True)

## Concatenate all runs, condition-wise
raw_loc, events_loc = mne.concatenate_raws(raw_loc, events_list=events_loc)
raw_imgloc, events_imgloc = mne.concatenate_raws(raw_imgloc, events_list=events_imgloc)
raw_1obj, events_1obj = mne.concatenate_raws(raw_1obj, events_list=events_1obj)
raw_2obj, events_2obj = mne.concatenate_raws(raw_2obj, events_list=events_2obj)

# # apply projections
# raw_loc = raw_loc.apply_proj()
# raw_imgloc = raw_imgloc.apply_proj()
# raw_1obj = raw_1obj.apply_proj()
# raw_2obj = raw_2obj.apply_proj()

if args.ch_var_reject:
    # detect and reject bad channels
    bads = get_deviant_ch(raw_loc, thresh=args.ch_var_reject)
    bads.update(get_deviant_ch(raw_imgloc, thresh=args.ch_var_reject))
    bads.update(get_deviant_ch(raw_1obj, thresh=args.ch_var_reject))
    bads.update(get_deviant_ch(raw_2obj, thresh=args.ch_var_reject))
    # set_trace()
    raw_loc.info['bads'] = bads
    raw_imgloc.info['bads'] = bads
    raw_1obj.info['bads'] = bads
    raw_2obj.info['bads'] = bads

md_loc = pd.concat(md_loc)
md_imgloc = pd.concat(md_imgloc)
md_1obj = pd.concat(md_1obj)
md_2obj = pd.concat(md_2obj)

if args.overwrite:
    tmin, tmax = tmin_tmax_dict['localizer']
    epochs_loc = mne.Epochs(raw_loc, events_loc, event_id=TRIG_DICT['localizer_trial_start'], tmin=tmin, tmax=tmax, metadata=md_loc, preload=True)
    # epochs_loc = epochs_loc.decimate(10)
    if args.ch_var_reject:
        # detect and reject bad epochs
        bad_epochs = get_deviant_epo(epochs_loc, thresh=args.ch_var_reject)
        epochs_loc = reject_bad_epochs(epochs_loc, bad_epochs)
    epochs_loc.save(f"{out_dir}/localizer-epo.fif", overwrite=True)
    
    # imgloc
    tmin, tmax = tmin_tmax_dict['imgloc']
    epochs_imgloc = mne.Epochs(raw_imgloc, events_imgloc, event_id=TRIG_DICT['imgloc_trial_start'], tmin=tmin, tmax=tmax, metadata=md_imgloc, preload=True)
    # epochs_imgloc = epochs_imgloc.decimate(10)
    if args.ch_var_reject:
        # detect and reject bad epochs
        bad_epochs = get_deviant_epo(epochs_imgloc, thresh=args.ch_var_reject)
        epochs_imgloc = reject_bad_epochs(epochs_imgloc, bad_epochs)
    epochs_imgloc.save(f"{out_dir}/imgloc-epo.fif", overwrite=True)
    
    # one_object 
    tmin, tmax = tmin_tmax_dict['one_object']
    epochs_1obj = mne.Epochs(raw_1obj, events_1obj, event_id=TRIG_DICT['one_object_trial_start'], tmin=tmin, tmax=tmax, metadata=md_1obj, preload=True)
    # epochs_1obj = epochs_1obj.decimate(10)
    if args.ch_var_reject:
        # detect and reject bad epochs
        bad_epochs = get_deviant_epo(epochs_1obj, thresh=args.ch_var_reject)
        epochs_1obj = reject_bad_epochs(epochs_1obj, bad_epochs)
    epochs_1obj.save(f"{out_dir}/one_object-epo.fif", overwrite=True)
    
    # two_objects
    tmin, tmax = tmin_tmax_dict['two_objects']
    epochs_2obj = mne.Epochs(raw_2obj, events_2obj, event_id=TRIG_DICT['two_objects_trial_start'], tmin=tmin, tmax=tmax, metadata=md_2obj, preload=True)
    # epochs_2obj = epochs_2obj.decimate(10)
    if args.ch_var_reject:
        # detect and reject bad epochs
        bad_epochs = get_deviant_epo(epochs_2obj, thresh=args.ch_var_reject)
        epochs_2obj = reject_bad_epochs(epochs_2obj, bad_epochs)
    epochs_2obj.save(f"{out_dir}/two_objects-epo.fif", overwrite=True)


evo_loc = epochs_loc.average()
evo_loc.plot(spatial_colors=True, show=True)
plt.savefig(f'{out_dir_plots}/epochs_loc.png')
plt.close()

evo_imgloc = epochs_imgloc.average()
evo_imgloc.plot(spatial_colors=True, show=True)
plt.savefig(f'{out_dir_plots}/epochs_imgloc.png')
plt.close()

evo_1obj = epochs_1obj.average()
evo_1obj.plot(spatial_colors=True, show=True)
plt.savefig(f'{out_dir_plots}/epochs_1obj.png')
plt.close()

evo_2obj = epochs_2obj.average()
evo_2obj.plot(spatial_colors=True, show=True)
plt.savefig(f'{out_dir_plots}/epochs_2obj.png')
plt.close()

plt.show()
plt.pause(0.1)


# block_type = 'localizer'
# tmin, tmax = -.2, 1.

# block_type = 'one_object'
# tmin, tmax = -.5, 2.

# block_type = 'two_objects'
# tmin, tmax = -.5, 3.

# trig = TRIG_DICT[f'{block_type}_trial_start']

