import os.path as op
import os
from glob import glob
from ipdb import set_trace
import mne
import argparse
import pandas as pd
import numpy as np
import pickle
import importlib

from utils.params import *
from utils.reject import *


parser = argparse.ArgumentParser(description='Load and convert to MNE Raw, then preprocess, make Epochs and save')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='02_jm100042',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--plot', default=False, action='store_true', help='Whether to plot some channels and power spectrum')
parser.add_argument('--show', default=False, action='store_true', help='Whether to show some channels and power spectrum ("need to be locally or ssh -X"')
# parser.add_argument('-o', '--out-dir', default='/Data/Epochs', help='output directory')
# parser.add_argument('--ref-run', default=8, type=int, help='reference run for head position for maxwell filter')
# parser.add_argument('--pass-fosca', default=False, action='store_true', help='exceptions for the first pilot"')
args = parser.parse_args()

### TODO: OPTIONALLY PASS A LIST OF BAD MEG SENSORS AS ARGUMENT? OR A PATH TO A TXT OR CSV FILE CONTAINING THE BAD SENSORS

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)

np.random.seed(42)

import matplotlib
if args.show:
    matplotlib.use('Qt5Agg') # output to screen (locally or through ssh -X)
else:
    matplotlib.use('Agg') # no output to screen.
import matplotlib.pyplot as plt


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
out_dir = op.join(f"{args.root_path}/Data/{args.epochs_dir}/{args.subject}")
out_dir_plots = op.join(out_dir, 'plots')
if not op.exists(out_dir_plots):
    os.makedirs(out_dir_plots)
elif args.overwrite:
    print("overwriting previous output directory")
else:
    print("output directory already exists and overwriting is set to False ... exiting")
    exit()

# get head position of the reference run
ref_info = mne.io.read_info(all_runs_fns[args.ref_run], verbose='warning')
ref_run_head_pos = ref_info['dev_head_t']
# compute head origin from digitization points (same for each run)
_, head_origin, _ = mne.bem.fit_sphere_to_headshape(ref_info, dig_kinds=['hpi', 'cardinal', 'extra'], units='m')
# calibration files
ct_sparse_fn = f"{args.root_path}/Data/SSS/ct_sparse_nspn.fif"
sss_cal_fn = f"{args.root_path}/Data/SSS/sss_cal_nspn.dat"

raw_loc = []
raw_1obj = []
raw_2obj = []

events_loc = []
events_1obj = []
events_2obj = []

md_loc = []
md_1obj = []
md_2obj = []

# change to dict for cleaner use?
# LOC = {"raw": [], "events": [], "md": []}

def fn2blocktype(fn):
    if 'loc' in op.basename(raw_fn_in):
        return 'localizer'
    elif '1obj' in op.basename(raw_fn_in):
        return 'one_object'
    elif '2obj' in op.basename(raw_fn_in):
        return 'two_objects'


for i_run, raw_fn_in in enumerate(all_runs_fns):
    # if not "1obj" in raw_fn_in:
    #     continue
    if "run10" in raw_fn_in: continue

    print("doing file ", raw_fn_in)
    run_nb = op.basename(raw_fn_in).split("run")[1].split("_")[0]

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

    md["run_nb"] = str(run_nb)

    # if "run05" in raw_fn_in: # hack to get the hpi registration
    #     qwe = mne.io.read_info("../Data/orig/theo/run05_1obj_first.fif", verbose='warning')
    #     raw.info['dev_head_t'] = qwe['dev_head_t']


    # MAXWELL FILTERING
    mne.channels.fix_mag_coil_types(raw.info)
    # noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(raw, origin=head_origin, coord_frame='head', calibration=sss_cal_fn, cross_talk=ct_sparse_fn, verbose="warning")
    # if args.plot: plot_deviant_maxwell(scores, f"{out_dir_plots}/run_{run_nb}")
    raw.info['bads'] = list(bads) #+ noisy_chs + flat_chs
    print(f"bad channels automatically detected and interpolated based on Maxwell filtering: {raw.info['bads']}")
    raw = mne.preprocessing.maxwell_filter(raw, origin=head_origin, coord_frame='head', calibration=sss_cal_fn, 
                                    cross_talk=ct_sparse_fn, destination=ref_run_head_pos, verbose='warning')
    
    raw = raw.del_proj("all")
    
    if args.plot:
        # plot power spectral densitiy
        fig = raw.plot_psd(area_mode='range', fmin=0., fmax=200., average=True)
        plt.savefig(f'{out_dir_plots}/run_{run_nb}_psd_before_anything.png')
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
        plt.savefig(f'{out_dir_plots}/run_{run_nb}_psd_after_bandpass_{args.l_freq}_{args.h_freq}hz.png')
        plt.close()


    if args.notch:
        print(f"applying notch filter at {args.notch} and 3 harmonics")
        notch_freqs = [args.notch, args.notch*2, args.notch*3, args.notch*4]
        raw = raw.notch_filter(notch_freqs)

    if args.plot:
        # plot power spectral densitiy
        fig = raw.plot_psd(area_mode='range', fmin=0., fmax=200., average=True)
        plt.savefig(f'{out_dir_plots}/run_{run_nb}_psd_after_bandpass_{args.l_freq}_{args.h_freq}hz_and_notch_{args.notch}hz.png')
        plt.close()

    block_type = fn2blocktype(raw_fn_in)

    ## get events
    min_duration = 0.002
    if block_type == "two_objects": # for some reason the two objects block do not zork with Chrsitos panacea
        if args.subject == "05_mb140004" and "run6_2obj" in raw_fn_in:
            min_duration = 0.002
        events = mne.find_events(raw, stim_channel='STI101', verbose=True, min_duration=min_duration)
    else:
        # load events with Christos' panacea code to correctly get all triggers
        # if  args.subject == "04_ag170045" and "run3_1obj" in raw_fn_in or \
        #     args.subject == "05_mb140004" and "run10_loc" in raw_fn_in or \
        #     args.subject == "05_mb140004" and "run9_1_obj" in raw_fn_in or \
        #     args.subject == "06_ll180197" and "run10_loc" in raw_fn_in:  # exception, must change min_duration
        #         min_duration = 0.002
        events = mne.find_events(raw, stim_channel='STI101', verbose=True, min_duration=min_duration, consecutive='increasing', uint_cast=True, mask=128 + 256 + 512 + 1024 + 2048 + 4096 + 8192 + 16384 + 32768, mask_type='not_and',)

    if args.subject == "05_mb140004" and "run8" in raw_fn_in:
        # missing the 12th trial_start trigger: fix works
        trial_start_to_word_delay = events[events[:,2]==10][0][0] - events[events[:,2]==105][0][0]
        twelth_word_onset = events[events[:,2]==10][11, 0]
        infered_trial_onset = twelth_word_onset - trial_start_to_word_delay
        idx_to_insert = np.where(events == twelth_word_onset)[0][0]
        event_to_insert = np.array((infered_trial_onset, 0, 105))
        events = np.vstack((events[0:idx_to_insert], event_to_insert, events[idx_to_insert::]))

    if args.plot:
        fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)
        plt.savefig(f'{out_dir_plots}/run_{run_nb}_{block_type}_events.png', dpi=500)
        plt.close()  

    
    # # remove "start" and "end" triggers
    # events = events[np.where(events[:,2] != 16434)]
    # events = events[np.where(events[:,2] != 16384)]
    # # remove butotn presses
    # events = events[np.where(events[:,2] != 8232)]
    # events = events[np.where(events[:,2] != 8224)]
    # print(f"found {len(events)} triggers")

    if block_type == 'localizer':
        raw_loc.append(raw)
        events = events[np.where(events==TRIG_DICT['localizer_trial_start'])[0]]
        events_loc.append(events)
        md_loc.append(md)
    elif block_type == 'one_object':
        raw_1obj.append(raw)
        events = events[np.where(events==TRIG_DICT['one_object_trial_start'])[0]]

        ## HACK FOR WHEN WE MISSED A FEW TRIALS AT THE BEGINING OF THE BLOCK...
        if args.subject == "01_js180232" and "run3" in raw_fn_in:
            md = md.iloc[len(md)-len(events)::] # just skip the first few trials
        elif args.subject == "09_jl190711" and "run2" in raw_fn_in:
            md = md.iloc[1::] # just skip the first trial

        events_1obj.append(events)
        md_1obj.append(md)

    elif block_type == 'two_objects':
        raw_2obj.append(raw)
        events = events[np.where(events==TRIG_DICT['two_objects_trial_start'])[0]]

        ## HACKS FOR THE MISSING TRIALS
        if args.subject == "01_js180232" and "run5" in raw_fn_in:
            md = md.iloc[len(md)-len(events)::] # just skip the first few trials
        events_2obj.append(events)
        md_2obj.append(md)

    else:
        raise RuntimeError("Unknown block type", "Unknown block type")
    

    print(f"Found {len(events)} events")

    ## epochs for this block plot
    if args.plot:
        tmin, tmax = tmin_tmax_dict[block_type]
        try:
            epo = mne.Epochs(raw, events, event_id=TRIG_DICT[f'{block_type}_trial_start'], tmin=tmin, tmax=tmax, metadata=md, baseline=(None, 0), preload=True)
        except:
            set_trace()
        evo = epo.average()
        evo.plot(spatial_colors=True, show=False)
        plt.savefig(f'{out_dir_plots}/run_{run_nb}_{block_type}_evoked.png')
        
        # plot epochs for stim channel only to check the triggers
        info = mne.create_info(["STIM"], sfreq=epo.info['sfreq'], ch_types='misc')
        sti_data = epo.pick("STI101").get_data()
        # sti_data[np.where(sti_data==8232)[0]] = 0 # remove button presses
        # sti_data[np.where(sti_data==8224)[0]] = 0 # remove button presses
        epo_stim = mne.EpochsArray(sti_data, info, tmin=epo.tmin)
        epo_stim.plot_image(picks="STIM", show=False, scalings=dict(misc=1), units=dict(misc='UA'), vmin=0, vmax=TRIG_DICT[f'{block_type}_trial_start']+20)
        plt.savefig(f'{out_dir_plots}/run_{run_nb}_{block_type}_stim_channel.png')
        plt.close("all")


## Concatenate all runs, condition-wise
if raw_loc: raw_loc, events_loc = mne.concatenate_raws(raw_loc, events_list=events_loc)
if raw_1obj: raw_1obj, events_1obj = mne.concatenate_raws(raw_1obj, events_list=events_1obj)
if raw_2obj: raw_2obj, events_2obj = mne.concatenate_raws(raw_2obj, events_list=events_2obj)

# # apply projections
# raw_loc = raw_loc.apply_proj()
# raw_1obj = raw_1obj.apply_proj()
# raw_2obj = raw_2obj.apply_proj()

if args.ch_var_reject:
    # detect and reject bad channels
    if raw_loc: bads = get_deviant_ch(raw_loc, thresh=args.ch_var_reject)
    if raw_1obj: bads.update(get_deviant_ch(raw_1obj, thresh=args.ch_var_reject))
    if raw_2obj: bads.update(get_deviant_ch(raw_2obj, thresh=args.ch_var_reject))
    raw_loc.info['bads'] = bads
    raw_1obj.info['bads'] = bads
    raw_2obj.info['bads'] = bads

if md_loc: md_loc = pd.concat(md_loc)
if md_1obj: md_1obj = pd.concat(md_1obj)
if md_2obj: md_2obj = pd.concat(md_2obj)


# localizer
if raw_loc:
    tmin, tmax = tmin_tmax_dict['localizer']
    try:
        epochs_loc = mne.Epochs(raw_loc, events_loc, event_id=TRIG_DICT['localizer_trial_start'], tmin=tmin-1., tmax=tmax+1., metadata=md_loc, preload=True)
    except:
        set_trace()
    if args.epo_var_reject:
        # detect and reject bad epochs
        bad_epochs = get_deviant_epo(epochs_loc, thresh=args.epo_var_reject)
        epochs_loc = reject_bad_epochs(epochs_loc, bad_epochs)

    # resample to the final sfreq
    if args.sfreq:
        print("Resampling data to %.1f Hz" % args.sfreq)
        raw.resample(args.sfreq, npad='auto')
        resample_str = f'_and_resampling_{args.sfreq}hz'

    epochs_loc.save(f"{out_dir}/localizer-epo.fif", overwrite=True)


# one_object 
if raw_1obj:
    tmin, tmax = tmin_tmax_dict['one_object']
    try:
        epochs_1obj = mne.Epochs(raw_1obj, events_1obj, event_id=TRIG_DICT['one_object_trial_start'], tmin=tmin-1., tmax=tmax+1., metadata=md_1obj, preload=True)
    except:
        set_trace()
    if args.epo_var_reject:
        # detect and reject bad epochs
        bad_epochs = get_deviant_epo(epochs_1obj, thresh=args.epo_var_reject)
        epochs_1obj = reject_bad_epochs(epochs_1obj, bad_epochs)

    # resample to the final sfreq
    if args.sfreq:
        print("Resampling data to %.1f Hz" % args.sfreq)
        raw.resample(args.sfreq, npad='auto')
        resample_str = f'_and_resampling_{args.sfreq}hz'

    epochs_1obj.save(f"{out_dir}/one_object-epo.fif", overwrite=True)


# two_objects
if raw_2obj:
    tmin, tmax = tmin_tmax_dict['two_objects']
    try:
        epochs_2obj = mne.Epochs(raw_2obj, events_2obj, event_id=TRIG_DICT['two_objects_trial_start'], tmin=tmin-1., tmax=tmax+1., metadata=md_2obj, preload=True)
    except:
        set_trace()
    if args.epo_var_reject:
        # detect and reject bad epochs
        bad_epochs = get_deviant_epo(epochs_2obj, thresh=args.epo_var_reject)
        epochs_2obj = reject_bad_epochs(epochs_2obj, bad_epochs)

    # resample to the final sfreq
    if args.sfreq:
        print("Resampling data to %.1f Hz" % args.sfreq)
        raw.resample(args.sfreq, npad='auto')
        resample_str = f'_and_resampling_{args.sfreq}hz'

    epochs_2obj.save(f"{out_dir}/two_objects-epo.fif", overwrite=True)


if args.plot:

    if raw_loc:
        epochs_loc.apply_baseline((None, 0))    
        evo_loc = epochs_loc.average()
        evo_loc.plot(spatial_colors=True, show=True)
        plt.savefig(f'{out_dir_plots}/epochs_loc.png')
        # plt.close()

    if raw_1obj:
        epochs_1obj.apply_baseline((None, 0))
        evo_1obj = epochs_1obj.average()
        evo_1obj.plot(spatial_colors=True, show=True)
        plt.savefig(f'{out_dir_plots}/epochs_1obj.png')
        # plt.close()

    if raw_2obj:
        epochs_2obj.apply_baseline((None, 0))
        evo_2obj = epochs_2obj.average()
        evo_2obj.plot(spatial_colors=True, show=True)
        plt.savefig(f'{out_dir_plots}/epochs_2obj.png')
        # plt.close()

plt.show()
plt.pause(0.1)


# block_type = 'localizer'
# tmin, tmax = -.2, 1.

# block_type = 'one_object'
# tmin, tmax = -.5, 2.

# block_type = 'two_objects'
# tmin, tmax = -.5, 3.

# trig = TRIG_DICT[f'{block_type}_trial_start']

