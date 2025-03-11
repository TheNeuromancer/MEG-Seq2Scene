import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import mne
import numpy as np
from ipdb import set_trace
import argparse
import pickle
import time
import os.path as op
import importlib

from utils.decod import *

parser = argparse.ArgumentParser(description='SEEG high gammas extraction and plotting of evoked power')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-r', '--root-path', default='/neurospin/unicog/protocols/MEG/Seq2Scene/', help='root path')
parser.add_argument('-i', '--in-dir', default='Data/Epochs/', help='input directory')
parser.add_argument('-o', '--out-dir', default='/Results/Epochs', help='output directory')
parser.add_argument('-s', '--subject', default='all',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('-v', '--version', default='2', help='version of the script')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--cat', default=0, type=int, help='How many timesteps to concatenate (default to one)')
parser.add_argument('--mean', default=False, action='store_true', help='Whether to actually average over instead of concatenating consecutive timesteps')
parser.add_argument('--smooth', default=False, action='store_true', help='Whether to apply gaussian kernel smoothing')
parser.add_argument('--baseline', action='store_true', default=False, help='Whether to apply baseline correction')
parser.add_argument('--tmin', default=-.2, type=float, help='Start of the epochs')
parser.add_argument('--tmax', default=6., type=float, help='End of the epochs')
parser.add_argument('--sfreq', default=100, type=int, help='Final sampling frequency to use (applying resampling)')
parser.add_argument('--clip', default=False, action='store_true', help='Whether to clip to the 99th percentile for each channel')
parser.add_argument('--subtract-evoked', action='store_true', default=False, help='Whether to subtract the evoked signal from the epochs')
parser.add_argument('--localizer', action='store_true', default=False, help='Whether to use only electrode that were significant in the localizer')
parser.add_argument('--auc_thresh', default=0.05, type=float, help='pvalue threshold under which a channel is kept for the localizer')
parser.add_argument('--freq-band', default='', help='name of frequency band to use for filtering (theta, alpha, beta, gamma)')
parser.add_argument('--dummy', action='store_true', default=False, help='Accelerates everything so that we can test that the pipeline is working. Will not yield any interesting result!!')
parser.add_argument('--response_lock', action='store_true',  default=None, help='Whether to Use response locked epochs or classical stim-locked')
parser.add_argument('--label', default='Evoked', help='help to identify the result latter')
parser.add_argument('--save_evos', action='store_true',  default=None, help='Whether to save raw data for latter')

# we hack these from utils/decod.py to load data
parser.add_argument('--train-cond', default='localizer', help='localizer, one_object or two_objects') # 'localizer', 'one_object', 
parser.add_argument('--test-cond', default=['two_objects'], help='localizer, one_object or two_objects')
parser.add_argument('--train-query-1', default="Colour1!='pwetpwet'", help='Metadata query for training classes(e.g., word_position==1),\n metadata is word-based (events are word onsets) and contains the following keys:\n\nsentence_string\n\nword_position (integer representing the word position from 1 to 8)\n\nword_string (str)\n\n pos (str: noun, verb, det, det2, article, prepo)\n')
parser.add_argument('--train-query-2', default="Perf==1", help='Metadata query for training classes(e.g., word_position==1),\n metadata is word-based (events are word onsets) and contains the following keys:\n\nsentence_string\n\nword_position (integer representing the word position from 1 to 8)\n\nword_string (str)\n\n pos (str: noun, verb, det, det2, article, prepo)\n')

# useless args here
parser.add_argument('--reduc-dim', default=0, type=float, help='number of PCA components, or percentage of variance to keep. If 0, do not apply PCA')
parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle sentence labels before training')
parser.add_argument('--n_folds', default=5, type=int, help='Number of cross-validation folds')
parser.add_argument('--C', default=0.1, type=float, help='Regularization parameter')
parser.add_argument('--timegen', action='store_true', default=False, help='Whether to test probe trained at one time point also on all other timepoints')
parser.add_argument('--split-queries', action='append', default=[], help='Metadata query for splitting the test data')
parser.add_argument('--mask_baseline', action='store_true', default=False, help='Whether to apply baseline correction from mask to first word onset')

print(mne.__version__)
args = parser.parse_args()
print(args)

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
for arg in vars(args): # update config with arguments from the argparse 
    if getattr(args, arg) is not None: # !! this is important, we can have only "None" as the default argument, else it will overwrite the config everytime !!
        setattr(config, arg, getattr(args, arg))
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))

np.random.seed(args.seed)
start_time = time.time()

### GET EPOCHS FILENAMES ###
if args.subject == "all":
    all_subs_all_fns = {}
    for  sub in args.all_subjects:
        args.subject = sub # replace for loading
        _, all_fns, _, _ = get_paths(args, verbose=False)
        all_subs_all_fns[sub] = all_fns
    args.subject = "all" # write back 
else:
    _, all_fns, _, _ = get_paths(args) # discard the first fn and use the test_fns to get all possible fns
    print(all_fns)
# out_fn.replace('Decoding', 'Plotting_HG')

out_dir = args.root_path + args.out_dir+'_'+args.version +'/' +  args.subject + '/'
if op.exists(out_dir): # warn and stop if args.overwrite is set to False
    print('\noutput folder already exists...')
    if args.overwrite:
        print('overwrite is set to True ... overwriting\n')
    else:
        print('overwrite is set to False ... exiting\n')
        exit()
else:
    print('Constructing output dirtectory: ', out_dir)
    os.makedirs(out_dir)

# epo_report = mne.report.Report()
evo_report = mne.report.Report()

def add_word_onsets(fig, word_onsets, image_onset):
    for w_onset in word_onsets:
        fig.axes[0].axvline(x=w_onset, linestyle='--', color='k')
        fig.axes[1].axvline(x=w_onset, linestyle='--', color='k')
    for img_onset in image_onset:
        fig.axes[0].axvline(x=img_onset, linestyle='-', color='k')
        fig.axes[1].axvline(x=img_onset, linestyle='-', color='k')


for fn, cond in zip(all_fns, args.test_cond):
    print("Doing: ", fn, cond)

    ### LOAD EPOCHS ###
    if args.subject == "all":
        all_subs_epochs = []
        all_subs_evos = []
        all_subs_evos_complexities = {'0': [], '1': [], '2': []}
        for sub in args.all_subjects:
            new_fn = f"{op.dirname(op.dirname(fn))}/{sub}/{op.basename(fn)}" # fn for this subject
            epochs, test_split_query_indices = load_data(args, new_fn, args.train_query_1, args.train_query_2)            
            epochs = epochs.pick_types(meg=True)
            all_subs_epochs.append(epochs)
            evo = deepcopy(epochs).average()
            all_subs_evos.append(evo)
            if cond == "two_objects": # store complexity evokeds
                for complexity in ['0','1','2']:
                    all_subs_evos_complexities[complexity].append(deepcopy(epochs)[f"Complexity=={complexity}"].average())

        # evo = mne.grand_average(all_subs_evos) # grand evo for all trials
        # set_trace()
        dat_med = np.median([e.data for e in all_subs_evos], 0)
        evo = mne.EvokedArray(dat_med, info=all_subs_evos[0].info, tmin=all_subs_evos[0].tmin)
        if cond == "two_objects": # store complexity evokeds
            dat_meds = {k: np.median([e.data for e in all_subs_evos_complexities[k]], 0) for k in ['0','1','2']}
            evos_complexities = {k: mne.EvokedArray(dat_meds[k], info=all_subs_evos[0].info, tmin=all_subs_evos[0].tmin) for k in ['0','1','2']} # grand evo for each complexity level
            # evos_complexities = {k: mne.grand_average(all_subs_evos_complexities[k]) for k in ['0','1','2']} # grand evo for each complexity level

        if args.save_evos:
            pickle.dump(evo, open(f"{out_dir}/grand-evos.p"))
            pickle.dump(all_subs_evos, open(f"{out_dir}/all_subs-evos.p"))
            pickle.dump(evos_complexities, open(f"{out_dir}/complexities-grand-evos.p"))
            pickle.dump(all_subs_evos_complexities, open(f"{out_dir}/complexities-all_subs-evos.p"))

    else:
        epochs, test_split_query_indices = load_data(args, fn, args.train_query_1, args.train_query_2)
        epochs = epochs.pick_types(meg=True)
        evo = deepcopy(epochs).average()
        # epochs = epochs[0] # keep only the first epochs, shoudl be containing all trials of interest
        # metadata = epochs.metadata
        evos_complexities = {k: deepcopy(epochs)[f"Complexity=={k}"].average() for k in ['0','1','2']}
    
    
    word_onsets, image_onset = get_onsets(cond)

    fig = evo.plot(spatial_colors=True)
    add_word_onsets(fig, word_onsets, image_onset)

    # evo_report.add_figs_to_section(fig, captions=cond, section=cond)
    evo_report.add_figure(fig, title=cond, section=cond)
    plt.close('all')

    if cond == "two_objects": # do complexity plots
        # evos = []
        for complexity in ['0','1','2']:
            # evos.append(deepcopy(epochs)[f"Complexity=={complexity}"].average())

            fig = evos_complexities[complexity].plot(spatial_colors=True)
            add_word_onsets(fig, word_onsets, image_onset)
            evo_report.add_figure(fig, title=f"Complexity {complexity}", section=cond)
            plt.close('all')

        # fig, ax = plt.subplots(dpi=1000)
        # mne.viz.plot_evoked_topo(evos_complexities.values(), background_color='w', axes=ax)
        # evo_report.add_figure(fig, title=f"all ch complexity", section=cond)
        # plt.close('all')

        fig = mne.viz.plot_compare_evokeds(evos_complexities, combine='gfp', legend='upper left', show_sensors='upper right')
        evo_report.add_figure(fig, title=f"complexities gfp", section=cond)
        plt.close('all')

        fig = mne.viz.plot_compare_evokeds(evos_complexities, combine='median', legend='upper left', show_sensors='upper right')
        evo_report.add_figure(fig, title=f"complexities median", section=cond)
        plt.close('all')

        # set_trace()
        # dat0 = np.concatenate([deepcopy(e)[f"Complexity==0"].get_data() for e in all_subs_epochs])
        # dat1 = np.concatenate([deepcopy(e)[f"Complexity==1"].get_data() for e in all_subs_epochs])
        # dat2 = np.concatenate([deepcopy(e)[f"Complexity==2"].get_data() for e in all_subs_epochs])
        
        # fig, ax = plt.subplots(dpi=400)
        # times = evo.times
        # _, clusters0, pclusters0, h0 = mne.stats.permutation_cluster_test([dat0, dat1], threshold=None, n_permutations=1024, out_type='mask')
        # # _, clusters1, pclusters1, h0 = mne.stats.permutation_cluster_test([dat_meds['1'], dat_meds['2']], threshold=None, n_permutations=1024, out_type='mask')
        # for i, complexity in enumerate(['0','1','2']):
        #     plt.plot(times, dat_meds, label=complexity)


# epo_report.save(out_dir + 'epochs_report.html', open_browser=False, overwrite=True)
evo_report.save(out_dir + 'evoked_report.html', open_browser=False, overwrite=True)


    # Epochs plots
    # for order in orders: # define order to sort the trials by
    # orders = dict(condition=np.argsort(metadata.condition).tolist(), 
    #             structure=np.argsort(metadata.sentence_structure).tolist())

    # for ch in epochs.ch_names:
    #     fig = epochs.plot_image(picks=ch) #, order=orders[order])            
    #     for w_onset in word_onsets:
    #          fig[0].axes[0].axvline(x=w_onset, linestyle='--', color='k')
    #     for img_onset in image_onset:
    #          fig.axes[0].axvline(x=img_onset, linestyle='-', color='k')

    #     epo_report.add_figs_to_section(fig, captions=ch, section=cond)
    #     # plt.savefig(out_dir + f'all_trials_{order}_{ch}.png')
    #     plt.close('all')

    # epo_report.add_epochs(epochs=epochs, title=f'Epochs {cond}')