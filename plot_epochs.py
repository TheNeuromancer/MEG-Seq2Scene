import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.use('Agg') # no output to screen.
import mne
import numpy as np
from ipdb import set_trace
import argparse
import pickle
import time
import os.path as op

from utils.decod import *

parser = argparse.ArgumentParser(description='SEEG high gammas extraction and plotting of evoked power')
parser.add_argument('-r', '--root-path', default='/neurospin/unicog/protocols/MEG/Seq2Scene/', help='root path')
parser.add_argument('-i', '--in-dir', default='Data/Epochs/', help='input directory')
parser.add_argument('-o', '--out-dir', default='/Results/Epochs', help='output directory')
parser.add_argument('-p', '--subject', default='theo',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('-v', '--version', default='1', help='version of the script')
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
parser.add_argument('--path2loc', default='Single_Chan_vs5/CMR_sent', help='path to the localizer results (dict with value 1 for each channel that passes the test, 0 otherwise')
parser.add_argument('--pval-thresh', default=0.05, type=float, help='pvalue threshold under which a channel is kept for the localizer')
parser.add_argument('--freq-band', default='', help='name of frequency band to use for filtering (theta, alpha, beta, gamma)')

# we hack these from utils/decod.py to load data
parser.add_argument('--train-cond', default='localizer', help='localizer, one_object or two_objects')
parser.add_argument('--test-cond', default=['localizer', 'imgloc', 'one_object', 'two_objects'], help='localizer, one_object or two_objects')
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

np.random.seed(args.seed)
start_time = time.time()

### GET EPOCHS FILENAMES ###
_, all_fns, _ = get_paths(args) # discard the first fn and use the test_fns to get all possible fns
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

epo_report = mne.report.Report()
evo_report = mne.report.Report()

for fn, cond in zip(all_fns, args.test_cond):
    print("Doing: ", fn, cond)

    ### LOAD EPOCHS ###
    epochs, test_split_query_indices = load_data(args, fn, args.train_query_1, args.train_query_2)
    epochs = epochs[0] # keep only the first epochs, shoudl be containing all trials of interest
    epochs = epochs.pick_types(meg=True)
    metadata = epochs.metadata

    word_onsets, image_onset = get_onsets(cond)

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

    evo = epochs.average()
    fig = evo.plot(spatial_colors=True)
    for w_onset in word_onsets:
        fig.axes[0].axvline(x=w_onset, linestyle='--', color='k')
        fig.axes[1].axvline(x=w_onset, linestyle='--', color='k')
    for img_onset in image_onset:
        fig.axes[0].axvline(x=img_onset, linestyle='-', color='k')
        fig.axes[1].axvline(x=img_onset, linestyle='-', color='k')

    evo_report.add_figs_to_section(fig, captions=cond, section=cond)
    plt.close('all')

epo_report.save(out_dir + 'epochs_report.html', open_browser=False, overwrite=True)
evo_report.save(out_dir + 'evoked_report.html', open_browser=False, overwrite=True)