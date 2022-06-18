import mne
from sklearn.preprocessing import LabelEncoder
from natsort import natsorted
# from ipdb import set_trace
import numpy as np
from glob import glob
import time
import os.path as op
import os

from utils.params import TRIG_DICT


short_to_long_cond = {"loc": "localizer", "one_obj": "one_object", "two_obj": "two_objects",
                      "localizer": "localizer", "one_object":"one_object"}

full_fn_to_short = {"two_objects": 'scenes', "one_object": 'obj'}

colors = ["vert", "bleu", "rouge"]
shapes = ["triangle", "cercle", "carre"]

def num2sub_name(num, all_subjects):
  if not num.isdigit():
    return num
  else: 
    sub_name = [sub for sub in all_subjects if num == sub[0:2]]
    print(all_subjects)
    assert len(sub_name) == 1
    return sub_name[0]


def get_paths(args, dirname='Decoding'):
    subject_string = args.subject if args.subject!='grand' else ''
    in_dir = f"{args.root_path}/Data/{args.epochs_dir}/{subject_string}"
    print("\nGetting training filename:")
    print(in_dir + f'/{args.train_cond}*-epo.fif')
    if isinstance(args.train_cond, list):
        print("Getting MULTIPLE training filenameS:")
        train_fn = [natsorted(glob(in_dir + f'/{cond}*-epo.fif'))[0] for cond in args.train_cond]
    else:
        print(in_dir + f'/{args.train_cond}*-epo.fif')
        train_fn = natsorted(glob(in_dir + f'/{args.train_cond}*-epo.fif'))[0]
        # assert len(train_fn) == 1
    print(train_fn)
    print("\nGetting test filenames:")
    test_fns = [natsorted(glob(in_dir + f'/{cond}*-epo.fif'))[0] for cond in args.test_cond]
    print(test_fns)

    ### SET UP OUTPUT DIRECTORY AND FILENAME
    out_fn = get_out_fn(args, dirname=dirname)
    print(f"out fn: {out_fn}")

    test_out_fns = []
    if hasattr(args, "test_query_1") and hasattr(args, "test_query_2"): # typically for classical decoding
        for i_fn, (test_cond, test_query1, test_query2) in enumerate(zip(args.test_cond, args.test_query_1, args.test_query_2)):
            test_query_str = f"{'_'.join(test_query1.split())}_vs_{'_'.join(test_query2.split())}"
            # add an int to the label to split the different tests
            new_out_fn = f"{out_fn.split('-')[0]}_{i_fn}-{'-'.join(out_fn.split('-')[1::])}"
            test_out_fns.append(shorten_filename(f"{new_out_fn}_tested_on_{test_cond}_{test_query_str}"))
    elif hasattr(args, "test_query"): # typically for OVR and window decoding
        for i_fn, (test_cond, test_query) in enumerate(zip(args.test_cond, args.test_query)):
            test_query_str = '_'.join(test_query.split())
            new_out_fn = f"{out_fn.split('-')[0]}_{i_fn}-{'-'.join(out_fn.split('-')[1::])}" # add an int to the label to split the different tests
            test_out_fns.append(shorten_filename(f"{new_out_fn}_tested_on_{test_cond}_{test_query_str}"))

    # wait for a random time in order to avoid conflit (parallel jobs that try to construct the same directory)
    rand_time = float(str(abs(hash(str(args))))[0:8]) / 100000000
    print(f'sleeping {rand_time} seconds to desynchronize parallel scripts')
    time.sleep(rand_time)
    
    out_dir = op.dirname(out_fn)
    if not op.exists(out_dir):
        rand_time = float(str(abs(hash(str(args))))[0:8]) / 10000000
        print(f'sleeping some {rand_time} more seconds before attempting to create out dir')
        time.sleep(rand_time)
        if not op.exists(out_dir):
            print('Constructing output directory')
            os.makedirs(out_dir)
    else:
        print('output directory already exists')
        if op.exists(f"{out_dir}_AUC.npy"):
            if args.overwrite:
                print('overwrite is set to True ... overwriting\n')
            else:
                print('overwrite is set to False ... exiting smoothly')
                exit()
    return train_fn, test_fns, out_fn, test_out_fns


def get_out_fn(args, dirname='Decoding'):
    if args.dummy: # temporary directory
        out_dir = f'{args.root_path}/Results/TMP/{dirname}_v{args.version}/{args.epochs_dir}/{args.subject}'
    else:
        out_dir = f'{args.root_path}/Results/{dirname}_v{args.version}/{args.epochs_dir}/{args.subject}'

    cat_string = f"_{args.cat}cat" if args.cat else ""
    cat_string = f"{cat_string[0:-3]}mean" if args.mean else cat_string
    reduc_dim_str = f"_reduc{args.reduc_dim}comp" if args.reduc_dim else "" # %50 
    baseline_str = "_baseline" if args.baseline else ""
    smooth_str = f"_{args.smooth}smooth" if args.smooth else ""
    shuffle_str = '_shuffled' if args.shuffle else ''
    fband_str = f'_{args.freq_band}' if args.freq_band else ''
    cond_str = f"_cond-{args.train_cond}-" if isinstance(args.train_cond, str) else "" # empty string if we have a list of training conditions.
    if hasattr(args, 'train_query_1'):
        if args.train_query_1 and args.train_query_2:
            train_query_1 = '_'.join(args.train_query_1.split())
            train_query_2 = '_'.join(args.train_query_2.split())
        else:
            train_query_1 = ''
            train_query_2 = ''
        out_fn = f'{out_dir}/{args.label}-{train_query_1}_vs_{train_query_2}{reduc_dim_str}{shuffle_str}{fband_str}{cond_str}'
    else: # RSA
        out_fn = f'{out_dir}/{args.label}-{reduc_dim_str}{shuffle_str}{fband_str}{cond_str}'

    out_fn += "dawn-" if args.xdawn else ""
    out_fn += "autoreject-" if args.autoreject else ""
    out_fn += args.filter if args.filter else ""

    out_fn = shorten_filename(out_fn)
    print('\noutput file will be in: ' + out_fn)
    print('eg:' + out_fn + '_AUC_diag.npy\n')
    return out_fn


def shorten_filename(fn):
    # shorten the output fn because we sometimes go over the 255-characters limit imposed by ubuntu
    fn = fn.replace("'", "")
    fn = fn.replace('"', '')
    fn = fn.replace('[', '')
    fn = fn.replace('(', '')
    fn = fn.replace(')', '')
    fn = fn.replace(']', '')
    fn = fn.replace(',', '')
    fn = fn.replace('Colour', 'C')
    fn = fn.replace('Shape', 'S')
    fn = fn.replace('Binding', 'Bd')
    fn = fn.replace('Object', 'Obj')
    fn = fn.replace('Relation', 'R')
    fn = fn.replace('two_objects', 'scenes')
    fn = fn.replace('one_object', 'obj')
    fn = fn.replace('cercle', 'cl')
    fn = fn.replace('carre', 'ca')
    fn = fn.replace('triangle', 'tr')
    fn = fn.replace('bleu', 'bl')
    fn = fn.replace('vert', 'vr')
    fn = fn.replace('rouge', 'rg')
    fn = fn.replace('==', '=')
    fn = fn.replace('Matching=match', 'match')
    fn = fn.replace('Matching=nonmatch', 'nonmatch')
    fn = fn.replace('Flash=0', 'noflash')
    fn = fn.replace('Flash=1', 'flash')
    
    # if fn is still too long, make some ugly changes
    if len(fn) > 255:
        fn = fn.replace('reduc', '')
        fn = fn.replace('baseline', 'bl')
        fn = fn.replace('and_', '')
        fn = fn.replace('or_', '_')
        fn = fn.replace('cond', 'cd')

    # if fn is still too long, make some even uglier changes
    if len(fn) > 255: # remove underscores but only in the fn, not in the path
        fn = fn.replace(op.basename(fn), op.basename(fn).replace('_', ''))

    # if fn is still too long, crop brutally
    if len(fn) > 255:    
        print(f'\n\nOutput fn is too long.\n\n{fn}\nis longer than UNIX limit which is 255 characters...cropping brutally')
        fn = fn[0:255]

    return fn

def back2fullname(name):
    name = name.replace('R', 'Relation')
    name = name.replace('C', 'Colour')
    name = name.replace('S', 'Shape')
    name = name.replace('All1stObj', 'First Object')
    name = name.replace('All2ndObj', 'Second Object')
    return name

def get_onsets(cond, version="v1"):
    """ get the word and image onsets depending on the condition
    """
    if version == "v1": # first version, longger SOA
        SOA_dict = {"localizer": .9, "one_object": .65, "two_objects": .65, "obj": .65, "scenes": .65}
    else: # second version for subject 9 and up
        SOA_dict = {"localizer": .9, "one_object": .6, "two_objects": .6, "obj": .6, "scenes": .6}
    delay_dict = {"localizer": None, "one_object": 1., "two_objects": 2., "obj": 1., "scenes": 2.}
    nwords_dict = {"localizer": 1, "one_object": 2, "two_objects": 5, "obj": 2, "scenes": 5}

    SOA = SOA_dict[cond]
    delay = delay_dict[cond]
    nwords = nwords_dict[cond]

    word_onsets = []
    image_onset = []
    for i_w in range(nwords):
        word_onsets.append(i_w * SOA)

    if delay: # for one_object and two_objects conditions
        image_onset.append((i_w+1) * SOA + delay)

    return word_onsets, image_onset


def Xdawn(epochs4xdawn, epochs2transform, factor, n_comp=10):
    md = epochs4xdawn.metadata
    y = np.sum([md[subfac] for subfac in factor.split("+")], 0)
    labencod = LabelEncoder()
    y = labencod.fit_transform(y)
    xdawn = mne.preprocessing.Xdawn(n_components=n_comp, correct_overlap=False)
    xdawn.fit(epochs4xdawn, y=y)
    epochs = xdawn.apply(epochs2transform)['1']
    return epochs
    # evo = epochs.average()
    # plot = evo.plot(spatial_clors=True)
    # plt.savefig('tmp2.png')

    # epochs2 = xdawn.apply(epochs)
    # evo = epochs2['1'].average()
    # plot = evo.plot(spatial_colors=True)
    # plt.savefig('tmp2.png')


def get_flash_indices(events):
    ''' get trials where a flash was 
    presented right before the image '''
    events_ids = events[:,2]
    trial_starts = np.where(events_ids == TRIG_DICT['two_objects_trial_start'])[0]
    flashes = np.where(events_ids == 5)[0] # 5 is the trigger value for the flash
    print(f"Found {len(flashes)} flashes")
    if not len(flashes): # early subject did not have any flash
        return [False for _ in trial_starts]
    all_flash_present = []
    for i_trial in range(len(trial_starts)-1):
        flash_present = False # assume no flash
        for flash in flashes: # check that a flash happened between the start and end of the trial
            if trial_starts[i_trial] < flash < trial_starts[i_trial+1]:
                flash_present = True
                break
        all_flash_present.append(flash_present)
    if flashes[-1] > trial_starts[-1]: # we still have the last trial to test for the presence of a flash
        all_flash_present.append(True)
    else:
        all_flash_present.append(False)
    return all_flash_present


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    
    see also: 
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')

    return y[(window_len//2):-(window_len//2)]
    # return y[(window_len//2-1):-(window_len//2)]
    # return y


def predict(clf, data, multiclass=False):
    """
    wrapper for predicting from any sklearn classifier
    """
    try:
        if multiclass:
            pred = clf.predict_proba(data)
        else:
            pred = clf.predict_proba(data)[:,1] # no multiclass so just keep the proba for class 1
    except AttributeError: # no predict_proba method
        pred = clf.predict(data)
    return pred


def complement_md(md):
    """ add entries to metadata:
    Right_obj and Left_obj = what object was on which side
    Complexity = 2 minus number of shared properties
    """
    right_obj, left_obj = [], [] 
    for i, line in md.iterrows():
        if line.Relation == "à gauche d'":
            left_obj.append(f"{line.Shape1}_{line.Colour1}")
            right_obj.append(f"{line.Shape2}_{line.Colour2}")
        elif line.Relation == "à droite d'":
            right_obj.append(f"{line.Shape1}_{line.Colour1}")
            left_obj.append(f"{line.Shape2}_{line.Colour2}")
        else:
            raise RuntimeError("Could not parse metadata to get left and right obejcts")
    md["Right_obj"] = right_obj
    md["Left_obj"] = left_obj
    # from ipdb import set_trace; set_trace()
    return md

def add_complexity_to_md(line):
    complexity = 2
    if line.Shape1 == line.Shape2:
        complexity -= 1
    if line.Colour1 == line.Colour2:
        complexity -= 1
    return complexity


def get_ylabel_from_fn(fn):
    # acc or AUC
    if fn[-7:-4] == 'acc' or fn[-12:-9] == 'acc':
        ylabel = 'Accuracy'
    elif fn[-7:-4] == 'AUC' or fn[-12:-9] == 'AUC':
        ylabel = 'AUC'
    elif fn[-9:-4] == 'preds': # or fn[-12:-9] == 'AUC':
        ylabel = 'prediction'
    elif 'Accuracy' in fn:
        ylabel = 'Accuracy'
    elif 'AUC' in fn:
        ylabel = 'AUC'
    else:
        print('\n\nDid not find a correct label in the filename')
        set_trace()
    return ylabel



prop2cond = {'triangle': "shape", 'cercle': "shape", 'carre': "shape", 'vert': "color", 'bleu': "color", 'rouge': "color", 
            'triangle vert': "object", 'triangle bleu': "object", 'triangle rouge': "object", 'cercle vert': "object", 
            'cercle bleu': "object", 'cercle rouge': "object", 'carre vert': "object", 'carre bleu': "object", 'carre rouge': "object", 
            'triangle 1': "shape1", 'cercle 1': "shape1", 'carre 1': "shape1", 'vert 1': "color1", 'bleu 1': "color1", 'rouge 1': "color1", 
            'triangle 2': "shape2", 'cercle 2': "shape2", 'carre 2': "shape2", 'vert 2': "color2", 'bleu 2': "color2", 'rouge 2': "color2", 
            'triangle vert 1': "object1", 'triangle bleu 1': "object1", 'triangle rouge 1': "object1", 'cercle vert 1': "object1", 
            'cercle bleu 1': "object1", 'cercle rouge 1': "object1", 'carre vert 1': "object1", 'carre bleu 1': "object1", 'carre rouge 1': "object1", 
            'triangle vert 2': "object2", 'triangle bleu 2': "object2", 'triangle rouge 2': "object2", 'cercle vert 2': "object2", 
            'cercle bleu 2': "object2", 'cercle rouge 2': "object2", 'carre vert 2': "object2", 'carre bleu 2': "object2", 'carre rouge 2': "object2"}


def group_conds(properties):
    """group back individual conditions, 
    eg: rouge to Colors
    """ 
    all_conds = [] # all condition names, same order as properties
    unique_conds = [] # set of all condition names
    conds_indices = [] # all indices of conditions relative to unique_conds
    groups_indices = [] # all indices of conditions to be grouped together
    for lab in properties:
        cond = prop2cond[lab]
        all_conds.append(cond)
        if cond not in unique_conds:
            unique_conds.append(cond)
        conds_indices.append(unique_conds.index(cond))
    groups_indices = [np.where(idx==conds_indices)[0] for idx in np.unique(conds_indices)]
    return unique_conds, groups_indices
    # return all_conds, unique_conds, conds_indices, groups_indices


def win_ave_smooth(data, nb_cat, mean=True):
    """ smoothing throughmoving average window
    data should be a list if np arrays of shape
    n_epochs * n_ch * n_times
    """
    if nb_cat == 0: return data

    # needs a list of epochs data
    if not isinstance(data, list): data = [data]

    # loop over datum
    for i_d, query_data in enumerate(data):
        sz = query_data.shape
        if mean:
            new_data = np.zeros_like(query_data)
        else:
            new_data = np.zeros((sz[0], sz[1]*nb_cat, sz[2]))

        for t in range(sz[2]):
            nb_to_cat = nb_cat if t>nb_cat else t
            if mean: # average consecutive timepoints
                new_data[:,:,t] = query_data[:,:,t-nb_to_cat:t+1].mean(axis=2)
            else: # concatenate
                if nb_to_cat < nb_cat: # we miss some data points before tmin 
                    # just take the first timesteps and copy them
                    dat = query_data[:,:,t-nb_to_cat:t+1]
                    # dat = dat.reshape(sz[0], sz[1] * dat.shape[2])
                    while dat.shape[2] < nb_cat:
                        idx = np.random.choice(nb_to_cat+1) # take a random number below the current timepoint
                        dat = np.concatenate((dat, dat[:,:,idx,np.newaxis]), axis=2) # add it to the data
                    new_data[:,:,t] = dat.reshape(sz[0], sz[1] * nb_cat)
                else:
                    new_data[:,:,t] = query_data[:,:,t-nb_to_cat:t].reshape(sz[0], sz[1] * nb_to_cat)
        data[i_d] = new_data
    return data


def quality_from_cond(sub, cond, label='Matching', scoring='max', dir='/neurospin/unicog/protocols/MEG/Seq2Scene/Results/Decoding_test_quality_v7/Quality_test'):
    """ get the score (from decoding_test_quality.sh)
    for a single run
    """ 
    run_fns = glob(f"{dir}/{sub}_{label}-_cond-{cond}-*")
    sep_char = "_"if label == "Matching" else "#"
    runs = [op.basename(fn).split('run_nb=')[-1].split(sep_char)[0] for fn in run_fns]
    score_per_run = {}
    for run in runs:
        _fn = f"{dir}/{sub}_{label}-_cond-{cond}-run_nb={run}*.txt" 
        fn = glob(_fn)
        if not fn: return
        if len(fn) > 1: from ipdb import set_trace; set_trace()
        score = float(op.basename(fn[0]).split(scoring)[-1][0:5])
        score_per_run[run] = score
    return score_per_run

    # from ipdb import set_trace; set_trace()
