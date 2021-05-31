import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import mne
import numpy as np
from ipdb import set_trace
import argparse
import pickle
import time
import importlib 
import mne_rsa

from utils.decod import plot_diag
from utils.RSA import *

parser = argparse.ArgumentParser(description='MEG Decoding analysis')
parser.add_argument('-c', '--config', default='config', help='path to config file')
parser.add_argument('-s', '--subject', default='theo',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle sentence labels before training')
parser.add_argument('--freq-band', default='', help='name of frequency band to use for filtering (theta, alpha, beta, gamma)')
# parser.add_argument('--timegen', action='store_true', default=False, help='Whether to test probe trained at one time point also on all other timepoints')
# parser.add_argument('--query', default='Colour1', help='Metadata query for training classes')
parser.add_argument('--train-cond', default='localizer', help='localizer, one_object or two_objects')
parser.add_argument('--test-cond', default=[], action='append', help='localizer, one_object or two_objects')
parser.add_argument('--label', default='', help='help to identify the result latter')
parser.add_argument('--filter', default='', help='md query to filter trials before anything else (eg to use only matching trials')

parser.add_argument('--distance_metric', default='confusion', help='Metric to compute RSA')
parser.add_argument('--rsa_metric', default='spearman', help='Metric to compute RSA')
parser.add_argument('--min-nb-trial', default=2, type=int, help='minimum number of trial in a class to keep it in the decoding to get confusion matrices')
parser.add_argument('-x', '--xdawn', action='store_true',  default=False, help='Whether to apply Xdawn spatial filtering before training decoder')

# not implemented
parser.add_argument('--localizer', action='store_true', default=False, help='Whether to use only electrode that were significant in the localizer')
parser.add_argument('--path2loc', default='Single_Chan_vs5/CMR_sent', help='path to the localizer results (dict with value 1 for each channel that passes the test, 0 otherwise')
parser.add_argument('--pval-thresh', default=0.05, type=float, help='pvalue threshold under which a channel is kept for the localizer')
args = parser.parse_args()

# import config parameters
config = importlib.import_module(f"configs.{args.config}", "Config").Config()
# update argparse with arguments from the config
for arg in vars(config): setattr(args, arg, getattr(config, arg))
args.subject = num2sub_name(args.subject, args.all_subjects) # get full subject name if only the number was passed as argument
print(args)
print("matplotlib: ", matplotlib.__version__)

np.random.seed(args.seed)
start_time = time.time()

###########################
######## LOADING  #########
###########################
### GET EPOCHS FILENAMES ##
print('\nLoading epochs')

train_fn, base_out_fn = get_paths_rsa(args)
out_fn = f"{base_out_fn}_{args.distance_metric}_{args.rsa_metric}"
print(f"full out fn: {out_fn}")
epochs = load_data_rsa(args, train_fn)
if args.train_cond == "two_objects":
    epochs.metadata = complement_md(epochs.metadata)

if args.xdawn and args.train_cond == "two_objects":
    path_oneobj = f"{op.dirname(train_fn)}/one_object-epo.fif"
    epochs4xdawn = load_data_rsa(args, train_fn)
    epochs = Xdawn(epochs4xdawn=epochs4xdawn, epochs2transform=epochs, factor="Shape1+Colour1")


###########################
##### MODEL MATRICES ######
###########################
print('\nStarting RSA')

md = epochs.metadata
model_dsms = []
if args.train_cond == "localizer":
    factors = ["Loc_word"]
elif args.train_cond == "one_object":
    factors = ["Colour1", "Shape1", "Shape1+Colour1"]
elif args.train_cond == "two_objects":
    if args.distance_metric == "confusion":
        # factors = ["Shape", "Colour", "Shape+Colour", "Left_obj", "Right_obj", "Right_obj+Left_obj"]
        # factors = ["Shape", "Colour", "Shape+Colour", "Shape1+Colour1+Shape2+Colour2"]
        factors = ["Shape1", "Colour1", "Shape2", "Colour2", "Shape1+Colour1", "Shape2+Colour2", "Relation", "Right_obj", "Left_obj"]
    else:
        factors = ["Shape1", "Colour1", "Shape2", "Colour2", "Shape1+Colour1", "Shape2+Colour2", "Relation", "Right_obj", "Left_obj"] #, "Colour1+Colour2", "Shape1+Shape2", "Shape1+Colour1+Shape2+Colour2"]
# factors += ["Matching"]
n_factors = len(factors)

print("do we want to decode the order in which stims were presented, or the actual place on the screen that has to be imagined for the matching task? \
in any case, we can use the other as a dsm predictor ...")

dsm_models = []
if args.distance_metric == "confusion":
    class_queries = get_class_queries(args.train_cond)
    for factor in factors:
        dsm_models.append(get_dsm_from_queries(factor, class_queries))

else: # any other distance metric
    factor_values = []
    for factor in factors:
        # if "+" in factor: # joint factor
        #     # factor_values.append("_".join([md[subfac] for subfac in factor.split("+")]))
        factor_values.append(np.sum([md[subfac] for subfac in factor.split("+")], 0))
        # else: # simple factor
        #     factor_values.append(md[factor])

    for factor_val in factor_values:
        dsm_models.append(squareform(mne_rsa.compute_dsm(factor_val, metric=lambda x1, x2: 1 if x1!=x2 else 0)))

    # PLOT reduced models dsms
    reduced_dsm_models = []
    for factor_val in factor_values:
        reduced_dsm_models.append(mne_rsa.compute_dsm(np.unique(factor_val), metric=lambda x1, x2: 1 if x1!=x2 else 0))
    fig = mne_rsa.plot_dsms(reduced_dsm_models, names=factors)
    plt.savefig(f'{out_fn}_dsms_reduced.png')
    plt.close()


## PLOT MODELS
fig = mne_rsa.plot_dsms(dsm_models, names=factors)
plt.savefig(f'{out_fn}_dsms_full.png')
plt.close()



###########################
######## RSA PROPER #######
###########################

data = epochs.get_data()
n_times = data.shape[2]
times = np.linspace(epochs.tmin, epochs.tmax, n_times)
version = "v1" if int(args.subject[0:2]) < 8 else "v2" # first 8 subjects, different timings

# n_times=2
if args.distance_metric == "confusion":
    path2confusions = f"{base_out_fn}_confusions.npy"
    if op.exists(path2confusions):
        print("Loading confusion matrices from disk")
        confusion_matrices = np.load(path2confusions)
        AUC = np.load(f"{base_out_fn}_AUC.npy")
    else:
        print("Computing confusion matrices")
        AUC, confusion_matrices = decoder_confusion(args, epochs, class_queries, n_times)
        np.save(path2confusions, confusion_matrices)
        np.save(f"{base_out_fn}_AUC.npy", AUC)
        
    plot_diag(data_mean=AUC, out_fn=f"{base_out_fn}_AUC.png", train_cond=args.train_cond, 
              train_tmin=times[0], train_tmax=times[-1], ylabel='AUC', version=version)
    # confusion_matrices is of shape (n_times*n_classes*n_classes)
    
    if args.rsa_metric != "regression":
        scoring_fn = spearmanr if args.rsa_metric=="spearman" else pearsonr
        rsa_results = np.zeros((n_factors, n_times))
        for i_fac, factor in enumerate(factors):
            print(f"doing factor {factor}")
            for t in range(n_times):
                rsa_results[i_fac, t] = scoring_fn(1 - confusion_matrices[t].ravel(), dsm_models[i_fac].ravel())[0]
            
            fig = mne_rsa.plot_dsms(np.mean(confusion_matrices, 0), names=factor)
            plt.savefig(f'{out_fn}_dsms_confusion_{factor}.png')
            plt.close()

    elif args.rsa_metric == "regression":
        dsm_models_for_reg = np.array([d.ravel() for d in dsm_models]).T
        rsa_results = np.zeros((n_factors, n_times))
        for t in range(n_times):
            rsa_results[:,t] = regression_score(confusion_matrices[t].ravel(), dsm_models_for_reg)
        # normalize betas
        # scaler = MinMaxScaler()
        # rsa_results = scaler.fit_transform(rsa_results.T).T

else: # anything else that confusion matrices
    def generate_meg_dsms():
        """Generate DSMs for each time sample."""
        for t in range(n_times):
            # yield mne_rsa.compute_dsm_cv(data[:, :, t], metric='correlation')
            yield mne_rsa.compute_dsm(data[:, :, t], metric=args.distance_metric)

    rsa_results = mne_rsa.rsa(generate_meg_dsms(), dsm_models, metric=args.rsa_metric, verbose=True, n_data_dsms=n_times, n_jobs=-1)
    rsa_results = rsa_results.T # transpose to get n_factors as the first dimension

## PLOT RESULTS
plot_rsa(rsa_results, factors, out_fn, times, cond=args.train_cond, data_std=None, ylabel=args.rsa_metric, version=version)
if len(factors) > 1:
    multi_plot_rsa(rsa_results, factors, out_fn, times, cond=args.train_cond, ylabel=args.rsa_metric, version=version)

## SAVE RESULTS
save_rsa_results(args, out_fn, rsa_results, factors)

