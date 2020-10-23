import mne
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from ipdb import set_trace
import argparse
from glob import glob
import os.path as op
import os
import os.path as op

from mne.report import Report
from mne.preprocessing import ICA

# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.

parser = argparse.ArgumentParser(description='MEG ICA components estimation')
parser.add_argument('-r', '--root-path', default='/neurospin/unicog/protocols/MEG/Seq2Scene/',help='Path to parent project folder')
parser.add_argument('-i', '--in-dir', default='Epochs',help='Input directory')
parser.add_argument('-s', '--subject', default='theo',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')
parser.add_argument('--plot', default=False, action='store_true', help='Whether to make an MNE Report with ICA results')

print(mne.__version__)
mne.set_log_level(verbose='warning')
args = parser.parse_args()
print(args)

np.random.seed(42)

in_dir = op.join(args.root_path, 'Data', args.in_dir, args.subject)
all_epo_fns = sorted(glob(in_dir + f'/*-epo.fif'))
print(all_epo_fns)
out_dir = op.join(args.root_path, 'Data', f'ICA_{args.in_dir}', args.subject)

## make output dir
if not op.exists(out_dir):
            os.makedirs(out_dir)

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

for fn in all_epo_fns:
  epochs = mne.read_epochs(fn, preload=True, verbose=False)

  # print('look into reject based on EOG ...')
  # # don't reject based on EOG to keep blink artifacts
  # # in the ICA computation.
  # reject_ica = config.reject
  # if reject_ica and 'eog' in reject_ica:
  #     reject_ica = dict(reject_ica)
  #     del reject_ica['eog']

  # produce high-pass filtered version of the data for ICA
  epochs_for_ica = epochs.copy() # .filter(l_freq=1., h_freq=None, filter_length=f'{epochs.tmax-epochs.tmin}s')

  print("  Running ICA...")

  # run ICA on MEG
  picks = mne.pick_types(epochs_for_ica.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
  ch_type = 'meg'
  n_components = 0.999

  print('Running ICA for MEG')
  ica = ICA(method='fastica', random_state=42, n_components=n_components, verbose='warning')
  ica.fit(epochs_for_ica, picks=picks, decim=1, verbose='warning')

  print('  Fit %d components (explaining at least %0.1f%% of the variance)' % (ica.n_components_, 100 * n_components))

  ica_fname = f'{out_dir}/{op.basename(fn).split("-")[0]}-ica.fif'
  ica.save(ica_fname)

  if args.plot:
      # plot ICA components to html report
      report_fname = f'{out_dir}/{op.basename(fn).split("-")[0]}-report.html'
      report = Report(report_fname, verbose=False)

      for idx in range(0, ica.n_components_):
          figure = ica.plot_properties(epochs_for_ica, picks=idx, psd_args={'fmax': 60}, show=False)
          report.add_figs_to_section(figure, section=args.subject, captions=(ch_type.upper()+' - ICA Components'))

      report.save(report_fname, overwrite=True, open_browser=False)