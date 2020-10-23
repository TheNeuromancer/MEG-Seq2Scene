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

from utils.params import *

# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.

parser = argparse.ArgumentParser(description='MEG ICA components estimation')
parser.add_argument('-r', '--root-path', default='/neurospin/unicog/protocols/MEG/Seq2Scene/',help='Path to parent project folder')
parser.add_argument('-i', '--in-dir', default='Epochs',help='Input directory')
parser.add_argument('-s', '--subject', default='theo',help='subject name')
parser.add_argument('-w', '--overwrite', action='store_true',  default=False, help='Whether to overwrite the output directory')

print(mne.__version__)
mne.set_log_level(verbose='warning')
args = parser.parse_args()
print(args)

np.random.seed(42)

in_dir = op.join(args.root_path, 'Data', args.in_dir, args.subject)
all_epo_fns = sorted(glob(in_dir + f'/*-epo.fif'))
print(all_epo_fns)

ica_dir = op.join(args.root_path, 'Data', f'ICA_{args.in_dir}', args.subject)
all_ica_fns = sorted(glob(ica_dir + f'/*-ica.fif'))
print(all_ica_fns)

out_dir = op.join(args.root_path, 'Data', f'{args.in_dir}_after_ica', args.subject)

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


for epo_fn, ica_fn in zip(all_epo_fns, all_ica_fns):
    print(f"Doing files {epo_fn} and {ica_fn}")

    epochs = mne.read_epochs(epo_fn, preload=True)
    ica = mne.preprocessing.read_ica(ica_fn)

    cond = op.basename(epo_fn).split('-')[0]
    assert cond == op.basename(ica_fn).split('-')[0]

    report_fname = f'{out_dir}/{cond}-report.html'
    report = Report(report_fname, verbose=False)
    
    reject = ica_eog[f"{args.subject}_{cond}"]
    ica.exclude.extend(reject)

    # fig = epochs.plot_topo_image(vmin=-150, vmax=150, title='ERF images', layout_scale=1.)
    # report.add_figs_to_section(fig, section='Before', captions="Epochs") #, image_format='svg')
    fig = epochs.plot_image(vmin=-150, vmax=150, picks='mag', combine='mean')
    report.add_figs_to_section(fig, section='Before', captions="Evoked-Mag-Mean")
    fig = epochs.plot_image(vmin=-50, vmax=50, picks='grad', combine='mean')
    report.add_figs_to_section(fig, section='Before', captions="Evoked-Grad-Mean")
    fig = epochs.plot_image(vmin=180, vmax=250, picks='mag', combine='gfp')
    report.add_figs_to_section(fig, section='Before', captions="Evoked-Mag-GFP")
    fig = epochs.plot_image(vmin=30, vmax=70, picks='grad', combine='gfp')
    report.add_figs_to_section(fig, section='Before', captions="Evoked-Grad-GFP")
    evoked = epochs.average()
    fig = evoked.plot(spatial_colors=True, show=True)
    report.add_figs_to_section(fig, section='Before', captions="Evoked")
    plt.close('all')

    ica.apply(epochs)

    # fig = epochs.plot_topo_image(vmin=-150, vmax=150, title='ERF images', layout_scale=1.)
    # report.add_figs_to_section(fig, section='After', captions="Epochs") #, image_format='svg')
    fig = epochs.plot_image(vmin=-150, vmax=150, picks='mag', combine='mean')
    report.add_figs_to_section(fig, section='Before', captions="Evoked-Mag-Mean")
    fig = epochs.plot_image(vmin=-50, vmax=50, picks='grad', combine='mean')
    report.add_figs_to_section(fig, section='Before', captions="Evoked-Grad-Mean")
    fig = epochs.plot_image(vmin=180, vmax=250, picks='mag', combine='gfp')
    report.add_figs_to_section(fig, section='Before', captions="Evoked-Mag-GFP")
    fig = epochs.plot_image(vmin=30, vmax=70, picks='grad', combine='gfp')
    report.add_figs_to_section(fig, section='Before', captions="Evoked-Grad-GFP")
    evoked = epochs.average()
    fig = evoked.plot(spatial_colors=True, show=True)
    report.add_figs_to_section(fig, section='After', captions="Evoked")
    plt.close('all')

    tmin, tmax = tmin_tmax_dict[cond]
    epochs.crop(tmin, tmax)
    epochs.save(f"{out_dir}/{op.basename(epo_fn)}", overwrite=True)
    report.save(report_fname, overwrite=True, open_browser=False)