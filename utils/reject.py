import mne
import numpy as np
import os.path as op
import os
from scipy import signal
from copy import deepcopy
# from ipdb import set_trace
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_deviant_maxwell(scores, out_fn):
	for ch_type in ['grad', 'mag']:
		ch_subset = scores['ch_types'] == ch_type
		ch_names = scores['ch_names'][ch_subset]
		scores_data = scores['scores_noisy'][ch_subset]
		limits = scores['limits_noisy'][ch_subset]
		bins = scores['bins']  # The the windows that were evaluated.
		# We will label each segment by its start and stop time, with up to 3
		# digits before and 3 digits after the decimal place (1 ms precision).
		bin_labels = [f'{start:3.3f} – {stop:3.3f}' for start, stop in bins]

		# We store the data in a Pandas DataFrame. The seaborn heatmap function
		# we will call below will then be able to automatically assign the correct
		# labels to all axes.
		data_to_plot = pd.DataFrame(data=scores_data, columns=pd.Index(bin_labels, name='Time (s)'), index=pd.Index(ch_names, name='Channel'))

		# First, plot the "raw" scores.
		fig, ax = plt.subplots(1, 2, figsize=(12, 8))
		fig.suptitle(f'Automated noisy channel detection: {ch_type}',
		             fontsize=16, fontweight='bold')
		sns.heatmap(data=data_to_plot, cmap='Reds', cbar_kws=dict(label='Score'),
		            ax=ax[0])
		[ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
		    for x in range(1, len(bins))]
		ax[0].set_title('All Scores', fontweight='bold')

		# Now, adjust the color range to highlight segments that exceeded the limit.
		sns.heatmap(data=data_to_plot,
		            vmin=np.nanmin(limits),  # bads in input data have NaN limits
		            cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
		[ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
		    for x in range(1, len(bins))]
		ax[1].set_title('Scores > Limit', fontweight='bold')

		# The figure title should not overlap with the subplots.
		fig.tight_layout(rect=[0, 0.03, 1, 0.95])
		plt.savefig(f"{out_fn}_{ch_type}_maxwell_filter_scores.png")
		plt.close()



def get_deviant_ch(raw, thresh=10):
	# thresh: number of time the median in order to consider a channel bad

	print('Computing temporal variance for channels rejection... threshold is set to ', thresh)
	all_deviants = set()

	for ch_type in ['grad', 'mag']:
		local_raw = raw.copy().pick([ch_type])
		ch_names = local_raw.ch_names
		
		# Get the data
		raw_data = local_raw.get_data()
		# Detrend the data
		raw_data = signal.detrend(raw_data)

		# Get the channel-variance per modality
		variance = np.var(raw_data, axis=1)
		# Get the median of variance
		variance_median = np.median(variance)

		highs = np.where(variance > thresh*variance_median)[0].tolist()
		lows = np.where(variance < variance_median/thresh)[0].tolist()
		deviants = lows + highs
		deviants_ch_names = [ch_names[i] for i in deviants]
		# print('found deviants: ', deviants_ch_names)


		# Get the gradient deviant sensors
		raw_data_grad = np.gradient(raw_data, axis=1)

		# Get the channel-variance per modality
		gradient_variance = np.var(raw_data, axis=1)
		# Get the median of gradient_variance
		gradient_variance_median = np.median(gradient_variance)

		highs = np.where(gradient_variance > thresh*gradient_variance_median)[0].tolist()
		lows = np.where(gradient_variance < variance_median/thresh)[0].tolist()
		gradient_deviants = lows + highs
		gradient_deviants_ch_names = [ch_names[i] for i in gradient_deviants]
		# print('found deviants with gradient: ', gradient_deviants_ch_names)

		all_deviants.update(set(deviants_ch_names + gradient_deviants_ch_names))

	print(f'found {len(all_deviants)} deviants: {all_deviants}')

	return all_deviants


def get_deviant_epo(epo, thresh=5):
	# thresh: number of time the median in order to consider a channel bad; for a given trial
	# Now gets the variance across time points and channels instead of separately for each channel

	print('Computing temporal variance for epochs rejection... threshold is set to ', thresh, ('above this times the std to be considered deviant)'))
	print('std is computed across channels andtime points')

	local_epo = epo.copy().pick(['meg'])

	# Get the data
	if isinstance(local_epo, mne.time_frequency.EpochsTFR):
		epo_data = local_epo.data
	else:
		epo_data = local_epo.get_data()

	# Get the channel-variance per modality
	variance = np.var(epo_data, axis=(1,2))
	# Get the median of variance
	variance_median = np.median(variance, 0)

	highs = np.where(variance > thresh*variance_median)[0].tolist()
	lows = np.where(variance < variance_median/thresh)[0].tolist()
	
	deviants = lows + highs
	# print('found deviant epochs: ', deviants)

	# Get the gradient deviant sensors
	epo_data_grad = np.gradient(epo_data, axis=2)
	# Get the channel-variance per modality
	gradient_variance = np.var(epo_data, axis=(1,2))
	# Get the median of gradient_variance
	gradient_variance_median = np.median(gradient_variance, 0)

	highs_gradient = np.where(gradient_variance > thresh*gradient_variance_median)[0].tolist()
	lows_gradient = np.where(gradient_variance < gradient_variance_median/thresh)[0].tolist()
	
	gradient_deviants = lows_gradient + highs_gradient
	# print('found deviant epochs with gradient: ', gradient_deviants)

	all_deviants = set(deviants + gradient_deviants)

	print(f'found {len(all_deviants)} deviant epochs: {all_deviants}')

	return all_deviants


def reject_bad_epochs(epo, bad_epo):
	bad_epo = [b for b in bad_epo] # change to list
	if isinstance(epo, mne.time_frequency.EpochsTFR):
		data = epo.data
	else:
		data = epo.get_data()
	metadata = epo.metadata

	data = np.delete(data, bad_epo, axis=0)
	try:
		metadata = metadata.reset_index().drop(bad_epo, axis=0)
	except:
		set_trace()

	new_epo = mne.EpochsArray(data, epo.info, metadata=metadata, tmin=epo.times[0])

	return new_epo 