import mne
import pickle

raw = mne.io.Raw("~/scratch/s2s/Data/orig/19/run4_1obj.fif", preload=True, verbose='error', allow_maxshield=True)
raw = raw.pick_types(meg=True) # remove misc channels

mag_int = 3022 # mag has this number in the info
grad_into = 3012
mag_idx = []
grad_idx = []

for i, ch in enumerate(raw.info['chs']):
	if ch['coil_type'] == mag_int:
		mag_idx.append(i)
	elif "MISC" in ch['ch_name']:
		pass
	else:
		grad_idx.append(i)
all_idx = sorted(mag_idx + grad_idx)

raw_mag = raw.copy().pick_types(meg='mag')
raw_grad = raw.copy().pick_types(meg='grad')

out_path = "/home/users/d/desborde/scratch/s2s/Data/"
pickle.dump(mag_idx, open(f"{out_path}/mag_indices.p", "wb"))
pickle.dump(grad_idx, open(f"{out_path}/grad_indices.p", "wb"))
pickle.dump(all_idx, open(f"{out_path}/all_indices.p", "wb"))
pickle.dump(raw_mag.info, open(f"{out_path}/mag_info.p", "wb"))
pickle.dump(raw_grad.info, open(f"{out_path}/grad_info.p", "wb"))
pickle.dump(raw.info, open(f"{out_path}/all_info.p", "wb"))