from glob import glob
import os

fns = glob("/home/users/d/desborde/scratch/s2s/Results/Decoding_ovr_v10/Epochs_100hz_nofilter/*/Img2WordCat_0-Loc_Cat_word_cond-localizer-Perf=1_tested_on_localizer_Loc_Cat_image_AUC.npy")
# Img2WordCat_1-Loc_Cat_word_cond-localizer-Perf=1_tested_on_localizer_Loc_Cat_image_AUC.npy
	# Img2WordCat-Loc_Cat_word_cond-localizer-Perf=1_AUC.npy")
print(fns)
for fn in fns:
	new_fn = fn.replace("Img2WordCat", "Word2ImgCat")
	os.rename(fn, new_fn)
	print(new_fn)