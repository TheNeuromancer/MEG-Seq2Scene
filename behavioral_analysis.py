#!/usr/bin/env python
from expyriment.misc import data_preprocessing, constants
from ipdb import set_trace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import statsmodels
from glob import glob
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.api as sm

from utils.plot_utils import *
from utils.decod import make_sns_barplot


# strategies = {"01": 'None', "02": 'None', "03": 'None', "04": 'None', "05": 'None', "06": 'None', "07": 'None', 
# 			  "08": 'None', "09": 'sequential', "10": 'visual', "11": 'visual', "12": 'visual', "13": '\"too easy\"', 
# 			  "14": 'repeat-sent-then-img2sent', "15": 'both', "16": 'sequential', 
# 			  "17": 'None', "18": 'None', "19": 'None', "20": 'None', "21": 'None', "22": 'None', "26": 'None',  "27": 'None',# missing, look for the cahier de labo pictures
# 			  "23": 'sequential', "24": 'both', "25": 'visual', "28": 'sequential_backward', "29": 'seq-shape-then-colors', "30": 'sequential'}

strategies = {"01": 'None', "02": 'None', "03": 'None', "04": 'None', "05": 'None', "06": 'None', "07": 'None', 
			  "08": 'None', "09": 'sequential', "10": 'visual', "11": 'visual', "12": 'visual', "13": 'sequential', 
			  "14": 'sequential', "15": 'sequential', "16": 'sequential', 
			  "17": 'None', "18": 'None', "19": 'None', "20": 'None', "21": 'None', "22": 'None', "26": 'None',  "27": 'None',# missing, look for the cahier de labo pictures
			  "23": 'sequential', "24": 'sequential', "25": 'visual', "28": 'sequential', "29": 'sequential', "30": 'sequential'}

out_dir = Path("../s2s_results/behavior")
behavior_path = Path("../s2s_sub_stimuli/")
all_subs = [s.name for s in behavior_path.glob("*")]
print(f"Found {len(all_subs)} subjects")
print(all_subs)
for sub in all_subs:
	# print(f"Doing subject {sub}")
	all_fns = list((behavior_path / sub).glob("*_metadata.csv"))
	if not all_fns: continue
	all_md = []
	for fn in all_fns:
		all_md.append(pd.read_csv(fn, encoding='ISO-8859-1'))
	md = pd.concat(all_md)
	md["Subject"] = [sub[0:2] for i in range(len(md))]
	md["Strategy"] = [strategies[sub[0:2]] for i in range(len(md))]

	if all_subs.index(sub) == 0:
		all_subs_md = deepcopy(md)
	else:
		all_subs_md = pd.concat((all_subs_md, md))
# reset index
all_subs_md.reset_index(inplace=True, drop=True)
# remove non-entries
all_subs_md = all_subs_md.dropna()

# add intercept manually
# all_subs_md["Intercept"] = 1

# remove localizer
all_subs_md = all_subs_md.query("NbObjects!=0.75")

# put Perf as type float 
all_subs_md.Perf = all_subs_md.Perf.apply(float)
all_subs_md["Error rate"] = np.logical_not(all_subs_md["Perf"]).astype(float)

all_subs_md["Trial type"] = all_subs_md["NbObjects"].apply(lambda x: "Short" if x==1.0 else "Long")
print(all_subs_md["Trial type"])


# all_subs_md.to_csv("~/tmp.csv")

## reject outliers
for sub in all_subs_md.Subject.unique():
	std = all_subs_md.query(f"Subject=='{sub}'").RT.std()
	all_subs_md["RT"] = np.where((all_subs_md.RT>(3*std)) & (all_subs_md.Subject==sub), np.nan, all_subs_md["RT"])

results_1obj = deepcopy(all_subs_md.query("NbObjects==1"))
results_2obj = deepcopy(all_subs_md.query("NbObjects==2"))

print("Keeping only correctly answered trials for RT plots")
all_subs_md_correct = all_subs_md.query("Perf==1")
results_1obj_correct = deepcopy(all_subs_md_correct.query("NbObjects==1"))
results_2obj_correct = deepcopy(all_subs_md_correct.query("NbObjects==2"))



##################
#### OVERALL #####
##################

## split by subjects
box_pairs = []
ymin, ymax = None, None
make_sns_barplot(all_subs_md, x='Trial type', y='Error rate', hue='Subject', legend=False, box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_overall_errorrate.png', tight=True)
make_sns_barplot(all_subs_md, x='Trial type', y='Error rate', hue='Subject', legend=False, dodge=False, kind='point', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_overall_errorrate_point.png', tight=True)
make_sns_barplot(all_subs_md, x='Trial type', y='Error rate', hue='Subject', legend=False, dodge=True, kind='point', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_overall_errorrate_point_dodge.png', tight=True)
make_sns_barplot(all_subs_md, x='Trial type', y='Error rate', hue='Subject', legend=False, kind='box', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_overall_errorrate_box.png', tight=True)

make_sns_barplot(all_subs_md, x='Trial type', y='RT', hue='Subject', legend=False, box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_overall_RT.png', tight=True)
make_sns_barplot(all_subs_md, x='Trial type', y='RT', hue='Subject', legend=False, dodge=False, kind='point', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_overall_RT_point.png', tight=True)
make_sns_barplot(all_subs_md, x='Trial type', y='RT', hue='Subject', legend=False, dodge=True, kind='point', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_overall_RT_point_dodge.png', tight=True)
make_sns_barplot(all_subs_md, x='Trial type', y='RT', hue='Subject', legend=False, kind='box', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_overall_RT_box.png', tight=True)

exit()

##################
###### EASY ######
##################


# # ### COLOUR ###

box_pairs = [("vert", "rouge"), ("rouge", "bleu"), ("vert", "bleu")]
# ymin, ymax = results_1obj.Error rate.mean()-.05, results_1obj.Error rate.mean()+.05
make_sns_barplot(results_1obj, x='Colour1', y='Error rate', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_color_errorrate.png', tight=True)
make_sns_barplot(results_1obj, x='Colour1', y='Error rate', kind='point', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_color_errorrate_point.png', tight=True)
make_sns_barplot(results_1obj, x='Colour1', y='Error rate', kind='box', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_color_errorrate_box.png', tight=True)

# ymin, ymax = results_1obj.RT.mean()-100, results_1obj.RT.mean()+100
ymin, ymax = None, None
make_sns_barplot(results_1obj_correct, x='Colour1', y='RT', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_color_RT.png', tight=True)
make_sns_barplot(results_1obj_correct, x='Colour1', y='RT', kind='point', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_color_RT_point.png', tight=True)
make_sns_barplot(results_1obj_correct, x='Colour1', y='RT', kind='box', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_color_RT_box.png', tight=True)


### SHAPE ###
box_pairs = [("carre", "cercle"), ("cercle", "triangle"), ("carre", "triangle")]
# ymin, ymax = results_1obj.Error rate.mean()-.05, results_1obj.Error rate.mean()+.05
make_sns_barplot(results_1obj, x='Shape1', y='Error rate', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_shape_errorrate.png', tight=True)
make_sns_barplot(results_1obj, x='Shape1', y='Error rate', kind='point', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_shape_errorrate_point.png', tight=True)
make_sns_barplot(results_1obj, x='Shape1', y='Error rate', kind='box', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_shape_errorrate_box.png', tight=True)

# ymin, ymax = results_1obj.RT.mean()-100, results_1obj.RT.mean()+100
make_sns_barplot(results_1obj_correct, x='Shape1', y='RT', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_shape_RT.png', tight=True)
make_sns_barplot(results_1obj_correct, x='Shape1', y='RT', kind='point', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_shape_RT_point.png', tight=True)
make_sns_barplot(results_1obj_correct, x='Shape1', y='RT', kind='box', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_shape_RT_box.png', tight=True)

### MAPPING ###
box_pairs = [("correct_right", "correct_left")]
# ymin, ymax = results_1obj.Error rate.mean()-.05, results_1obj.Error rate.mean()+.05
make_sns_barplot(results_1obj, x='Mapping', y='Error rate', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_mapping_errorrate.png', tight=True)
make_sns_barplot(results_1obj, x='Mapping', y='Error rate', kind='point', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_mapping_errorrate_point.png', tight=True)
make_sns_barplot(results_1obj, x='Mapping', y='Error rate', kind='box', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_mapping_errorrate_box.png', tight=True)

# ymin, ymax = results_1obj.RT.mean()-100, results_1obj.RT.mean()+100
make_sns_barplot(results_1obj_correct, x='Mapping', y='RT', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_mapping_RT.png', tight=True)
make_sns_barplot(results_1obj_correct, x='Mapping', y='RT', kind='point', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_mapping_RT_point.png', tight=True)
make_sns_barplot(results_1obj_correct, x='Mapping', y='RT', kind='box', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_mapping_RT_box.png', tight=True)

### MATCHING ###
box_pairs = [("match", "nonmatch")]
# ymin, ymax = results_1obj.Error rate.mean()-.05, results_1obj.Error rate.mean()+.05
make_sns_barplot(results_1obj, x='Matching', y='Error rate', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_matching_errorrate.png', tight=True)
make_sns_barplot(results_1obj, x='Matching', y='Error rate', kind='point', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_matching_errorrate_point.png', tight=True)
make_sns_barplot(results_1obj, x='Matching', y='Error rate', kind='box', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_matching_errorrate_box.png', tight=True)

# ymin, ymax = results_1obj.RT.mean()-100, results_1obj.RT.mean()+100
make_sns_barplot(results_1obj_correct, x='Matching', y='RT', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_matching_RT.png', tight=True)
make_sns_barplot(results_1obj_correct, x='Matching', y='RT', kind='point', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_matching_RT_point.png', tight=True)
make_sns_barplot(results_1obj_correct, x='Matching', y='RT', kind='box', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_1obj_matching_RT_box.png', tight=True)



# queries = ['Colour1 == "rouge"', 'Colour1 == "bleu"', 'Colour1 == "vert"']
# plot_histograms(results=results_1obj, queries=queries, colors=('r', 'b', 'g'), fn=f'{out_dir}/all_subs_easy_hist_Colour1.png')
# plot_means(results=results_1obj, grouping_query='Colour1', fn=f'{out_dir}/all_subs_easy_colours.png')

# ### SHAPE ###
# queries = ['Shape1 == "carre"', 'Shape1 == "cercle"', 'Shape1 == "triangle"']
# plot_histograms(results=results_1obj, queries=queries, colors=('orange', 'brown', 'olive'), fn=f'{out_dir}/all_subs_easy_hist_Shape1.png')

# plot_means(results=results_1obj, grouping_query='Shape1', fn=f'{out_dir}/all_subs_easy_shapes.png')

# ### MAPPING ###

# queries = ['Mapping == "correct_left"', 'Mapping == "correct_right"']
# plot_histograms(results=results_1obj, queries=queries, colors=('grey', 'pink'), fn=f'{out_dir}/all_subs_easy_hist_Mapping.png')

# plot_means(results=results_1obj, grouping_query='Mapping', fn=f'{out_dir}/all_subs_easy_mapping.png')

# ### MATCHING ###
# queries = ['Button == "left"', 'Button == "right"']
# plot_histograms(results=results_1obj, queries=queries, colors=('grey', 'pink'), fn=f'{out_dir}/all_subs_easy_hist_Button.png')

# plot_means(results=results_1obj, grouping_query='Matching', fn=f'{out_dir}/all_subs_easy_matching.png')



# # #################
# # ##### HARD ######
# # #################

if True:
	# ### COLOUR ###
	box_pairs = [("vert", "rouge"), ("rouge", "bleu"), ("vert", "bleu")]
	ymin, ymax = results_2obj["Error rate"].mean()-.05, results_2obj["Error rate"].mean()+.05
	make_sns_barplot(results_2obj, x='Colour1', y='Error rate', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_color1_errorrate.png', tight=True)

	ymin, ymax = results_2obj.RT.mean()-100, results_2obj.RT.mean()+100
	make_sns_barplot(results_2obj_correct, x='Colour1', y='RT', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_color1_RT.png', tight=True)


	ymin, ymax = results_2obj["Error rate"].mean()-.05, results_2obj["Error rate"].mean()+.05
	make_sns_barplot(results_2obj, x='Colour2', y='Error rate', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_color2_errorrate.png', tight=True)

	ymin, ymax = results_2obj.RT.mean()-100, results_2obj.RT.mean()+100
	make_sns_barplot(results_2obj_correct, x='Colour2', y='RT', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_color2_RT.png', tight=True)

	### SHAPE ###
	box_pairs = [("carre", "cercle"), ("cercle", "triangle"), ("carre", "triangle")]
	ymin, ymax = results_2obj["Error rate"].mean()-.05, results_2obj["Error rate"].mean()+.05
	make_sns_barplot(results_2obj, x='Shape1', y='Error rate', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_shape1_errorrate.png', tight=True)

	ymin, ymax = results_2obj.RT.mean()-100, results_2obj.RT.mean()+100
	make_sns_barplot(results_2obj_correct, x='Shape1', y='RT', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_shape1_RT.png', tight=True)


	ymin, ymax = results_2obj["Error rate"].mean()-.05, results_2obj["Error rate"].mean()+.05
	make_sns_barplot(results_2obj, x='Shape2', y='Error rate', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_shape2_errorrate.png', tight=True)

	ymin, ymax = results_2obj.RT.mean()-100, results_2obj.RT.mean()+100
	make_sns_barplot(results_2obj_correct, x='Shape2', y='RT', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_shape2_RT.png', tight=True)

	### MAPPING ###
	box_pairs = [("correct_right", "correct_left")]
	ymin, ymax = results_2obj["Error rate"].mean()-.05, results_2obj["Error rate"].mean()+.05
	make_sns_barplot(results_2obj, x='Mapping', y='Error rate', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_mapping_errorrate.png', tight=True)

	ymin, ymax = results_2obj.RT.mean()-100, results_2obj.RT.mean()+100
	make_sns_barplot(results_2obj_correct, x='Mapping', y='RT', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_mapping_RT.png', tight=True)

	### MATCHING ###
	box_pairs = [("match", "nonmatch")]
	ymin, ymax = results_2obj["Error rate"].mean()-.05, results_2obj["Error rate"].mean()+.05
	make_sns_barplot(results_2obj, x='Matching', y='Error rate', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_matching_errorrate.png', tight=True)

	ymin, ymax = results_2obj.RT.mean()-100, results_2obj.RT.mean()+100
	make_sns_barplot(results_2obj_correct, x='Matching', y='RT', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_matching_RT.png', tight=True)


	### DIFFICULTY ###
	box_pairs = [("D0", "D1"), ("D1", "D2"), ("D0", "D2")]
	ymin, ymax = results_2obj["Error rate"].mean()-.05, results_2obj["Error rate"].mean()+.05
	make_sns_barplot(results_2obj, x='Difficulty', y='Error rate', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_bar_2obj_difficulty_errorrate.png', tight=True, order=["D0", "D1", "D2"])
	make_sns_barplot(results_2obj, x='Difficulty', y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_difficulty_errorrate.png', tight=True, order=["D0", "D1", "D2"])
	make_sns_barplot(results_2obj, x='Difficulty', y='Error rate', kind='violin', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_difficulty_errorrate.png', tight=True, order=["D0", "D1", "D2"])

	ymin, ymax = results_2obj.RT.mean()-400, results_2obj.RT.mean()+200
	make_sns_barplot(results_2obj_correct, x='Difficulty', y='RT', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_bar_2obj_difficulty_RT.png', tight=True, order=["D0", "D1", "D2"])
	make_sns_barplot(results_2obj_correct, x='Difficulty', y='RT', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_difficulty_RT.png', tight=True, order=["D0", "D1", "D2"])
	make_sns_barplot(results_2obj_correct, x='Difficulty', y='RT', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_difficulty_RT.png', tight=True, order=["D0", "D1", "D2"])


	### ERROR_TYPE ###
	# set_trace()
	box_pairs = [("None", "l0"), ("l0", "l1"), ("None", "l1"), ("l1", "l2"), ("l0", "l2")]
	ymin, ymax = results_2obj["Error rate"].mean()-.1, results_2obj["Error rate"].mean()+.05
	make_sns_barplot(results_2obj, x='Error_type', y='Error rate', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_bar_2obj_Error_type_errorrate.png', tight=True, order=["None", "l0", "l1", "l2"])
	make_sns_barplot(results_2obj, x='Error_type', y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Error_type_errorrate.png', tight=True, order=["None", "l0", "l1", "l2"])
	make_sns_barplot(results_2obj, x='Error_type', y='Error rate', kind='violin', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Error_type_errorrate.png', tight=True, order=["None", "l0", "l1", "l2"])

	ymin, ymax = results_2obj.RT.mean()-300, results_2obj.RT.mean()+300
	make_sns_barplot(results_2obj_correct, x='Error_type', y='RT', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_bar_2obj_Error_type_RT.png', tight=True, order=["None", "l0", "l1", "l2"])
	make_sns_barplot(results_2obj_correct, x='Error_type', y='RT', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Error_type_RT.png', tight=True, order=["None", "l0", "l1", "l2"])
	make_sns_barplot(results_2obj_correct, x='Error_type', y='RT', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Error_type_RT.png', tight=True, order=["None", "l0", "l1", "l2"])




	## With interaction
	### DIFFICULTY ###
	box_pairs = [(("D0", "None"), ("D0", "l0")), 
				 (("D1", "None"), ("D1", "l0")),(("D1", "l0"), ("D1", "l2")),
				 (("D2", "None"), ("D2", "l0")),(("D2", "l0"), ("D2", "l1")),(("D2", "l1"), ("D2", "l2")),(("D2", "l0"), ("D2", "l2"))] 
	ymin, ymax = results_2obj["Error rate"].mean()-.1, results_2obj["Error rate"].mean()+.05
	make_sns_barplot(results_2obj, x='Difficulty', y='Error rate', hue='Error_type', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_bar_2obj_difficulty_error_type_errorrate.png', tight=True, order=["D0", "D1", "D2"], hue_order=["None", "l0", "l1", "l2"])
	make_sns_barplot(results_2obj, x='Difficulty', y='Error rate', hue='Error_type', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_difficulty_error_type_errorrate1.png', tight=True, order=["D0", "D1", "D2"], hue_order=["None", "l0", "l1", "l2"])
	make_sns_barplot(results_2obj, x='Difficulty', y='Error rate', hue='Error_type', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_difficulty_error_type_errorrate1.png', tight=True, order=["D0", "D1", "D2"], hue_order=["None", "l0", "l1", "l2"])

	ymin, ymax = results_2obj.RT.mean()-400, results_2obj.RT.mean()+200
	make_sns_barplot(results_2obj_correct, x='Difficulty', y='RT', hue='Error_type', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_bar_2obj_difficulty_error_type_RT.png', tight=True, order=["D0", "D1", "D2"], hue_order=["None", "l0", "l1", "l2"])
	make_sns_barplot(results_2obj_correct, x='Difficulty', y='RT', hue='Error_type', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_difficulty_error_type_RT.png', tight=True, order=["D0", "D1", "D2"], hue_order=["None", "l0", "l1", "l2"])
	make_sns_barplot(results_2obj_correct, x='Difficulty', y='RT', hue='Error_type', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_difficulty_error_type_RT.png', tight=True, order=["D0", "D1", "D2"], hue_order=["None", "l0", "l1", "l2"])


	### ERROR_TYPE ###
	box_pairs = [(("None", "D0"), ("None", "D1")), (("None", "D0"), ("None", "D2")), (("None", "D1"), ("None", "D2")),
				 (("l0", "D0"), ("l0", "D1")), (("l0", "D0"), ("l0", "D2")), (("l0", "D1"), ("l0", "D2")),
				 (("l2", "D1"), ("l2", "D2"))]
	ymin, ymax = results_2obj["Error rate"].mean()-.1, results_2obj["Error rate"].mean()+.05
	make_sns_barplot(results_2obj, x='Error_type', y='Error rate', hue='Difficulty', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_bar_2obj_Error_type_difficulty_errorrate.png', tight=True, order=["None", "l0", "l1", "l2"], hue_order=["D0", "D1", "D2"])
	make_sns_barplot(results_2obj, x='Error_type', y='Error rate', hue='Difficulty', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Error_type_difficulty_errorrate.png', tight=True, order=["None", "l0", "l1", "l2"], hue_order=["D0", "D1", "D2"])
	make_sns_barplot(results_2obj, x='Error_type', y='Error rate', hue='Difficulty', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Error_type_difficulty_errorrate.png', tight=True, order=["None", "l0", "l1", "l2"], hue_order=["D0", "D1", "D2"])

	ymin, ymax = results_2obj.RT.mean()-600, results_2obj.RT.mean()+300
	make_sns_barplot(results_2obj_correct, x='Error_type', y='RT', hue='Difficulty', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_bar_2obj_Error_type_difficulty_RT.png', tight=True, order=["None", "l0", "l1", "l2"], hue_order=["D0", "D1", "D2"])
	make_sns_barplot(results_2obj_correct, x='Error_type', y='RT', hue='Difficulty', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Error_type_difficulty_RT.png', tight=True, order=["None", "l0", "l1", "l2"], hue_order=["D0", "D1", "D2"])
	make_sns_barplot(results_2obj_correct, x='Error_type', y='RT', hue='Difficulty', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Error_type_difficulty_RT.png', tight=True, order=["None", "l0", "l1", "l2"], hue_order=["D0", "D1", "D2"])


# ### MAPPING ###
# queries = ['Mapping == "correct_left"', 'Mapping == "correct_right"']
# plot_histograms(results=results_2obj, queries=queries, colors=('grey', 'pink'), fn=f'{out_dir}/all_subs_hard_hist_Mapping.png')
# plot_means(results=results_2obj, grouping_query='Mapping', fn=f'{out_dir}/all_subs_hard_Mapping.png')

# ### MATCHING ###
# queries = ['Button == "left"', 'Button == "right"']
# plot_histograms(results=results_2obj, queries=queries, colors=('grey', 'pink'), fn=f'{out_dir}/all_subs_hard_hist_Matching.png')
# plot_means(results=results_2obj, grouping_query='Matching', fn=f'{out_dir}/all_subs_hard_Matching.png')


# ### INTESRESTING STUFF ###

# ### DIFFICULTY (NB OF SHARED FEATURES) ### (No error trials only)
# local_results_2obj = deepcopy(results_2obj)
# local_results_2obj = local_results_2obj.query("Error_type=='None'")
# queries = ['Difficulty == "D0"', 'Difficulty == "D1"', 'Difficulty == "D2"']
# plot_histograms(results=local_results_2obj, queries=queries, colors=('coral', 'chartreuse', 'deepskyblue'), fn=f'{out_dir}/all_subs_hard_hist_Difficulty.png')
# plot_means(results=local_results_2obj, grouping_query='Difficulty', fn=f'{out_dir}/all_subs_hard_Difficulty.png')


# ### ERROR_TYPE ###
# queries = ['Error_type == "l0"', 'Error_type == "l1"', 'Error_type == "l2"']
# plot_histograms(results=results_2obj, queries=queries, colors=('coral', 'chartreuse', 'deepskyblue'), fn=f'{out_dir}/all_subs_hard_hist_Error_type.png')
# plot_means(results=results_2obj, grouping_query='Error_type', fn=f'{out_dir}/all_subs_hard_Error_type.png')


# ### VIOLATION POSITION ###
# Violated_position = 3 <-> L2 (relation inversion) error 
# Violated_position in [1,2,4,5] <-> L0 (dumb change) error
# Violated_position in [1.5, 4.5] <-> L1 (binding) error

if True:
	# Just position
	local_results_2obj = deepcopy(results_2obj).query('Error_type=="l0"')
	local_results_2obj_correct = deepcopy(results_2obj_correct).query('Error_type=="l0"')
	box_pairs = [(1.0, 2.0), (1.0, 4.0), (1.0, 5.0)] #, (2.0, 4.0), (2.0, 5.0), (4.0, 5.0)]
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue=None, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position_RT.png', tight=True)
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position_RT.png', tight=True)
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue=None, kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position_RT.png', tight=True)

	# position*relation
	box_pairs = []
	box_pairs = [((1.0, "à gauche d'"), (1.0, "à droite d'")), ((1.0, "à gauche d'"), (2.0, "à gauche d'")), ((1.0, "à gauche d'"), (4.0, "à gauche d'"))]
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Relation_RT.png', tight=True)
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Relation_RT.png', tight=True)
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Relation_RT.png', tight=True)

	# position*relation*strategy
	# box_pairs = [((1.0, 'sequential'), (4.0, 'sequential')), ((1.0, 'visual'), (4.0, 'visual'))]
	box_pairs = [((1.0, 'sequential'), (1.0, 'visual')), ((2.0, 'sequential'), (2.0, 'visual')),
				 ((4.0, 'sequential'), (4.0, 'visual')), ((5.0, 'sequential'), (5.0, 'visual'))]
	# box_pairs = []
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Strategy', col='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Relation*Strategy_RT.png', tight=True)
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Strategy', col='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Relation*Strategy_RT.png', tight=True)
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Strategy', col='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Relation*Strategy_RT.png', tight=True)

	# position*strategy
	# box_pairs = []
	box_pairs = [((1.0, 'sequential'), (1.0, 'visual')), ((2.0, 'sequential'), (2.0, 'visual')),
				 ((4.0, 'sequential'), (4.0, 'visual')), ((5.0, 'sequential'), (5.0, 'visual'))]
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Strategy', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Strategy_RT.png', tight=True)
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Strategy', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Strategy_RT.png', tight=True)
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Strategy', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Strategy_RT.png', tight=True)


	# split by difficulty
	box_pairs = []
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Difficulty', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Difficulty_RT.png', tight=True)
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Difficulty', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Difficulty_RT.png', tight=True)
	make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Difficulty', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Difficulty_RT.png', tight=True)



	### SAME WITH PERF
	box_pairs = [(1.0, 2.0), (1.0, 4.0), (1.0, 5.0)] #, (2.0, 4.0), (2.0, 5.0), (4.0, 5.0)]
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position_error_rate.png', tight=True)
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position_error_rate.png', tight=True)
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position_error_rate.png', tight=True)

	# position*relation
	box_pairs = [((1.0, "à gauche d'"), (1.0, "à droite d'")), ((1.0, "à gauche d'"), (2.0, "à gauche d'")), ((1.0, "à gauche d'"), (4.0, "à gauche d'"))]
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Relation_error_rate.png', tight=True)
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Relation_error_rate.png', tight=True)
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Relation_error_rate.png', tight=True)

	# position*relation*strategy
	# box_pairs = [((1.0, 'sequential'), (4.0, 'sequential')), ((1.0, 'visual'), (4.0, 'visual'))]
	box_pairs = [((1.0, 'sequential'), (1.0, 'visual')), ((2.0, 'sequential'), (2.0, 'visual')),
				 ((4.0, 'sequential'), (4.0, 'visual')), ((5.0, 'sequential'), (5.0, 'visual'))]
	# box_pairs = []
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', col='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Relation*Strategy_error_rate.png', tight=True)
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', col='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Relation*Strategy_error_rate.png', tight=True)
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', col='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Relation*Strategy_error_rate.png', tight=True)

	# position*strategy
	# box_pairs = []
	box_pairs = [((1.0, 'sequential'), (1.0, 'visual')), ((2.0, 'sequential'), (2.0, 'visual')),
				 ((4.0, 'sequential'), (4.0, 'visual')), ((5.0, 'sequential'), (5.0, 'visual'))]
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Strategy_error_rate.png', tight=True)
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Strategy_error_rate.png', tight=True)
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Strategy_error_rate.png', tight=True)


	# split by difficulty
	box_pairs = []
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Difficulty', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Difficulty_error_rate.png', tight=True)
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Difficulty', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Difficulty_error_rate.png', tight=True)
	make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Difficulty', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Difficulty_error_rate.png', tight=True)




## L1 trials
local_results_2obj = deepcopy(results_2obj).query('Error_type=="l1"')
local_results_2obj_correct = deepcopy(results_2obj_correct).query('Error_type=="l1"')
box_pairs = [(1.5, 4.5)]
make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue=None, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position_RT.png', tight=True)
make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position_RT.png', tight=True)
make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue=None, kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position_RT.png', tight=True)

# position*relation
box_pairs = []
box_pairs = [((1.5, "à gauche d'"), (1.5, "à droite d'")), ((4.5, "à gauche d'"), (4.5, "à gauche d'"))] #, ((1.0, "à gauche d'"), (4.0, "à gauche d'"))]
make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position*Relation_RT.png', tight=True)
make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position*Relation_RT.png', tight=True)
make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position*Relation_RT.png', tight=True)

# position*strategy
# box_pairs = []
box_pairs = [((1.5, 'sequential'), (1.5, 'visual')), ((4.5, 'sequential'), (4.5, 'visual')),
			 ((1.5, 'sequential'), (4.5, 'sequential')), ((1.5, 'visual'), (4.5, 'visual'))]
make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Strategy', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position*Strategy_RT.png', tight=True)
make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Strategy', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position*Strategy_RT.png', tight=True)
make_sns_barplot(local_results_2obj_correct, x='Violated_position', y='RT', hue='Strategy', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position*Strategy_RT.png', tight=True)


## L1 trials -- Error rate
local_results_2obj = deepcopy(results_2obj).query('Error_type=="l1"')
box_pairs = [(1.5, 4.5)]
make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position_error_rate.png', tight=True)
make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position_error_rate.png', tight=True)
make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position_error_rate.png', tight=True)

# position*relation
box_pairs = []
box_pairs = [((1.5, "à gauche d'"), (1.5, "à droite d'")), ((4.5, "à gauche d'"), (4.5, "à gauche d'"))] #, ((1.0, "à gauche d'"), (4.0, "à gauche d'"))]
make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position*Relation_error_rate.png', tight=True)
make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position*Relation_error_rate.png', tight=True)
make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position*Relation_error_rate.png', tight=True)

# position*strategy
# box_pairs = []
box_pairs = [((1.5, 'sequential'), (1.5, 'visual')), ((4.5, 'sequential'), (4.5, 'visual')),
			 ((1.5, 'sequential'), (4.5, 'sequential')), ((1.5, 'visual'), (4.5, 'visual'))]
make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position*Strategy_error_rate.png', tight=True)
make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position*Strategy_error_rate.png', tight=True)
make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position*Strategy_error_rate.png', tight=True)



# queries = ['Violated_position == "1"', 'Violated_position == "2"', 'Violated_position == "3"', 'Violated_position == "4"', 'Violated_position == "5"']
# plot_histograms(results=local_results_2obj, queries=queries, colors=('coral', 'chartreuse', 'deepskyblue', 'crimson', 'navy'), fn=f'{out_dir}/all_subs_hard_hist_Violated_position.png')
# plot_means(results=local_results_2obj, grouping_query='Violated_position', fn=f'{out_dir}/all_subs_hard_Violated_position.png')


# ### RELATION ###
# queries = ['Relation in ["à gauche d\'", "à droite d\'"]', 'Relation in ["dans", "autour de"]']
# plot_histograms(results=results_2obj, queries=queries, colors=('coral', 'deepskyblue'), fn=f'{out_dir}/all_subs_hard_hist_Relation.png')
# plot_means(results=results_2obj, grouping_query='Relation', fn=f'{out_dir}/all_subs_hard_Relation.png')


# # ### SUBJECT_ID ###
# # queries = ['subject_id==1', 'subject_id==2']
# # plot_histograms(results=results_2obj, queries=queries, colors=('coral', 'deepskyblue'), fn=f'{out_dir}/all_subs_hard_hist_subject_id.png')
# # plot_means(results=results_2obj, grouping_query='subject_id', fn=f'{out_dir}/all_subs_hard_subject_id.png')


### INTERACTIONS ###	
# plot_means(results=results_2obj, grouping_query=['Difficulty', 'Error_type'], fn=f'{out_dir}/all_subs_hard_Difficulty*Error_type.png')

# plot_interaction(results=results_2obj, factor1='Error_type', factor2='Difficulty', dep_var='RT', colors=['red', 'blue', 'green'], 
# 									markers=['D', '^', '*'], fn=f'{out_dir}/all_subs_hard_Error_type*Difficulty_RT.png')

# plot_interaction(results=results_2obj, factor1='Error_type', factor2='Difficulty', dep_var='Perf', colors=['red', 'blue', 'green'], 
# 									markers=['D', '^', '*'], fn=f'{out_dir}/all_subs_hard_Error_type*Difficulty_Perf.png')


# plot_interaction(results=results_2obj.query('Error_type=="l0"'), factor1='Violated_position', factor2='Difficulty', dep_var='RT', 
# 		colors=['red', 'blue', 'green'], markers=['D', '^', '*'], fn=f'{out_dir}/all_subs_hard_Violated_position*Difficulty_RT.png')

# plot_interaction(results=results_2obj.query('Error_type=="l0"'), factor1='Violated_position', factor2='Difficulty', dep_var='Perf', 
# 		colors=['red', 'blue', 'green'], markers=['D', '^', '*'], fn=f'{out_dir}/all_subs_hard_Violated_position*Difficulty_Perf.png')


# ### VIOLATED OBJECT POSITION ###
# local_results_2obj = deepcopy(results_2obj).query('Error_type=="l0"')
# violated_positions = local_results_2obj['Violated_position'].to_numpy().astype(int)
# relations = local_results_2obj['Relation'].to_numpy()
# print(violated_positions)
# print(relations)
# local_results_2obj['Violated_object'] = get_violated_objects(violated_positions, relations)
# queries = ['Violated_object=="droite"', 'Violated_object=="gauche"', 'Violated_object=="dans"', 'Violated_object=="autour"']
# plot_histograms(results=local_results_2obj, queries=queries, colors=('coral', 'deepskyblue', 'olive', 'brown'), fn=f'{out_dir}/all_subs_hard_hist_violated_object.png')
# plot_means(results=local_results_2obj, grouping_query='Violated_object', fn=f'{out_dir}/all_subs_hard_violated_object.png')


# # two_way_anova(results_2obj, ['Difficulty', 'Error_type'], typ=3)



## stats
# # packnames = ('lme4', 'lmerTest', 'emmeans', "geepack")
# # from rpy2.robjects.packages import importr
# # from rpy2.robjects.vectors import StrVector
# # utils = importr("utils")
# # utils.chooseCRANmirror(ind=1)
# # utils.install_packages(StrVector(packnames))


# # # #Import necessary packages
# # # from rpy2.robjects.packages import importr
# # import rpy2.robjects as robjects
# # from rpy2.robjects import pandas2ri
# # #Must be activated
# # pandas2ri.activate()
# # from rpy2.robjects import FloatVector
# # from rpy2.robjects.packages import importr

# # stats = importr('stats')
# # base = importr('base')
# # lme4 = importr('lme4')

# # # ctl = FloatVector([4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14])
# # # trt = FloatVector([4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69])
# # # group = base.gl(2, 10, 20, labels = ['Ctl','Trt'])
# # # weight = ctl + trt
# # # robjects.globalenv['Colour1'] = results_1obj.Colour1
# # # robjects.globalenv['Shape1'] = results_1obj.Shape1

# # results_1obj["Matching"][results_1obj["Matching"]=="match"] = 1
# # results_1obj["Matching"][results_1obj["Matching"]=="nonmatch"] = -1
# # results_1obj["Mapping"][results_1obj["Mapping"]=="correct_left"] = 1
# # results_1obj["Mapping"][results_1obj["Mapping"]=="correct_right"] = -1
# # robjects.globalenv['Matching'] = results_1obj.Matching
# # robjects.globalenv['Colour1'] = results_1obj.Colour1
# # robjects.globalenv['Shape1'] = results_1obj.Shape1

# # results_1obj["red"] = pd.get_dummies(results_1obj.Colour1).iloc[:,0]
# # results_1obj["blue"] = pd.get_dummies(results_1obj.Colour1).iloc[:,1]
# # results_1obj["green"] = pd.get_dummies(results_1obj.Colour1).iloc[:,2]
# # robjects.globalenv['red'] = pd.get_dummies(results_1obj.Colour1).iloc[:,0]
# # robjects.globalenv['green'] = results_1obj.blue
# # robjects.globalenv['blue'] = results_1obj.green
# # # robjects.globalenv['Matching'] = (results_1obj.Matching == "nonmatch").values.astype(int)
# # robjects.globalenv['RT'] = results_1obj.RT
# # robjects.globalenv['Subject'] = results_1obj.Subject
# # lm = lme4.lmer('RT ~ Matching + red + green + blue + Shape1 + (1|Subject)')
# # print(base.summary(lm))



# # results_1obj["red"] = pd.get_dummies(results_1obj.Colour1).iloc[:,0]
# # results_1obj["blue"] = pd.get_dummies(results_1obj.Colour1).iloc[:,1]
# # results_1obj["green"] = pd.get_dummies(results_1obj.Colour1).iloc[:,2]
# # res = smf.mixedlm("RT ~ 1 + red + blue + green", results_1obj, groups=results_1obj["Subject"]).fit(method="bfgs")
# # print(res.summary())


# # results_1obj["Matching"][results_1obj["Matching"]=="match"] = .5
# # results_1obj["Matching"][results_1obj["Matching"]=="nonmatch"] = -.5
# # results_1obj["Mapping"][results_1obj["Mapping"]=="correct_left"] = .5
# # results_1obj["Mapping"][results_1obj["Mapping"]=="correct_right"] = -.5

# ## Stats single object
# smf.mixedlm("RT ~ Matching + Mapping + Colour1 + Shape1", results_1obj, groups=results_1obj["Subject"]).fit(method="bfgs").summary()
# smf.mixedlm("Perf ~ Matching + Mapping + Colour1 + Shape1", results_1obj, groups=results_1obj["Subject"]).fit(method="bfgs").summary()

# ## Stats two objects
# # smf.mixedlm("RT ~ Matching + Mapping + Colour1 + Shape1 + Colour2 + Shape2 + Relation", results_2obj, groups=results_2obj["Subject"]).fit(method="bfgs").summary()
# smf.mixedlm("RT ~ Matching + Mapping + Colour1 + Shape1 + Colour2 + Shape2 + Relation + Difficulty", results_2obj, groups=results_2obj["Subject"]).fit(method="bfgs").summary()
# smf.mixedlm("Perf ~ Matching + Mapping + Colour1 + Shape1 + Colour2 + Shape2 + Relation + Difficulty", results_2obj, groups=results_2obj["Subject"]).fit(method="bfgs").summary()


# smf.mixedlm("RT ~ ordered(Difficulty)  ", results_2obj, groups=results_2obj["Subject"]).fit(method="bfgs").summary()
# set_trace()

# # Error types
# # violated_results_2obj = results_2obj[results_2obj["Error_type"] != "None"]
# # # violated_results_2obj["L0"] = pd.get_dummies(violated_results_2obj.Error_type).iloc[:,0]
# # # violated_results_2obj["L1"] = pd.get_dummies(violated_results_2obj.Error_type).iloc[:,1]
# # # violated_results_2obj["L2"] = pd.get_dummies(violated_results_2obj.Error_type).iloc[:,2]
# # smf.mixedlm("RT ~ Difficulty + Error_type ", violated_results_2obj, groups=violated_results_2obj["Subject"]).fit(method="bfgs").summary()

# # Singular matrx error
# # smf.mixedlm("RT ~ Difficulty + Error_type + Difficulty*Error_type ", violated_results_2obj, groups=violated_results_2obj["Subject"]).fit(method="bfgs").summary()
# # set_trace()

