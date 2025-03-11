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

# from utils.plot_utils import *
from utils.behavior import *


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

lang = "en"

out_dir = Path("../s2s_results/behavior")
behavior_path = Path("../s2s_sub_stimuli/")
prepared_data_path = behavior_path / "prepared.csv"

## Load model results
model_data_path = Path("../CLIP-analyze/results/behavior/hug_v3/")
# model_data_fn = "openai_clip-vit-base-patch32_original_meg_images"
# model_data_fn = "clip_vit_base_patch32_original_meg_images"
model_data_fn = "openai-RN50x16_original_meg_images"
# model_data_fn = "openai-RN50x4_M-BERT-Base-69_original_meg_images"
model_results_1obj = pd.read_csv(model_data_path / f"{model_data_fn}_object-1obj.csv")
# model_results_2obj = pd.read_csv(model_data_path / f"{model_data_fn}_scene-2obj-{lang}.csv")
model_results_2obj = pd.read_csv(model_data_path / f"{model_data_fn}_scene-2obj.csv")
model_results_2obj["Violation on"] = model_results_2obj["Violation on"].apply(lambda x: x.capitalize())
model_results_2obj["Violation on detailed"] = model_results_2obj.apply(lambda x: x["Violation on"] if x["Violation on"]!='Property' else x["Changed property"], axis=1)
# model_results_1obj["Changed property"] = model_results_1obj["Change"].apply(lambda x: "Shape" if 'l0_shape' in x else  "Color")
# model_results_2obj["Changed property"] = model_results_2obj["Change"].apply(lambda x: "Color" if "l0_colour" in x else "Shape" if "l0_shape" in x else "Other")
model_results = pd.concat((model_results_1obj, model_results_2obj))
model_results.reset_index(inplace=True, drop=True)
model_results["Trial type"] = model_results["NbObjects"].apply(lambda x: "One object" if x==1.0 else "Two objects")

recompute = False

if prepared_data_path.exists() and not recompute:
	all_subs_md = pd.read_csv(prepared_data_path)
	results_1obj = deepcopy(all_subs_md.query("NbObjects==1"))
	results_2obj = deepcopy(all_subs_md.query("NbObjects==2"))
else:
	all_subs = [s.name for s in behavior_path.glob("*")]
	print(f"Found {len(all_subs)} subjects")
	print(all_subs)
	if "prepared.csv" in all_subs:
		all_subs.remove("prepared.csv")
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
	print("loaded")

	# reset index
	all_subs_md.reset_index(inplace=True, drop=True)
	# remove non-entries
	all_subs_md = all_subs_md.dropna()
	# remove localizer
	all_subs_md = all_subs_md.query("NbObjects!=0.75")
	# put Perf as type float 
	all_subs_md.Perf = all_subs_md.Perf.apply(float)
	all_subs_md["Error rate"] = np.logical_not(all_subs_md["Perf"]).astype(float)
	all_subs_md["'match' button"] = all_subs_md["Mapping"].apply(lambda x: "Right" if x=='correct_right' else "Left")
	all_subs_md["Trial type"] = all_subs_md["NbObjects"].apply(lambda x: "One object" if x==1.0 else "Two objects")
	# print(all_subs_md["Trial type"])
	all_subs_md["Violation"] = all_subs_md["Matching"].apply(lambda x: "No" if x=='match' else "Yes")
	all_subs_md["# Shared features"] = all_subs_md["Difficulty"].apply(lambda x: {"D0": 2, "D1": 1, "D2": 0, "L1": None}[x])
	# Colour1, Shape1, Colour2, Shape2 describe the image. To get the sentence, we need to use Change
	features = {"Shape1": [], "Shape2": [], "Colour2": [],  "Colour1": [], "Relation": []}
	for i, row in all_subs_md.iterrows():
		for key in features.keys(): # initialize with the image features
			features[key].append(row[key])
		if row.Violation == "Yes": # then modify depending on the violation
			if row.Error_type == "l0": # get original value 
				changed = row.Change.split('_')[-2].capitalize()
				orig_val = row.Change.split('_')[-1].split('2')[0]
				features[changed][-1] = orig_val
			elif row.Error_type == "l1": # exchange properties
				changed_type = row.Change.split('_')[2].capitalize()
				for i1, i2 in ((1,2), (2,1)):
					features[f"{changed_type}{i1}"][-1] = row[f"{changed_type}{i2}"]
			elif row.Error_type == "l2":  # inverse relation
				features["Relation"][-1] = "à droite d'" if features["Relation"] == "à gauche d'" else "à gauche d'"

	for key in features.keys(): # text features are lowercased
		all_subs_md[key.lower()] = features[key]

	# specify what is shared in D1 trials: color or shape
	shared_feats = []
	for i, row in all_subs_md.iterrows():
		if row.colour1 == row.colour2 and row.shape1 == row.shape2:
			shared_feats.append("Both")
		elif row.colour1 == row.colour2:
			shared_feats.append("Color")
		elif row.shape1 == row.shape2:
			shared_feats.append("Shape")
		else:
			shared_feats.append("None")
	all_subs_md["Sharing"] = shared_feats

	## reject outliers
	for sub in all_subs_md.Subject.unique():
		std = all_subs_md.query(f"Subject=='{sub}'").RT.std()
		all_subs_md["RT"] = np.where((all_subs_md.RT>(3*std)) & (all_subs_md.Subject==sub), np.nan, all_subs_md["RT"])

	# ## Correct RTs with single-subject average effects
	# mean_per_subject_1obj = results_1obj.groupby("Subject").mean().RT.to_dict()
	# results_1obj["mean RT per subject"] = results_1obj["Subject"].apply(lambda x: mean_per_subject_1obj[x])
	# results_1obj["corrected RT"] = results_1obj["RT"] - results_1obj["mean RT per subject"]
	# # results_1obj["RT"] = results_1obj["corrected RT"]

	# mean_per_subject_2obj = results_2obj.groupby("Subject").mean().RT.to_dict()
	# results_2obj["mean RT per subject"] = results_2obj["Subject"].apply(lambda x: mean_per_subject_2obj[x])
	# results_2obj["corrected RT"] = results_2obj["RT"] - results_2obj["mean RT per subject"]

	mean_per_subject = all_subs_md.groupby("Subject").mean().RT.to_dict()
	all_subs_md["mean RT per subject"] = all_subs_md["Subject"].apply(lambda x: mean_per_subject[x])
	all_subs_md["corrected RT"] = all_subs_md["RT"] - all_subs_md["mean RT per subject"]
	# all_subs_md["RT"] = all_subs_md["corrected RT"]

	# SPLIT TO MAKE SEPARATE MANIPULATIONS
	results_1obj = deepcopy(all_subs_md.query("NbObjects==1"))
	results_2obj = deepcopy(all_subs_md.query("NbObjects==2"))

	## check what was changed (color or shape)
	results_1obj["Changed property"] = results_1obj["Change"].apply(lambda x: "Shape" if 'shape' in x else  "Color")
	results_2obj["Changed property"] = results_2obj["Change"].apply(lambda x: "Color" if "l0_colour" in x else "Shape" if "l0_shape" in x else "Other")

	## Type of violation for 2obj trials
	results_2obj["Violation on"] = results_2obj["Error_type"].apply(lambda x: {"None": 'None', "l0": 'Property', "l1": 'Binding', "l2": 'Relation'}[x])
	results_2obj["Violation on detailed"] = results_2obj.apply(lambda x: {"None": 'None', "l1": 'Binding', "l2": 'Relation'}[x.Error_type] if x.Error_type!='l0' else "Color" if "colour" in x.Change else "Shape", axis=1)
	

	results_1obj["Violation ordinal position"] = ["None" for x in range(len(results_1obj))]
	results_2obj["Violation ordinal position"] = results_2obj["Violated_position"].apply(lambda x: "First" if x < 3 else "Second")

	# get the position in the image where the violation is 
	# specify what is shared in D1 trials: color or shape
	viol_side = []
	for i, row in results_2obj.iterrows():
		if row.Error_type != 'l0': 
			viol_side.append("None")
			continue
		if row.Violated_position < 3:
			if row.relation == "à droite d'":
				viol_side.append("Right")
			elif row.relation == "à gauche d'":
				viol_side.append("Left")
		elif row.Violated_position > 3:
			if row.relation == "à droite d'":
				viol_side.append("Left")
			elif row.relation == "à gauche d'":
				viol_side.append("Right")
		else:
			raise
	results_2obj["Violation side"] = viol_side
	results_1obj["Violation side"] = ["None" for x in range(len(results_1obj))]

	# Final joining of 1on and 2obj trials
	all_subs_md = pd.concat((results_1obj, results_2obj))
	all_subs_md.reset_index(inplace=True, drop=True)

	all_subs_md.to_csv(prepared_data_path)

print("starting")
# set_trace()
box_pairs = []

##################
### FINAL PLOT ###
##################

out_fn = f"stat_overall_behavior_bar_with_clip_{lang}"

# behavior_plot(all_subs_md, results_1obj, results_2obj, out_fn=f'{out_dir}/stat_overall_behavior.png', dodge=True, jitter=True, colors=None, tight=True)
# behavior_barplot(all_subs_md, results_1obj, results_2obj, out_fn=f'{out_dir}/stat_overall_behavior_bar.png', colors=None, tight=True)

# behavior_barplot_humans_and_model(all_subs_md, results_1obj, results_2obj, model_results, model_perf='similarity', out_fn=f'{out_dir}/{out_fn}_similarity.png', colors=None, tight=True, model_hline=False)
behavior_barplot_humans_and_model(all_subs_md, results_1obj, results_2obj, model_results, model_perf='Error rate', out_fn=f'{out_dir}/{out_fn}_errorate.png', colors=None, tight=True, redo_legend=True)

# behavior_barplot_humans_and_model(all_subs_md, results_1obj, results_2obj, model_results, model_perf='Error rate', rt='corrected RT', 
# 	min_rt=None, out_fn=f'{out_dir}/{out_fn}_errorate_corrected_rt.png', colors=None, tight=True, redo_legend=True)
exit()

##################
#### OVERALL #####
##################

# box_pairs = []
# make_sns_barplot(all_subs_md, x='Trial type', y='Error rate', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_overall_errorrate_point_dodge.png')
# make_sns_barplot(all_subs_md, x='Trial type', y='RT', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_overall_RT_point_dodge.png')

# # split by violation
# make_sns_barplot(all_subs_md, x='Trial type', y='Error rate', dodge=.02, col='Violation', kind='point', out_fn=f'{out_dir}/stat_overall*violation_errorrate_point_dodge.png')
# make_sns_barplot(all_subs_md, x='Trial type', y='RT', dodge=.02, col='Violation', kind='point', out_fn=f'{out_dir}/stat_overall*violation_RT_point_dodge.png')

# # ## split by subjects
# box_pairs = []
# make_sns_barplot(all_subs_md, x='Trial type', y='Error rate', hue='Subject', legend=False, dodge=.2, kind='point', out_fn=f'{out_dir}/stat_overall_all_subs_errorrate_point_dodge.png')
# make_sns_barplot(all_subs_md, x='Trial type', y='RT', hue='Subject', legend=False, dodge=.2, kind='point', out_fn=f'{out_dir}/stat_overall_all_subs_RT_point_dodge.png')

# ## With a panel for match-violation
# make_sns_barplot(all_subs_md, x='Trial type', y='Error rate', hue='Subject', col='Violation', legend=False, dodge=.2, kind='point', out_fn=f'{out_dir}/stat_overall_all_subs_by_violation_errorrate_point_dodge.png')
# make_sns_barplot(all_subs_md, x='Trial type', y='RT', hue='Subject', col='Violation', legend=False, dodge=.2, kind='point', out_fn=f'{out_dir}/stat_overall_all_subs_by_violation_RT_point_dodge.png')


# ##################
# ###### EASY ######
# ##################

# ### COLOUR ###
# box_pairs = [("vert", "rouge"), ("rouge", "bleu"), ("vert", "bleu")]
# make_sns_barplot(results_1obj, x='colour1', y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_color_errorrate_point.png')
# make_sns_barplot(results_1obj, x='colour1', y='Error rate', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_color_errorrate_box.png')

# make_sns_barplot(results_1obj.query("Perf==1"), x='colour1', y='RT', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_color_RT.png')
# make_sns_barplot(results_1obj.query("Perf==1"), x='colour1', y='RT', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_color_RT_point.png')
# make_sns_barplot(results_1obj.query("Perf==1"), x='colour1', y='RT', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_color_RT_box.png')

# ### SHAPE ###
# box_pairs = [("carre", "cercle"), ("cercle", "triangle"), ("carre", "triangle")]
# make_sns_barplot(results_1obj, x='shape1', y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_shape_errorrate_point.png')
# make_sns_barplot(results_1obj, x='shape1', y='Error rate', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_shape_errorrate_box.png')

# make_sns_barplot(results_1obj.query("Perf==1"), x='shape1', y='RT', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_shape_RT.png')
# make_sns_barplot(results_1obj.query("Perf==1"), x='shape1', y='RT', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_shape_RT_point.png')
# make_sns_barplot(results_1obj.query("Perf==1"), x='shape1', y='RT', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_shape_RT_box.png')

# ### SHAPE * COLOR ###
# make_sns_barplot(results_1obj, x='shape1', hue='colour1', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_1obj_shape*color_errorrate_point.png', order=['carre', 'triangle', 'cercle'], hue_order=['rouge', 'vert', 'bleu'], colors={"rouge":(1,0,0), "vert":(0,1,0), "bleu":(0,0,1)})
# make_sns_barplot(results_1obj.query("Perf==1"), x='shape1', hue='colour1', y='RT', kind='point', out_fn=f'{out_dir}/stat_1obj_shape*color_RT_point.png', order=['carre', 'triangle', 'cercle'], hue_order=['rouge', 'vert', 'bleu'], colors={"rouge":(1,0,0), "vert":(0,1,0), "bleu":(0,0,1)})

# ### MAPPING ###
# box_pairs = [("Right", "Left")]
# make_sns_barplot(results_1obj, x="'match' button", y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_mapping_errorrate_point.png', order=["Left", "Right"])
# make_sns_barplot(results_1obj, x="'match' button", y='Error rate', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_mapping_errorrate_box.png', order=["Left", "Right"])

# make_sns_barplot(results_1obj.query("Perf==1"), x="'match' button", y='RT', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_mapping_RT_point.png', order=["Left", "Right"])
# make_sns_barplot(results_1obj.query("Perf==1"), x="'match' button", y='RT', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_mapping_RT_box.png', order=["Left", "Right"])

# ### MAPPING * VIOLATION ###
# make_sns_barplot(results_1obj, x="'match' button", y='Error rate', kind='point', hue='Violation', out_fn=f'{out_dir}/stat_1obj_mapping*violation_errorrate_point.png', hue_order=["No", "Yes"], order=["Left", "Right"])
# make_sns_barplot(results_1obj.query("Perf==1"), x="'match' button", y='RT', kind='point', hue='Violation', out_fn=f'{out_dir}/stat_1obj_mapping*violation_RT_point.png', hue_order=["No", "Yes"], order=["Left", "Right"])

# ### VIOLATION ###
box_pairs = [("No", "Yes")]
make_sns_barplot(results_1obj, x='Violation', y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_violation_errorrate_point.png', order=["No", "Yes"])
# make_sns_barplot(results_1obj, x='Violation', y='Error rate', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_violation_errorrate_box.png', order=["No", "Yes"])

# make_sns_barplot(results_1obj.query("Perf==1"), x='Violation', y='RT', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_violation_RT.png', order=["No", "Yes"])
make_sns_barplot(results_1obj.query("Perf==1"), x='Violation', y='RT', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_violation_RT_point.png', order=["No", "Yes"])
# make_sns_barplot(results_1obj.query("Perf==1"), x='Violation', y='RT', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_violation_RT_box.png', order=["No", "Yes"])

# ### VIOLATION ON ### 
# make_sns_barplot(results_1obj, x='Violation on', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_1obj_violon_errorrate_point.png', order=["Shape", "Color"])
# make_sns_barplot(results_1obj.query("Perf==1"), x='Violation on', y='RT', kind='point', out_fn=f'{out_dir}/stat_1obj_violon_RT_point.png', order=["Shape", "Color"])

# ### SHAPE * VIOLATION ON ###
# make_sns_barplot(results_1obj, x='shape1', y='Error rate', kind='point', hue='Violation on', out_fn=f'{out_dir}/stat_1objshape*violon_errorrate_point.png', order=['carre', 'triangle', 'cercle'], hue_order=["Shape", "Color"])
# make_sns_barplot(results_1obj.query("Perf==1"), x='shape1', y='RT', kind='point', hue='Violation on', out_fn=f'{out_dir}/stat_1obj_shape*violon_RT_point.png', order=['carre', 'triangle', 'cercle'], hue_order=["Shape", "Color"])

# ### COLOR * VIOLATION ON ###
# make_sns_barplot(results_1obj, x='colour1', y='Error rate', kind='point', hue='Violation on', out_fn=f'{out_dir}/stat_1objcolor*violon_errorrate_point.png', order=['rouge', 'vert', 'bleu'], hue_order=["Shape", "Color"])
# make_sns_barplot(results_1obj.query("Perf==1"), x='colour1', y='RT', kind='point', hue='Violation on', out_fn=f'{out_dir}/stat_1obj_color*violon_RT_point.png', order=['rouge', 'vert', 'bleu'], hue_order=["Shape", "Color"])



# ## all of these showing all subjects
# make_sns_barplot(results_1obj, x='colour1', y='Error rate', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_1obj_all_subs_color_errorrate_point.png')
# make_sns_barplot(results_1obj.query("Perf==1"), x='colour1', y='RT', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_1obj_all_subs_color_RT_point.png')

# make_sns_barplot(results_1obj, x='shape1', y='Error rate', kind='point', dodge=.2, hue='Subject', legend=False, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_all_subs_shape_errorrate_point.png')
# make_sns_barplot(results_1obj.query("Perf==1"), x='shape1', y='RT', kind='point', dodge=.2, hue='Subject', legend=False, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_all_subs_shape_RT_point.png')

# make_sns_barplot(results_1obj, x="'match' button", y='Error rate', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_1obj_all_subs_mapping_errorrate_point.png', order=["Left", "Right"])
# make_sns_barplot(results_1obj.query("Perf==1"), x="'match' button", y='RT', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_1obj_all_subs_mapping_RT_point.png', order=["Left", "Right"])

# make_sns_barplot(results_1obj, x='Violation', y='Error rate', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_1obj_all_subs_violation_errorrate_point.png', order=["No", "Yes"])
# make_sns_barplot(results_1obj.query("Perf==1"), x='Violation', y='RT', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_1obj_all_subs_violation_RT_point.png', order=["No", "Yes"])

# make_sns_barplot(results_1obj, x='Violation on', y='Error rate', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_1obj_all_subs_violon_errorrate_point.png', order=["Shape", "Color"])
# make_sns_barplot(results_1obj.query("Perf==1"), x='Violation on', y='RT', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_1obj_all_subs_violon_RT_point.png', order=["Shape", "Color"])

# make_sns_barplot(results_1obj, x='shape1', hue='Subject', col='colour1', y='Error rate', kind='point', dodge=.2, legend=False, out_fn=f'{out_dir}/stat_1obj_all_shape*color_errorrate_point.png', order=['carre', 'triangle', 'cercle'])
# make_sns_barplot(results_1obj.query("Perf==1"), x='shape1', hue='Subject', col='colour1', y='RT', kind='point', dodge=.2, legend=False, out_fn=f'{out_dir}/stat_1obj_all_shape*color_RT_point.png', order=['carre', 'triangle', 'cercle'])



# # #################
# # ##### HARD ######
# # #################

# ### COLOUR ###
# make_sns_barplot(results_2obj, x='colour1', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_color1_errorrate_point.png')
# make_sns_barplot(results_2obj, x='colour2', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_color2_errorrate_point.png')

# make_sns_barplot(results_2obj.query("Perf==1"), x='colour1', y='RT', kind='point', out_fn=f'{out_dir}/stat_2obj_color1_RT_point.png')
# make_sns_barplot(results_2obj.query("Perf==1"), x='colour2', y='RT', kind='point', out_fn=f'{out_dir}/stat_2obj_color2_RT_point.png')

# ### SHAPE ###
# make_sns_barplot(results_2obj, x='shape1', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_shape1_errorrate_point.png')
# make_sns_barplot(results_2obj, x='shape2', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_shape2_errorrate_point.png')

# make_sns_barplot(results_2obj.query("Perf==1"), x='shape1', y='RT', kind='point', out_fn=f'{out_dir}/stat_2obj_shape1_RT_point.png')
# make_sns_barplot(results_2obj.query("Perf==1"), x='shape2', y='RT', kind='point', out_fn=f'{out_dir}/stat_2obj_shape2_RT_point.png')

# ### SHAPE * COLOR ###
# make_sns_barplot(results_2obj, x='shape1', hue='colour1', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_shape1*color1_errorrate_point.png', order=['carre', 'triangle', 'cercle'], hue_order=['rouge', 'vert', 'bleu'], colors={"rouge":(1,0,0), "vert":(0,1,0), "bleu":(0,0,1)})
# make_sns_barplot(results_2obj.query("Perf==1"), x='shape1', hue='colour1', y='RT', kind='point', out_fn=f'{out_dir}/stat_2obj_shape1*color1_RT_point.png', order=['carre', 'triangle', 'cercle'], hue_order=['rouge', 'vert', 'bleu'], colors={"rouge":(1,0,0), "vert":(0,1,0), "bleu":(0,0,1)})

# make_sns_barplot(results_2obj, x='shape2', hue='colour2', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_shape2*color2_errorrate_point.png', order=['carre', 'triangle', 'cercle'], hue_order=['rouge', 'vert', 'bleu'], colors={"rouge":(1,0,0), "vert":(0,1,0), "bleu":(0,0,1)})
# make_sns_barplot(results_2obj.query("Perf==1"), x='shape2', hue='colour2', y='RT', kind='point', out_fn=f'{out_dir}/stat_2obj_shape2*color2_RT_point.png', order=['carre', 'triangle', 'cercle'], hue_order=['rouge', 'vert', 'bleu'], colors={"rouge":(1,0,0), "vert":(0,1,0), "bleu":(0,0,1)})

# ### SHAPE * SHAPE ####
# make_sns_barplot(results_2obj, x='shape1', hue='shape2', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_shape1*shape2_errorrate_point.png', order=['carre', 'triangle', 'cercle'], hue_order=['carre', 'triangle', 'cercle'])
# make_sns_barplot(results_2obj.query("Perf==1"), x='shape1', hue='shape2', y='RT', kind='point', out_fn=f'{out_dir}/stat_2obj_shape1*shape2_RT_point.png', order=['carre', 'triangle', 'cercle'], hue_order=['carre', 'triangle', 'cercle'])

# ### COLOR * COLOR
# make_sns_barplot(results_2obj, x='colour1', hue='colour2', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_color1*color2_errorrate_point.png', order=['rouge', 'vert', 'bleu'], hue_order=['rouge', 'vert', 'bleu'], colors={"rouge":(1,0,0), "vert":(0,1,0), "bleu":(0,0,1)})
# make_sns_barplot(results_2obj.query("Perf==1"), x='colour1', hue='colour2', y='RT', kind='point', out_fn=f'{out_dir}/stat_2obj_color1*color2_RT_point.png', order=['rouge', 'vert', 'bleu'], hue_order=['rouge', 'vert', 'bleu'], colors={"rouge":(1,0,0), "vert":(0,1,0), "bleu":(0,0,1)})

# ### MAPPING ###
# box_pairs = [("Right", "Left")]
# make_sns_barplot(results_2obj, x="'match' button", y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_mapping_errorrate_point.png', order=["Left", "Right"])
# make_sns_barplot(results_2obj, x="'match' button", y='Error rate', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_mapping_errorrate_box.png', order=["Left", "Right"])

# make_sns_barplot(results_2obj.query("Perf==1"), x="'match' button", y='RT', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_mapping_RT_point.png', order=["Left", "Right"])
# make_sns_barplot(results_2obj.query("Perf==1"), x="'match' button", y='RT', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_mapping_RT_box.png', order=["Left", "Right"])

# ### MAPPING * VIOLATION ###
# make_sns_barplot(results_2obj, x="'match' button", y='Error rate', kind='point', hue='Violation', out_fn=f'{out_dir}/stat_2obj_mapping*violation_errorrate_point.png', hue_order=["No", "Yes"], order=["Left", "Right"])
# make_sns_barplot(results_2obj.query("Perf==1"), x="'match' button", y='RT', kind='point', hue='Violation', out_fn=f'{out_dir}/stat_2obj_mapping*violation_RT_point.png', hue_order=["No", "Yes"], order=["Left", "Right"])

# # ### RELATION ###
# make_sns_barplot(results_2obj, x="relation", y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_relation_errorrate_point.png', order=["à droite d'", "à gauche d'"])
# make_sns_barplot(results_2obj.query("Perf==1"), x="relation", y='RT', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_relation_RT_point.png', order=["à droite d'", "à gauche d'"])

# # ### RELATION * VIOLATION ###
# make_sns_barplot(results_2obj, x="relation", y='Error rate', kind='point', hue='Violation', out_fn=f'{out_dir}/stat_2obj_relation*violation_errorrate_point.png', order=["à droite d'", "à gauche d'"], hue_order=["No", "Yes"])
# make_sns_barplot(results_2obj.query("Perf==1"), x="relation", y='RT', kind='point', hue='Violation', out_fn=f'{out_dir}/stat_2obj_relation*violation_RT_point.png', order=["à droite d'", "à gauche d'"], hue_order=["No", "Yes"])

# # ### RELATION * MAPPING ###
# make_sns_barplot(results_2obj, x="relation", y='Error rate', kind='point', hue="'match' button", out_fn=f'{out_dir}/stat_2obj_relation*matching_errorrate_point.png', order=["à droite d'", "à gauche d'"], hue_order=["Left", "Right"])
# make_sns_barplot(results_2obj.query("Perf==1"), x="relation", y='RT', kind='point', hue="'match' button", out_fn=f'{out_dir}/stat_2obj_relation*matching_RT_point.png', order=["à droite d'", "à gauche d'"], hue_order=["Left", "Right"])


# ### VIOLATION ###
# box_pairs = [("No", "Yes")]
# make_sns_barplot(results_2obj, x='Violation', y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_violation_errorrate_point.png', order=["No", "Yes"])
# make_sns_barplot(results_2obj, x='Violation', y='Error rate', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_violation_errorrate_box.png', order=["No", "Yes"])

# make_sns_barplot(results_2obj.query("Perf==1"), x='Violation', y='RT', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_violation_RT.png', order=["No", "Yes"])
# make_sns_barplot(results_2obj.query("Perf==1"), x='Violation', y='RT', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_violation_RT_point.png', order=["No", "Yes"])
# make_sns_barplot(results_2obj.query("Perf==1"), x='Violation', y='RT', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_violation_RT_box.png', order=["No", "Yes"])


# ### VIOLATION ON PROPERTY - COLOR VS SHAPE ### 
local_results_2obj = deepcopy(results_2obj).query("`Violation on`=='property'")
local_results_2obj["Violation on"] = local_results_2obj["Change"].apply(lambda x: "Color" if "colour" in x else "Shape")
make_sns_barplot(local_results_2obj, x='Violation on', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_violon_errorrate_point.png', order=["Shape", "Color"])
make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violation on', y='RT', kind='point', out_fn=f'{out_dir}/stat_2obj_violon_RT_point.png', order=["Shape", "Color"])



# ## all of these showing all subjects
# make_sns_barplot(results_2obj, x='colour1', y='Error rate', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_2obj_all_subs_color_errorrate_point.png')
# make_sns_barplot(results_2obj.query("Perf==1"), x='colour1', y='RT', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_2obj_all_subs_color_RT_point.png')

# make_sns_barplot(results_2obj, x='shape1', y='Error rate', kind='point', dodge=.2, hue='Subject', legend=False, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_all_subs_shape_errorrate_point.png')
# make_sns_barplot(results_2obj.query("Perf==1"), x='shape1', y='RT', kind='point', dodge=.2, hue='Subject', legend=False, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_all_subs_shape_RT_point.png')

# make_sns_barplot(results_2obj, x="'match' button", y='Error rate', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_2obj_all_subs_mapping_errorrate_point.png', order=["Left", "Right"])
# make_sns_barplot(results_2obj.query("Perf==1"), x="'match' button", y='RT', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_2obj_all_subs_mapping_RT_point.png', order=["Left", "Right"])

# make_sns_barplot(results_2obj, x='Violation', y='Error rate', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_2obj_all_subs_violation_errorrate_point.png', order=["No", "Yes"])
# make_sns_barplot(results_2obj.query("Perf==1"), x='Violation', y='RT', kind='point', dodge=.2, hue='Subject', legend=False, out_fn=f'{out_dir}/stat_2obj_all_subs_violation_RT_point.png', order=["No", "Yes"])

# make_sns_barplot(results_2obj, x='shape1', hue='Subject', y='Error rate', kind='point', dodge=.2, legend=False, out_fn=f'{out_dir}/stat_1obj_all_shape*color_errorrate_point.png', order=['carre', 'triangle', 'cercle'])
# make_sns_barplot(results_2obj.query("Perf==1"), x='shape1', hue='Subject', y='RT', kind='point', dodge=.2, legend=False, out_fn=f'{out_dir}/stat_1obj_all_shape*color_RT_point.png', order=['carre', 'triangle', 'cercle'])

# make_sns_barplot(results_2obj, x="relation", y='Error rate', kind='point', dodge=.2, legend=False, hue='Subject', out_fn=f'{out_dir}/stat_2obj_all_subs_relation_errorrate_point.png', order=["à droite d'", "à gauche d'"])
# make_sns_barplot(results_2obj.query("Perf==1"), x="relation", y='RT', kind='point', dodge=.2, legend=False, hue='Subject', out_fn=f'{out_dir}/stat_2obj_all_subs_relation_RT_point.png', order=["à droite d'", "à gauche d'"])



# ## SHARED FEATURES IN 2 OBJ TRIALS
# # all subs
# make_sns_barplot(results_2obj, x='# Shared features', y='Error rate', hue='Subject', col='Violation', legend=False, dodge=.2, kind='point', out_fn=f'{out_dir}/stat_#shared_per_subj_by_violation_errorrate_point_dodge.png', order=[2, 1, 0])
# make_sns_barplot(results_2obj.query("Perf==1"), x='# Shared features', y='RT', hue='Subject', col='Violation', legend=False, dodge=.2, kind='point', out_fn=f'{out_dir}/stat_#shared_per_subj_by_violation_RT_point_dodge.png', order=[2, 1, 0])

# make_sns_barplot(results_2obj, x='Sharing', y='Error rate', hue='Subject', col='Violation', legend=False, dodge=.2, kind='point', out_fn=f'{out_dir}/stat_sharing_per_subj_by_violation_errorrate_point_dodge.png', order=["Both", "Shape", "Color", "None"])
# make_sns_barplot(results_2obj.query("Perf==1"), x='Sharing', y='RT', hue='Subject', col='Violation', legend=False, dodge=.2, kind='point', out_fn=f'{out_dir}/stat_sharing_per_subj_by_violation_RT_point_dodge.png', order=["Both", "Shape", "Color", "None"])

# # ave
# make_sns_barplot(results_2obj, x='# Shared features', y='Error rate', hue='Violation', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_#shared_by_violation_errorrate_point_dodge.png', order=[2, 1, 0])
# make_sns_barplot(results_2obj.query("Perf==1"), x='# Shared features', y='RT', hue='Violation', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_#shared_by_violation_RT_point_dodge.png', order=[2, 1, 0])

# make_sns_barplot(results_2obj, x='Sharing', y='Error rate', hue='Violation', col='colour1', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_sharing_by_violation_errorrate_point_dodge.png', order=["Both", "Shape", "Color", "None"])
# make_sns_barplot(results_2obj.query("Perf==1"), x='Sharing', y='RT', hue='Violation', col='colour1', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_sharing_by_violation_RT_point_dodge.png', order=["Both", "Shape", "Color", "None"])


# ### ERROR_TYPE ###
# make_sns_barplot(results_2obj, x='Violation on', y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violon_errorrate.png', order=["None", "property", "binding", "relation"])
# make_sns_barplot(results_2obj.query("Perf==1"), x='Violation on', y='RT', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Vilon on_RT.png', order=["None", "property", "binding", "relation"])

# ## SHARED FEATURES BY ERROR TYPE
# make_sns_barplot(results_2obj, x='# Shared features', y='Error rate', hue='Violation on', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_#shared_by_Violon_errorrate_point_dodge.png', order=[2, 1, 0], hue_order=["None", "property", "binding", "relation"])
# make_sns_barplot(results_2obj.query("Perf==1"), x='# Shared features', y='RT', hue='Violation on', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_#shared_by_Violon_RT_point_dodge.png', order=[2, 1, 0], hue_order=["None", "property", "binding", "relation"])

# ## SHARING BY ERROR TYPE
make_sns_barplot(results_2obj, x='Sharing', y='Error rate', hue='Violation on', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_sharing_by_Violon_errorrate_point_dodge.png', order=["Both", "Shape", "Color", "None"], hue_order=["None", "property", "binding", "relation"])
make_sns_barplot(results_2obj.query("Perf==1"), x='Sharing', y='RT', hue='Violation on', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_sharing_by_Violon_RT_point_dodge.png', order=["Both", "Shape", "Color", "None"], hue_order=["None", "property", "binding", "relation"])

# make_sns_barplot(results_2obj, x='Violation on', y='Error rate', hue='Sharing', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_Violon_by_sharing_errorrate_point_dodge.png', hue_order=["Both", "Shape", "Color", "None"], order=["None", "property", "binding", "relation"])
# make_sns_barplot(results_2obj.query("Perf==1"), x='Violation on', y='RT', hue='Sharing', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_Violon_by_sharing_RT_point_dodge.png', hue_order=["Both", "Shape", "Color", "None"], order=["None", "property", "binding", "relation"])



# ### VIOLATION POSITION ###
# Violated_position = 3 <-> L2 (relation inversion) error 
# Violated_position in [1,2,4,5] <-> L0 (dumb change) error
# Violated_position in [1.5, 4.5] <-> L1 (binding) error
if True:
	local_results_2obj = deepcopy(results_2obj).query('Error_type=="l0"')
	
	# Spatial positon
	make_sns_barplot(local_results_2obj, x='Violation side', y='Error rate', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violation_side_error_rate.png', order=["Left", "Right"])
	make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violation side', y='RT', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violation_side_RT.png', order=["Left", "Right"])
	
	make_sns_barplot(local_results_2obj, x='Violation side', y='Error rate', hue='Strategy', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violation_side*Strategy_error_rate.png', order=["Left", "Right"])
	make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violation side', y='RT', hue='Strategy', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violation_side*Relation*Strategy_RT.png', order=["Left", "Right"])
	

	# ordinal positon
	make_sns_barplot(local_results_2obj, x='Violation ordinal position', y='Error rate', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violation_ordinal_position_error_rate.png', order=["First", "Second"])
	make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violation ordinal position', y='RT', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violation_ordinal_position_RT.png', order=["First", "Second"])
	
	make_sns_barplot(local_results_2obj, x='Violation ordinal position', y='Error rate', hue='Strategy', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violation_ordinal_position*Strategy_error_rate.png', order=["First", "Second"])
	make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violation ordinal position', y='RT', hue='Strategy', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violation_ordinal_position*Relation*Strategy_RT.png', order=["First", "Second"])
	


	## SAME FOR ALL SUBJECTS
	# Spatial positon
	make_sns_barplot(local_results_2obj, x='Violation side', y='Error rate', hue='Subject', kind='point', dodge=.2, legend=False, out_fn=f'{out_dir}/stat_point_2obj_all_subs_Violation_side_error_rate.png', order=["Left", "Right"])
	make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violation side', y='RT', hue='Subject', kind='point', dodge=.2, legend=False, out_fn=f'{out_dir}/stat_point_2obj_all_subs_Violation_side_RT.png', order=["Left", "Right"])
	# ordinal positon
	make_sns_barplot(local_results_2obj, x='Violation ordinal position', y='Error rate', hue='Subject', kind='point', dodge=.2, legend=False, out_fn=f'{out_dir}/stat_point_2obj_all_subs_Violation_ordinal_position_error_rate.png', order=["First", "Second"])
	make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violation ordinal position', y='RT', hue='Subject', kind='point', dodge=.2, legend=False, out_fn=f'{out_dir}/stat_point_2obj_all_subs_Violation_ordinal_position_RT.png', order=["First", "Second"])
	


	# # Just position
	# # box_pairs = [(1.0, 2.0), (1.0, 4.0), (1.0, 5.0)] #, (2.0, 4.0), (2.0, 5.0), (4.0, 5.0)]
	# # make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue=None, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position_RT.png')
	# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position_RT.png')
	# # make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue=None, kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position_RT.png')

	# # position*relation
	# # box_pairs = [((1.0, "à gauche d'"), (1.0, "à droite d'")), ((1.0, "à gauche d'"), (2.0, "à gauche d'")), ((1.0, "à gauche d'"), (4.0, "à gauche d'"))]
	# # make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Relation_RT.png')
	# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Relation_RT.png')
	# # make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Relation_RT.png')

	# # position*relation*strategy
	# # box_pairs = [((1.0, 'sequential'), (1.0, 'visual')), ((2.0, 'sequential'), (2.0, 'visual')),
	# # 			 ((4.0, 'sequential'), (4.0, 'visual')), ((5.0, 'sequential'), (5.0, 'visual'))]
	# # make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Strategy', col='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Relation*Strategy_RT.png')
	# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Strategy', col='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Relation*Strategy_RT.png')
	# # make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Strategy', col='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Relation*Strategy_RT.png')

	# # position*strategy
	# # box_pairs = [((1.0, 'sequential'), (1.0, 'visual')), ((2.0, 'sequential'), (2.0, 'visual')),
	# # 			 ((4.0, 'sequential'), (4.0, 'visual')), ((5.0, 'sequential'), (5.0, 'visual'))]
	# # make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Strategy', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Strategy_RT.png')
	# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Strategy', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Strategy_RT.png')
	# # make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Strategy', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Strategy_RT.png')


	# # split by difficulty
	# # box_pairs = []
	# # make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='# Shared features', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*#Shared_features_RT.png')
	# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='# Shared features', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*#Shared_features_RT.png')
	# # make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='# Shared features', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*#Shared_features_RT.png')



	# ### SAME WITH PERF
	# # box_pairs = [(1.0, 2.0), (1.0, 4.0), (1.0, 5.0)] #, (2.0, 4.0), (2.0, 5.0), (4.0, 5.0)]
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position_error_rate.png')
	# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position_error_rate.png')
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position_error_rate.png')

	# # position*relation
	# # box_pairs = [((1.0, "à gauche d'"), (1.0, "à droite d'")), ((1.0, "à gauche d'"), (2.0, "à gauche d'")), ((1.0, "à gauche d'"), (4.0, "à gauche d'"))]
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Relation_error_rate.png')
	# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Relation_error_rate.png')
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Relation_error_rate.png')

	# # position*relation*strategy
	# # box_pairs = [((1.0, 'sequential'), (4.0, 'sequential')), ((1.0, 'visual'), (4.0, 'visual'))]
	# # box_pairs = [((1.0, 'sequential'), (1.0, 'visual')), ((2.0, 'sequential'), (2.0, 'visual')),
	# # 			 ((4.0, 'sequential'), (4.0, 'visual')), ((5.0, 'sequential'), (5.0, 'visual'))]
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', col='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Relation*Strategy_error_rate.png')
	# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', col='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Relation*Strategy_error_rate.png')
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', col='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Relation*Strategy_error_rate.png')

	# # position*strategy
	# # box_pairs = [((1.0, 'sequential'), (1.0, 'visual')), ((2.0, 'sequential'), (2.0, 'visual')),
	# # 			 ((4.0, 'sequential'), (4.0, 'visual')), ((5.0, 'sequential'), (5.0, 'visual'))]
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Strategy_error_rate.png')
	# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Strategy_error_rate.png')
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Strategy_error_rate.png')


	# # split by difficulty
	# # box_pairs = []
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='# Shared features', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*#_Shared_features_error_rate.png')
	# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='# Shared features', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*#_Shared_features_error_rate.png')
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='# Shared features', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*#_Shared_features_error_rate.png')




# ## L1 trials
# local_results_2obj = deepcopy(results_2obj).query('Error_type=="l1"')
# box_pairs = [(1.5, 4.5)]
# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue=None, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position_RT.png')
# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position_RT.png')
# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue=None, kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position_RT.png')

# # position*relation
# box_pairs = []
# box_pairs = [((1.5, "à gauche d'"), (1.5, "à droite d'")), ((4.5, "à gauche d'"), (4.5, "à gauche d'"))] #, ((1.0, "à gauche d'"), (4.0, "à gauche d'"))]
# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position*Relation_RT.png')
# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position*Relation_RT.png')
# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position*Relation_RT.png')

# # position*strategy
# # box_pairs = []
# box_pairs = [((1.5, 'sequential'), (1.5, 'visual')), ((4.5, 'sequential'), (4.5, 'visual')),
# 			 ((1.5, 'sequential'), (4.5, 'sequential')), ((1.5, 'visual'), (4.5, 'visual'))]
# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Strategy', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position*Strategy_RT.png')
# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Strategy', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position*Strategy_RT.png')
# make_sns_barplot(local_results_2obj.query("Perf==1"), x='Violated_position', y='RT', hue='Strategy', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position*Strategy_RT.png')


# ## L1 trials -- Error rate
# local_results_2obj = deepcopy(results_2obj).query('Error_type=="l1"')
# box_pairs = [(1.5, 4.5)]
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position_error_rate.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position_error_rate.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position_error_rate.png')

# # position*relation
# box_pairs = []
# box_pairs = [((1.5, "à gauche d'"), (1.5, "à droite d'")), ((4.5, "à gauche d'"), (4.5, "à gauche d'"))] #, ((1.0, "à gauche d'"), (4.0, "à gauche d'"))]
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position*Relation_error_rate.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position*Relation_error_rate.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position*Relation_error_rate.png')

# # position*strategy
# # box_pairs = []
# box_pairs = [((1.5, 'sequential'), (1.5, 'visual')), ((4.5, 'sequential'), (4.5, 'visual')),
# 			 ((1.5, 'sequential'), (4.5, 'sequential')), ((1.5, 'visual'), (4.5, 'visual'))]
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position*Strategy_error_rate.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position*Strategy_error_rate.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Strategy', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position*Strategy_error_rate.png')




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
# # # robjects.globalenv['colour1'] = results_1obj.colour1
# # # robjects.globalenv['shape1'] = results_1obj.shape1

# # results_1obj["Matching"][results_1obj["Matching"]=="match"] = 1
# # results_1obj["Matching"][results_1obj["Matching"]=="nonmatch"] = -1
# # results_1obj["Mapping"][results_1obj["Mapping"]=="correct_left"] = 1
# # results_1obj["Mapping"][results_1obj["Mapping"]=="correct_right"] = -1
# # robjects.globalenv['Matching'] = results_1obj.Matching
# # robjects.globalenv['colour1'] = results_1obj.colour1
# # robjects.globalenv['shape1'] = results_1obj.shape1

# # results_1obj["red"] = pd.get_dummies(results_1obj.colour1).iloc[:,0]
# # results_1obj["blue"] = pd.get_dummies(results_1obj.colour1).iloc[:,1]
# # results_1obj["green"] = pd.get_dummies(results_1obj.colour1).iloc[:,2]
# # robjects.globalenv['red'] = pd.get_dummies(results_1obj.colour1).iloc[:,0]
# # robjects.globalenv['green'] = results_1obj.blue
# # robjects.globalenv['blue'] = results_1obj.green
# # # robjects.globalenv['Matching'] = (results_1obj.Matching == "nonmatch").values.astype(int)
# # robjects.globalenv['RT'] = results_1obj.RT
# # robjects.globalenv['Subject'] = results_1obj.Subject
# # lm = lme4.lmer('RT ~ Matching + red + green + blue + shape1 + (1|Subject)')
# # print(base.summary(lm))



# # results_1obj["red"] = pd.get_dummies(results_1obj.colour1).iloc[:,0]
# # results_1obj["blue"] = pd.get_dummies(results_1obj.colour1).iloc[:,1]
# # results_1obj["green"] = pd.get_dummies(results_1obj.colour1).iloc[:,2]
# # res = smf.mixedlm("RT ~ 1 + red + blue + green", results_1obj, groups=results_1obj["Subject"]).fit(method="bfgs")
# # print(res.summary())


# # results_1obj["Matching"][results_1obj["Matching"]=="match"] = .5
# # results_1obj["Matching"][results_1obj["Matching"]=="nonmatch"] = -.5
# # results_1obj["Mapping"][results_1obj["Mapping"]=="correct_left"] = .5
# # results_1obj["Mapping"][results_1obj["Mapping"]=="correct_right"] = -.5

# ## Stats single object
# smf.mixedlm("RT ~ Matching + Mapping + colour1 + shape1", results_1obj, groups=results_1obj["Subject"]).fit(method="bfgs").summary()
# smf.mixedlm("Perf ~ Matching + Mapping + colour1 + shape1", results_1obj, groups=results_1obj["Subject"]).fit(method="bfgs").summary()

# ## Stats two objects
# # smf.mixedlm("RT ~ Matching + Mapping + colour1 + shape1 + colour2 + shape2 + Relation", results_2obj, groups=results_2obj["Subject"]).fit(method="bfgs").summary()
# smf.mixedlm("RT ~ Matching + Mapping + colour1 + shape1 + colour2 + shape2 + Relation + Difficulty", results_2obj, groups=results_2obj["Subject"]).fit(method="bfgs").summary()
# smf.mixedlm("Perf ~ Matching + Mapping + colour1 + shape1 + colour2 + shape2 + Relation + Difficulty", results_2obj, groups=results_2obj["Subject"]).fit(method="bfgs").summary()


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

