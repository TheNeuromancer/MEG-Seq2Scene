from ipdb import set_trace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import statsmodels
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot


def plot_histograms(results, queries, colors, fn):
	assert len(colors) == len(queries)

	# get the query values and save them as labels 
	labels = [query.split('==')[1].strip() if '==' in query else query.split('in')[1].strip() for query in queries]

	data_to_plot = []
	for query in queries:
		# set_trace()
		dat = results.query(query)['RT'].values
		dat = dat[~np.isnan(dat)] # reject nans
		data_to_plot.append(dat)
	
	# determine the bins (to have the same for both histogram)
	bins = np.histogram(np.concatenate(data_to_plot), bins=40)[1] #get the bin edges
	
	fig, ax = plt.subplots()
	for datum, color, label in zip(data_to_plot, colors, labels):
		plt.hist(datum, color=color, bins=bins, alpha=0.5, label=label)

	# vertical line for each mean
	ymin, ymax = ax.get_ylim()
	for datum, color in zip(data_to_plot, colors):
		plt.vlines(x=np.mean(datum), ymin=ymin, ymax=ymax, color=color)

	plt.xlabel('RT (ms)')
	plt.ylabel('Count')
	plt.legend()
	plt.tight_layout()
	fig.savefig(fn)
	plt.close(fig)
	return 


def plot_means(results, grouping_query, fn, colors=('darkviolet', 'dodgerblue')):
	grouped = results.groupby(grouping_query)
	labels = [*grouped.groups]
	mean_shape = grouped.mean()
	std_shape = grouped.std()
	fig, ax = plt.subplots()
	indices = np.arange(len(labels))
	width = 0.3

	ax.bar(indices-width/2, mean_shape.RT, yerr=std_shape.RT, width=width, color=colors[0])
	ax.yaxis.label.set_color(colors[0])
	ax.set_ylabel('RT')
	ax.tick_params(axis='y', colors=colors[0])

	ax2 = ax.twinx()
	ax2.bar(indices+width/2, mean_shape.Perf, yerr=std_shape.Perf, width=width, color=colors[1])
	plt.xticks(indices, labels, rotation='45', ha='right')
	ax2.yaxis.label.set_color(colors[1])
	ax2.set_ylabel('Perf')
	ax2.tick_params(axis='y', colors=colors[1])

	plt.tight_layout()
	fig.savefig(fn)
	plt.close(fig)
	return


def plot_interaction_means(results, grouping_query, fn, colors=('darkviolet', 'dodgerblue')):
	# grouped_multiple = df.groupby(['Team', 'Pos']).agg({'Age': ['mean', 'min', 'max']})
	grouped = results.groupby(grouping_query)
	labels = [*grouped.groups]
	mean_shape = grouped.mean()
	std_shape = grouped.std()
	fig, ax = plt.subplots()
	indices = np.arange(len(labels))
	width = 0.3

	ax.bar(indices-width/2, mean_shape.RT, yerr=std_shape.RT, width=width, color=colors[0])
	ax.yaxis.label.set_color(colors[0])
	ax.set_ylabel('RT')
	ax.tick_params(axis='y', colors=colors[0])

	ax2 = ax.twinx()
	ax2.bar(indices+width/2, mean_shape.Perf, yerr=std_shape.Perf, width=width, color=colors[1])
	plt.xticks(indices, labels, rotation='45', ha='right')
	ax2.yaxis.label.set_color(colors[1])
	ax2.set_ylabel('Perf')
	ax2.tick_params(axis='y', colors=colors[1])

	plt.tight_layout()
	fig.savefig(fn)
	plt.close(fig)
	return


def two_way_anova(results, factors, typ=3):
	formula = f'RT~C({factors[0]})+C({factors[1]})+C({factors[0]}):C({factors[1]})'
	model = ols(formula, results).fit()
	aov_table = anova_lm(model, typ=typ)
	print(aov_table)


def plot_interaction(results, factor1, factor2, fn, dep_var='RT', colors=['red', 'blue', 'green'], markers=['D', '^', '*']):
	fig, ax = plt.subplots(figsize=(6, 6))
	
	fig = interaction_plot(x=results[factor1].values, trace=results[factor2], 
		response=results[dep_var], colors=colors, markers=markers, ms=10, ax=ax, xlabel=factor1)

	# kinda convoluted way to get error bars
	aggreg = results.groupby([factor1, factor2]).agg({dep_var: ['mean', 'std']})
	mean_dep_var = aggreg.to_dict()[(dep_var, 'mean')]
	std_dep_var = aggreg.to_dict()[(dep_var, 'std')]
	fac1_labels = results[factor1].unique().tolist()
	fac1_labels.sort()
	fac2_labels = results[factor2].unique().tolist()
	fac2_labels.sort()
	for key in mean_dep_var.keys():
		x = fac1_labels.index(key[0])
		diff_idx = fac2_labels.index(key[1])
		plt.errorbar(x=x, y=mean_dep_var[key], yerr=std_dep_var[key], color=colors[diff_idx], marker=markers[diff_idx], alpha=.3)

	plt.savefig(fn)
	plt.close(fig)
	return


def get_violated_objects(violated_positions, relations):

	violated_objects = [] 

	for violated_position, relation in zip(violated_positions, relations):
		if violated_position == '3':
			# violtion is on the relation ... 
			#no sense in looking at the spatial position of the violated object
			violated_objects.append('None')
		
		if relation == 'dans':
			if violated_position < 3:
				violated_objects.append('dans')
			else:
				violated_objects.append('autour')
		elif relation == 'autour d\'':
			if violated_position < 3:
				violated_objects.append('autour')
			else:
				violated_objects.append('dans')
		
		elif relation == 'à droite d\'':
			if violated_position < 3:
				violated_objects.append('droite')
			else:
				violated_objects.append('gauche')
		elif relation == 'à gauche d\'':
			if violated_position < 3:
				violated_objects.append('gauche')
			else:
				violated_objects.append('droite')

		else:
			print(relation)
			raise

	print(violated_objects)
	return violated_objects
	
	