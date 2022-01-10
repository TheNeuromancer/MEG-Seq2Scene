import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from ipdb import set_trace
from glob import glob
import os.path as op
import os
import seaborn as sns
from statannot import add_stat_annotation
from statannotations.Annotator import Annotator
from copy import deepcopy


LIGHT_BLUE, DARK_BLUE = (.4, .4, .7), (0, .2, .5)
LIGHT_GRAY, DARK_GRAY = (.7, .8, .8), (.5, .6, .6)
LIGHT_RED, MEDIUM_RED, DARK_RED = (1, .7, .7), (.9, .4, .4), (.8, .2, .2)
PURE_RED, PURE_GREEN = (1,0,0), (0,1,0)
WHITE = (1,1,1)

BLUE =  (0.10588235294117647, 0.3843137254901961, 0.6470588235294118)
ORANGE =  (0.9882352941176471, 0.4117647058823529, 0.058823529411764705)
GREEN =  (0.14901960784313725, 0.5764705882352941, 0.12941176470588237)
RED =  (0.792156862745098, 0.06666666666666667, 0.11764705882352941)
PURPLE =  (0.5019607843137255, 0.30980392156862746, 0.6862745098039216)
BROWN =  (0.47058823529411764, 0.2627450980392157, 0.23137254901960785)

def make_sns_barplot(df, x, y, hue=None, box_pairs=[], kind='point', col=None, out_fn="tmp.png", ymin=None, ymax=None, 
                     hline=None, rotate_ticks=False, tight=True, ncol=1, order=None, hue_order=None, legend=True, dodge=True, jitter=True, colors=None):
    # fig, ax = plt.subplots(figsize=(9, 9))
    sns.set(font_scale = 2)
    # if kind == "bar":
    #     g = sns.barplot(x=x, y=y, hue=hue, data=df, ax=ax, ci=68, order=order, hue_order=hue_order) # ci=68 <=> standard error
    # else:
        # g = sns.boxplot(x=x, y=y, hue=hue, data=df, ax=ax, order=order, hue_order=hue_order, showfliers = False) #, order=order, hue_order=hue_order) # ci=68 <=> standard error
    if kind=="box":
        g = sns.catplot(x=x, y=y, hue=hue, col=col, data=df, kind=kind, order=order, hue_order=hue_order, ci=68, legend=False, palette=colors, showfliers=False)# ci=68 <=> standard error
    elif kind=="point":
        g = sns.catplot(x=x, y=y, hue=hue, col=col, data=df, kind=kind, order=order, hue_order=hue_order, ci=68, legend=False, palette=colors, jitter=jitter, dodge=dodge)# ci=68 <=> standard error
    else:
        g = sns.catplot(x=x, y=y, hue=hue, col=col, data=df, kind=kind, order=order, hue_order=hue_order, ci=68, legend=False, palette=colors)# ci=68 <=> standard error
    if ymin is not None or ymax is not None: ## SHOULD BE BEFORE CALLING ANNOTATOR!!
        g.set(ylim=(ymin, ymax))
    axes = g.axes[0] if col is not None else [g.ax]
    for ax in axes:
        ax.set_xlabel(x,fontsize=20)
        ax.set_ylabel(y, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if hline is not None:
            ax.axhline(y=hline, lw=1, ls='--', c='grey', zorder=-10)
        if rotate_ticks:
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha('right')
        
        if box_pairs and kind != "point":
            annotator = Annotator(ax, box_pairs, plot=f'{kind}plot', data=df, x=x, hue=hue, col=col, y=y, text_format='star', order=order, hue_order=hue_order) #,), line_offset_to_box=-1
            annotator.configure(test='Mann-Whitney', verbose=False, loc="inside", comparisons_correction="bonferroni", fontsize=12, use_fixed_offset=True).apply_and_annotate()
            # , line_offset=.0001, line_offset_to_group=.0001

        if tight:
            plt.tight_layout()
        if (ncol > 1 or hue is not None) and legend:
            ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=ncol, fontsize=12) # Put the legend out of the figure
    plt.savefig(out_fn, transparent=True, bbox_inches='tight', dpi=400)
    plt.close()


def behavior_plot(df_full, df_1obj, df_2obj, out_fn="tmp.png", dodge=True, jitter=True, colors=None, tight=True):
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(20, 6), sharey='row', sharex='col')
    sns.set(font_scale = 3)
    COLOR_MATCH = PURE_GREEN
    COLOR_MISMATCH = PURE_RED
    COLOR_MISMATCH2 = (.9,0.2,0)
    COLOR_MISMATCH3 = (.8,0.2,0)
    # ci=68 <=> standard error
    # First panel: overall perf for each bock type, split with violation
    g = sns.pointplot(x='Trial type', y='corrected RT', data=deepcopy(df_full).query("Perf==1"), order=['One object', 'Two objects'], 
        hue="Violation", ax=axes[0,0], ci=68, jitter=jitter, dodge=dodge, palette=[COLOR_MATCH, COLOR_MISMATCH], hue_order=("No", "Yes"), legend=False)
    g.set_xlabel('') 
    g.legend().set_visible(False)
    g = sns.pointplot(x='Trial type', y='Error rate', data=df_full, order=['One object', 'Two objects'], 
        hue="Violation", ax=axes[1,0], ci=68, jitter=jitter, dodge=dodge, palette=[COLOR_MATCH, COLOR_MISMATCH], hue_order=("No", "Yes"), legend=False)
    g.legend().set_visible(False)

    ## Second panel: feature mismatch for both one and two obj
    g = sns.pointplot(x='Changed property', y='corrected RT', data=deepcopy(df_full).query("Perf==1"), order=['Shape', 'Color'], ax=axes[0,1], ci=68, jitter=jitter, dodge=dodge,
        hue="Trial type", hue_order=['One object', 'Two objects'], palette=[COLOR_MISMATCH, COLOR_MISMATCH], linestyles=["--", "-"], legend=False)
    g.set_ylabel(''), g.set_xlabel('') 
    g.legend().set_visible(False)
    g = sns.pointplot(x='Changed property', y='Error rate', data=df_full, order=['Shape', 'Color'], ax=axes[1,1], ci=68, jitter=jitter, dodge=dodge,
        hue="Trial type", hue_order=['One object', 'Two objects'], palette=[COLOR_MISMATCH, COLOR_MISMATCH], linestyles=["--", "-"], legend=False)
    g.set_ylabel('')
    g.legend().set_visible(False)

    ## Third panel: kind of violation for 2 object trials
    g = sns.pointplot(x='Violation on', y='corrected RT', data=df_2obj, order=['property', 'binding', 'relation'], ax=axes[0,2], ci=68, jitter=jitter, dodge=dodge, color=COLOR_MISMATCH)
    g.set_ylabel(''), g.set_xlabel('') 
    g = sns.pointplot(x='Violation on', y='Error rate', data=df_2obj, order=['property', 'binding', 'relation'], ax=axes[1,2], ci=68, jitter=jitter, dodge=dodge, color=COLOR_MISMATCH)
    g.set_ylabel('')

    ## Fourth panel: ordinal position
    g = sns.pointplot(x='Violation ordinal position', y='corrected RT', data=deepcopy(df_2obj).query("Perf==1"), order=['First', 'Second'], ax=axes[0,3], ci=68, jitter=jitter, dodge=dodge, color=COLOR_MISMATCH)
    g.set_ylabel(''), g.set_xlabel('') 
    g = sns.pointplot(x='Violation ordinal position', y='Error rate', data=df_2obj, order=['First', 'Second'], ax=axes[1,3], ci=68, jitter=jitter, dodge=dodge, color=COLOR_MISMATCH)
    g.set_ylabel('')

    ## Fifth panel: spatial position
    g = sns.pointplot(x='Violation side', y='corrected RT', data=deepcopy(df_2obj).query("Perf==1"), order=['Left', 'Right'], ax=axes[0,4], ci=68, jitter=jitter, dodge=dodge, color=COLOR_MISMATCH)
    g.set_ylabel(''), g.set_xlabel('') 
    g = sns.pointplot(x='Violation side', y='Error rate', data=df_2obj, order=['Left', 'Right'], ax=axes[1,4], ci=68, jitter=jitter, dodge=dodge, color=COLOR_MISMATCH)
    g.set_ylabel('')

    ## Sixth panel: Shared features between objects
    g = sns.pointplot(x='Sharing', y='corrected RT', hue='Violation', hue_order=['No', 'Yes'], data=deepcopy(df_2obj).query("Perf==1"), 
        order=['Both', 'Shape', 'Color', 'None'], ax=axes[0,5], ci=68, jitter=jitter, dodge=dodge, palette=[COLOR_MATCH, COLOR_MISMATCH])
    g.set_ylabel(''), g.set_xlabel('')
    g.legend().set_visible(False)
    g = sns.pointplot(x='Sharing', y='Error rate', hue='Violation', hue_order=['No', 'Yes'], data=df_2obj, 
        order=['Both', 'Shape', 'Color', 'None'], ax=axes[1,5], ci=68, jitter=jitter, dodge=dodge, palette=[COLOR_MATCH, COLOR_MISMATCH])
    g.set_ylabel('')
    g.set_xlabel('Shared features between objects') 
    g.legend().set_visible(False)

    ## Seventh panel: Shared features between objects * kind of violation
    g = sns.pointplot(x='Sharing', y='corrected RT', data=deepcopy(df_2obj).query("Perf==1"), order=['Both', 'Shape', 'Color', 'None'], ax=axes[0,6], ci=68, jitter=0.2, dodge=0.2,
        hue="Violation on", hue_order=['property', 'binding', 'relation'], palette=(COLOR_MISMATCH, COLOR_MISMATCH2, COLOR_MISMATCH3), markers=["o", "^", "s"])
    g.set_ylabel(''), g.set_xlabel('')
    g.legend().set_visible(False)
    g = sns.pointplot(x='Sharing', y='Error rate', data=df_2obj, order=['Both', 'Shape', 'Color', 'None'], ax=axes[1,6], ci=68, jitter=0.2, dodge=0.2,
        hue="Violation on", hue_order=['property', 'binding', 'relation'], palette=(COLOR_MISMATCH, COLOR_MISMATCH2, COLOR_MISMATCH3), markers=["o", "^", "s"])
    g.set_ylabel('')
    g.set_xlabel('Shared features between objects') 
    g.legend().set_visible(False)
        
    if tight:
        plt.tight_layout()
        # if (ncol > 1 or hue is not None) and legend:
        #     ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=ncol, fontsize=12) # Put the legend out of the figure
    plt.subplots_adjust(hspace=0.07, wspace=0.07)
    plt.savefig(out_fn, transparent=True, bbox_inches='tight', dpi=400)
    plt.close()

    ## Make legend 
    fig, ax = plt.subplots()
    legendfig, legendax = plt.subplots(figsize=(5,5))
    lines = []
    ax.plot(0,0, c=COLOR_MATCH, label="Match", lw=4)
    ax.plot(0,0, c=COLOR_MISMATCH, label="Mismatch", lw=4)
    ax.plot(0,0, c='k', ls='-', label="Two objects", lw=4)
    ax.plot(0,0, c='k', ls='--', label="One object", lw=4)

    ax.plot(0,0, c=COLOR_MISMATCH, marker='o', label="Violation on property", lw=4, markersize=18)
    ax.plot(0,0, c=COLOR_MISMATCH2, marker='^', label="Violation on binding", lw=4, markersize=18)
    ax.plot(0,0, c=COLOR_MISMATCH3, marker='s', label="Violation on spatial relation", lw=4, markersize=18)

    legendfig.legend(*ax.get_legend_handles_labels(), loc='center')
    legendax.axis('off') # hide the axes frame and the x/y labels
    plt.tight_layout()
    legendfig.savefig(f"{out_fn[0:-4]}_legend.png", bbox_inches='tight', dpi=400)



# def behavior_barplot(df_full, df_1obj, df_2obj, out_fn="tmp.png", colors=None, tight=True):
#     fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(20, 6), sharey='row', sharex='col')
#     sns.set(font_scale = 3)
#     COLOR_MATCH = PURE_GREEN
#     COLOR_MISMATCH = PURE_RED
#     # COLOR_PROP_SHAPE, COLOR_PROP_COLOR = (.9, .5, .5), (.9, .3, .3)
#     COLOR_PROPERTY = LIGHT_RED
#     COLOR_BINDING, COLOR_RELATION = MEDIUM_RED, DARK_RED
#     COLOR_FIRST, COLOR_SECOND = LIGHT_BLUE, DARK_BLUE
#     COLOR_LEFT, COLOR_RIGHT = LIGHT_GREY, DARK_GREY
#     WIDTH = .385
#     # First panel: overall perf for each bock type, split with violation. white facecolor, colored edgecolor
#     edgecolors = [COLOR_MATCH]*2 + [COLOR_MISMATCH]*2
#     facecolors = [WHITE, COLOR_MATCH, WHITE, COLOR_MISMATCH]
#     g = sns.barplot(x='Trial type', y='RT', data=deepcopy(df_full).query("Perf==1"), order=['One object', 'Two objects'],
#         hue="Violation", ax=axes[0,0], ci=68, hue_order=("No", "Yes"))
#     for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
#         bar.set_edgecolor(edgecolor)
#         bar.set_facecolor(facecolor)
#         bar.set_width(WIDTH)
#     g.set_xlabel('') 
#     g.legend().set_visible(False)
#     g = sns.barplot(x='Trial type', y='Error rate', data=df_full, order=['One object', 'Two objects'],
#         hue="Violation", ax=axes[1,0], ci=68, hue_order=("No", "Yes"))
#     for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
#         bar.set_edgecolor(edgecolor)
#         bar.set_facecolor(facecolor)
#         bar.set_width(WIDTH)
#     g.legend().set_visible(False)

#     ## Second panel: feature mismatch for both one and two obj
#     # edgecolors = [COLOR_PROP_SHAPE, COLOR_PROP_COLOR, COLOR_PROP_SHAPE, COLOR_PROP_COLOR]
#     # facecolors = [WHITE, WHITE, COLOR_PROP_SHAPE, COLOR_PROP_COLOR]
#     edgecolors = [COLOR_PROPERTY, COLOR_PROPERTY, COLOR_PROPERTY, COLOR_PROPERTY]
#     facecolors = [WHITE, WHITE, COLOR_PROPERTY, COLOR_PROPERTY]
#     g = sns.barplot(x='Changed property', y='RT', data=deepcopy(df_full).query("Perf==1"), order=['Shape', 'Color'], ax=axes[0,1], ci=68,
#         hue="Trial type", hue_order=['One object', 'Two objects'])
#     for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
#         bar.set_edgecolor(edgecolor)
#         bar.set_facecolor(facecolor)
#         bar.set_width(WIDTH)
#     g.set_ylabel(''), g.set_xlabel('') 
#     g.legend().set_visible(False)
#     g = sns.barplot(x='Changed property', y='Error rate', data=df_full, order=['Shape', 'Color'], ax=axes[1,1], ci=68,
#         hue="Trial type", hue_order=['One object', 'Two objects'])
#     for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
#         bar.set_edgecolor(edgecolor)
#         bar.set_facecolor(facecolor)
#         bar.set_width(WIDTH)
#     g.set_ylabel(''), g.set_xlabel('Mismatch on')
#     g.legend().set_visible(False)

#     ## Third panel: kind of violation for 2 object trials
#     edgecolors = [COLOR_PROPERTY, COLOR_BINDING, COLOR_RELATION]
#     facecolors = [COLOR_PROPERTY, COLOR_BINDING, COLOR_RELATION]
#     g = sns.barplot(x='Violation on', y='RT', data=df_2obj, order=['property', 'binding', 'relation'], ax=axes[0,2], ci=68, color=COLOR_MISMATCH)
#     g.set_ylabel(''), g.set_xlabel('') 
#     for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
#         bar.set_edgecolor(edgecolor)
#         bar.set_facecolor(facecolor)
#     g = sns.barplot(x='Violation on', y='Error rate', data=df_2obj, order=['property', 'binding', 'relation'], ax=axes[1,2], ci=68, color=COLOR_MISMATCH)
#     g.set_ylabel(''), g.set_xlabel('Mismatch on')
#     for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
#         bar.set_edgecolor(edgecolor)
#         bar.set_facecolor(facecolor)

#     ## Fourth panel: ordinal position
#     edgecolors = [COLOR_FIRST, COLOR_SECOND]
#     facecolors = [COLOR_FIRST, COLOR_SECOND]
#     local_df_2obj = deepcopy(df_2obj)
#     local_df_2obj["Violation on"] = df_2obj["Violation ordinal position"].apply(lambda x: f"{x} object")
#     g = sns.barplot(x='Violation on', y='RT', data=deepcopy(local_df_2obj).query("Perf==1"), order=['First object', 'Second object'], ax=axes[0,3], ci=68, color=COLOR_MISMATCH)
#     g.set_ylabel(''), g.set_xlabel('') 
#     for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
#         bar.set_edgecolor(edgecolor)
#         bar.set_facecolor(facecolor)
#     g = sns.barplot(x='Violation on', y='Error rate', data=local_df_2obj, order=['First object', 'Second object'], ax=axes[1,3], ci=68, color=COLOR_MISMATCH)
#     g.set_ylabel(''), g.set_xlabel('Mismatch on')
#     for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
#         bar.set_edgecolor(edgecolor)
#         bar.set_facecolor(facecolor)

#     ## Fifth panel: spatial position
#     edgecolors = [COLOR_LEFT, COLOR_RIGHT]
#     facecolors = [COLOR_LEFT, COLOR_RIGHT]
#     local_df_2obj["Violation on"] = df_2obj["Violation side"].apply(lambda x: f"{x} object")
#     g = sns.barplot(x='Violation on', y='RT', data=deepcopy(local_df_2obj).query("Perf==1"), order=['Left object', 'Right object'], ax=axes[0,4], ci=68, color=COLOR_MISMATCH)
#     g.set_ylabel(''), g.set_xlabel('') 
#     for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
#         bar.set_edgecolor(edgecolor)
#         bar.set_facecolor(facecolor)
#     g = sns.barplot(x='Violation on', y='Error rate', data=local_df_2obj, order=['Left object', 'Right object'], ax=axes[1,4], ci=68, color=COLOR_MISMATCH)
#     g.set_ylabel(''), g.set_xlabel('Mismatch on')
#     for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
#         bar.set_edgecolor(edgecolor)
#         bar.set_facecolor(facecolor)

#     ## Sixth panel: Shared features between objects
#     g = sns.pointplot(x='Sharing', y='RT', hue='Violation', hue_order=['No', 'Yes'], data=deepcopy(df_2obj).query("Perf==1"), 
#         order=['Both', 'Shape', 'Color', 'None'], ax=axes[0,5], ci=68, jitter=True, dodge=True, palette=[COLOR_MATCH, COLOR_MISMATCH])
#     g.set_ylabel(''), g.set_xlabel('')
#     # g.legend().set_visible(False)
#     g.legend(prop={"size":12}, loc='lower right')
#     g = sns.pointplot(x='Sharing', y='Error rate', hue='Violation', hue_order=['No', 'Yes'], data=df_2obj, 
#         order=['Both', 'Shape', 'Color', 'None'], ax=axes[1,5], ci=68, jitter=True, dodge=True, palette=[COLOR_MATCH, COLOR_MISMATCH])
#     g.set_ylabel('')
#     g.set_xlabel('Shared features between objects') 
#     g.legend().set_visible(False)

#     ## Seventh panel: Shared features between objects * kind of violation
#     g = sns.pointplot(x='Sharing', y='RT', data=deepcopy(df_2obj).query("Perf==1"), order=['Both', 'Shape', 'Color', 'None'], ax=axes[0,6], ci=68, jitter=0.2, dodge=0.2,
#         hue="Violation on", hue_order=['property', 'binding', 'relation'], palette=(COLOR_PROPERTY, COLOR_BINDING, COLOR_RELATION)) #, markers=["o", "^", "s"])
#     g.set_ylabel(''), g.set_xlabel('')
#     g.legend(prop={"size":12}, loc='lower right')
#     g = sns.pointplot(x='Sharing', y='Error rate', data=df_2obj, order=['Both', 'Shape', 'Color', 'None'], ax=axes[1,6], ci=68, jitter=0.2, dodge=0.2,
#         hue="Violation on", hue_order=['property', 'binding', 'relation'], palette=(COLOR_PROPERTY, COLOR_BINDING, COLOR_RELATION)) #, markers=["o", "^", "s"])
#     g.set_ylabel('')
#     g.set_xlabel('Shared features between objects') 
#     g.legend().set_visible(False)
        
#         # if box_pairs and kind != "point":
#         #     annotator = Annotator(ax, box_pairs, plot=f'{kind}plot', data=df, x=x, hue=hue, col=col, y=y, text_format='star', order=order, hue_order=hue_order) #,), line_offset_to_box=-1
#         #     annotator.configure(test='Mann-Whitney', verbose=False, loc="inside", comparisons_correction="bonferroni", fontsize=12, use_fixed_offset=True).apply_and_annotate()
#         #     # , line_offset=.0001, line_offset_to_group=.0001
    
#     # set ymin to maximize visible variance
#     for ax in axes[0,:]: ax.set_ylim(250)

#     if tight:
#         plt.tight_layout()
#         # if (ncol > 1 or hue is not None) and legend:
#         #     ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=ncol, fontsize=12) # Put the legend out of the figure
#     plt.subplots_adjust(hspace=0.07, wspace=0.07)
#     plt.savefig(out_fn, transparent=False, bbox_inches='tight', dpi=400)
#     plt.close()

#     ## Make legend 
#     # fig, ax = plt.subplots()
#     legendfig, legendax = plt.subplots(figsize=(5,5))
#     patches = []
#     patches.append(mpatches.Patch(edgecolor='k', facecolor='w', label='One object'))
#     patches.append(mpatches.Patch(edgecolor='k', facecolor='k', label='Two objects'))
#     # r = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False) # filler to align legend columns correctly
#     patches.append(mpatches.Patch(color=COLOR_MATCH, label='Match'))
#     patches.append(mpatches.Patch(color=COLOR_MISMATCH, label='Mismatch'))
#     patches.append(mpatches.Patch(color=COLOR_FIRST, label='Mismatch on the first object'))
#     patches.append(mpatches.Patch(color=COLOR_SECOND, label='Mismatch on the second object'))
#     patches.append(mpatches.Patch(color=COLOR_LEFT, label='Mismatch on the object on the left'))
#     patches.append(mpatches.Patch(color=COLOR_RIGHT, label='Mismatch on the object on the right'))
#     patches.append(mpatches.Patch(color=COLOR_PROPERTY, label='Property mismatch'))
#     patches.append(mpatches.Patch(color=COLOR_BINDING, label='Binding mismatch'))
#     patches.append(mpatches.Patch(color=COLOR_RELATION, label='Spatial relation mismatch'))

#     legendfig.legend(handles=patches, loc='center', ncol=6)
#     legendax.axis('off') # hide the axes frame and the x/y labels
#     plt.tight_layout()
#     legendfig.savefig(f"{out_fn[0:-4]}_legend.png", bbox_inches='tight', dpi=400)


def behavior_barplot(df_full, df_1obj, df_2obj, out_fn="tmp.png", min_rt=250, colors=None, tight=True):
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(20, 6), sharey='row', sharex='col')
    sns.set(font_scale = 3)
    COLOR_MATCH = PURE_GREEN
    COLOR_MISMATCH = PURE_RED
    # COLOR_PROP_SHAPE, COLOR_PROP_COLOR = (.9, .5, .5), (.9, .3, .3)
    COLOR_PROPERTY = LIGHT_RED
    COLOR_BINDING, COLOR_RELATION = MEDIUM_RED, DARK_RED
    COLOR_FIRST, COLOR_SECOND = LIGHT_BLUE, DARK_BLUE
    COLOR_LEFT, COLOR_RIGHT = LIGHT_GREY, DARK_GREY
    WIDTH = .385
    # First panel: overall perf for each bock type, split with violation. white facecolor, colored edgecolor
    edgecolors = [COLOR_MATCH]*2 + [COLOR_MISMATCH]*2
    facecolors = [WHITE, COLOR_MATCH, WHITE, COLOR_MISMATCH]
    g = sns.barplot(x='Trial type', y='RT', data=deepcopy(df_full).query("Perf==1"), order=['One object', 'Two objects'],
        hue="Violation", ax=axes[0,0], ci=68, hue_order=("No", "Yes"))
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
        bar.set_width(WIDTH)
    g.set_xlabel('') 
    g.legend().set_visible(False)
    g = sns.barplot(x='Trial type', y='Error rate', data=df_full, order=['One object', 'Two objects'],
        hue="Violation", ax=axes[1,0], ci=68, hue_order=("No", "Yes"))
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
        bar.set_width(WIDTH)
    g.legend().set_visible(False)

    ## Second panel: feature mismatch for both one and two obj
    # edgecolors = [COLOR_PROP_SHAPE, COLOR_PROP_COLOR, COLOR_PROP_SHAPE, COLOR_PROP_COLOR]
    # facecolors = [WHITE, WHITE, COLOR_PROP_SHAPE, COLOR_PROP_COLOR]
    edgecolors = [COLOR_PROPERTY, COLOR_PROPERTY, COLOR_PROPERTY, COLOR_PROPERTY]
    facecolors = [WHITE, WHITE, COLOR_PROPERTY, COLOR_PROPERTY]
    g = sns.barplot(x='Changed property', y='RT', data=deepcopy(df_full).query("Perf==1"), order=['Shape', 'Color'], ax=axes[0,1], ci=68,
        hue="Trial type", hue_order=['One object', 'Two objects'])
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
        bar.set_width(WIDTH)
    g.set_ylabel(''), g.set_xlabel('') 
    g.legend().set_visible(False)
    g = sns.barplot(x='Changed property', y='Error rate', data=df_full, order=['Shape', 'Color'], ax=axes[1,1], ci=68,
        hue="Trial type", hue_order=['One object', 'Two objects'])
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
        bar.set_width(WIDTH)
    g.set_ylabel(''), g.set_xlabel('Mismatch on')
    g.legend().set_visible(False)

    ## Third panel: kind of violation for 2 object trials
    edgecolors = [COLOR_PROPERTY, COLOR_BINDING, COLOR_RELATION]
    facecolors = [COLOR_PROPERTY, COLOR_BINDING, COLOR_RELATION]
    g = sns.barplot(x='Violation on', y='RT', data=df_2obj, order=['property', 'binding', 'relation'], ax=axes[0,2], ci=68, color=COLOR_MISMATCH)
    g.set_ylabel(''), g.set_xlabel('') 
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
    g = sns.barplot(x='Violation on', y='Error rate', data=df_2obj, order=['property', 'binding', 'relation'], ax=axes[1,2], ci=68, color=COLOR_MISMATCH)
    g.set_ylabel(''), g.set_xlabel('Mismatch on')
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)

    ## Fourth panel: ordinal position
    edgecolors = [COLOR_FIRST, COLOR_SECOND]
    facecolors = [COLOR_FIRST, COLOR_SECOND]
    local_df_2obj = deepcopy(df_2obj)
    local_df_2obj["Violation on"] = df_2obj["Violation ordinal position"].apply(lambda x: f"{x} object")
    g = sns.barplot(x='Violation on', y='RT', data=deepcopy(local_df_2obj).query("Perf==1"), order=['First object', 'Second object'], ax=axes[0,3], ci=68, color=COLOR_MISMATCH)
    g.set_ylabel(''), g.set_xlabel('') 
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
    g = sns.barplot(x='Violation on', y='Error rate', data=local_df_2obj, order=['First object', 'Second object'], ax=axes[1,3], ci=68, color=COLOR_MISMATCH)
    g.set_ylabel(''), g.set_xlabel('Mismatch on')
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)

    ## Fifth panel: spatial position
    edgecolors = [COLOR_LEFT, COLOR_RIGHT]
    facecolors = [COLOR_LEFT, COLOR_RIGHT]
    local_df_2obj["Violation on"] = df_2obj["Violation side"].apply(lambda x: f"{x} object")
    g = sns.barplot(x='Violation on', y='RT', data=deepcopy(local_df_2obj).query("Perf==1"), order=['Left object', 'Right object'], ax=axes[0,4], ci=68, color=COLOR_MISMATCH)
    g.set_ylabel(''), g.set_xlabel('') 
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
    g = sns.barplot(x='Violation on', y='Error rate', data=local_df_2obj, order=['Left object', 'Right object'], ax=axes[1,4], ci=68, color=COLOR_MISMATCH)
    g.set_ylabel(''), g.set_xlabel('Mismatch on')
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)

    ## Sixth panel: Shared features between objects
    g = sns.pointplot(x='Sharing', y='RT', hue='Violation', hue_order=['No', 'Yes'], data=deepcopy(df_2obj).query("Perf==1"), 
        order=['Both', 'Shape', 'Color', 'None'], ax=axes[0,5], ci=68, jitter=True, dodge=True, palette=[COLOR_MATCH, COLOR_MISMATCH])
    g.set_ylabel(''), g.set_xlabel('')
    # g.legend().set_visible(False)
    g.legend(prop={"size":12}, loc='lower right')
    g = sns.pointplot(x='Sharing', y='Error rate', hue='Violation', hue_order=['No', 'Yes'], data=df_2obj, 
        order=['Both', 'Shape', 'Color', 'None'], ax=axes[1,5], ci=68, jitter=True, dodge=True, palette=[COLOR_MATCH, COLOR_MISMATCH])
    g.set_ylabel('')
    g.set_xlabel('Shared features between objects') 
    g.legend().set_visible(False)

    ## Seventh panel: Shared features between objects * kind of violation
    g = sns.pointplot(x='Sharing', y='RT', data=deepcopy(df_2obj).query("Perf==1"), order=['Both', 'Shape', 'Color', 'None'], ax=axes[0,6], ci=68, jitter=0.2, dodge=0.2,
        hue="Violation on", hue_order=['property', 'binding', 'relation'], palette=(COLOR_PROPERTY, COLOR_BINDING, COLOR_RELATION)) #, markers=["o", "^", "s"])
    g.set_ylabel(''), g.set_xlabel('')
    g.legend(prop={"size":12}, loc='lower right')
    g = sns.pointplot(x='Sharing', y='Error rate', data=df_2obj, order=['Both', 'Shape', 'Color', 'None'], ax=axes[1,6], ci=68, jitter=0.2, dodge=0.2,
        hue="Violation on", hue_order=['property', 'binding', 'relation'], palette=(COLOR_PROPERTY, COLOR_BINDING, COLOR_RELATION)) #, markers=["o", "^", "s"])
    g.set_ylabel('')
    g.set_xlabel('Shared features between objects') 
    g.legend().set_visible(False)
        
        # if box_pairs and kind != "point":
        #     annotator = Annotator(ax, box_pairs, plot=f'{kind}plot', data=df, x=x, hue=hue, col=col, y=y, text_format='star', order=order, hue_order=hue_order) #,), line_offset_to_box=-1
        #     annotator.configure(test='Mann-Whitney', verbose=False, loc="inside", comparisons_correction="bonferroni", fontsize=12, use_fixed_offset=True).apply_and_annotate()
        #     # , line_offset=.0001, line_offset_to_group=.0001
    
    # set ymin to maximize visible variance
    for ax in axes[0,:]: ax.set_ylim(min_rt)

    if tight:
        plt.tight_layout()
        # if (ncol > 1 or hue is not None) and legend:
        #     ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=ncol, fontsize=12) # Put the legend out of the figure
    plt.subplots_adjust(hspace=0.07, wspace=0.07)
    plt.savefig(out_fn, transparent=False, bbox_inches='tight', dpi=400)
    plt.close()

    ## Make legend 
    # fig, ax = plt.subplots()
    legendfig, legendax = plt.subplots(figsize=(5,5))
    patches = []
    patches.append(mpatches.Patch(edgecolor='k', facecolor='w', label='One object'))
    patches.append(mpatches.Patch(edgecolor='k', facecolor='k', label='Two objects'))
    # r = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False) # filler to align legend columns correctly
    patches.append(mpatches.Patch(color=COLOR_MATCH, label='Match'))
    patches.append(mpatches.Patch(color=COLOR_MISMATCH, label='Mismatch'))
    patches.append(mpatches.Patch(color=COLOR_FIRST, label='Mismatch on the first object'))
    patches.append(mpatches.Patch(color=COLOR_SECOND, label='Mismatch on the second object'))
    patches.append(mpatches.Patch(color=COLOR_LEFT, label='Mismatch on the object on the left'))
    patches.append(mpatches.Patch(color=COLOR_RIGHT, label='Mismatch on the object on the right'))
    patches.append(mpatches.Patch(color=COLOR_PROPERTY, label='Property mismatch'))
    patches.append(mpatches.Patch(color=COLOR_BINDING, label='Binding mismatch'))
    patches.append(mpatches.Patch(color=COLOR_RELATION, label='Spatial relation mismatch'))

    legendfig.legend(handles=patches, loc='center', ncol=6)
    legendax.axis('off') # hide the axes frame and the x/y labels
    plt.tight_layout()
    legendfig.savefig(f"{out_fn[0:-4]}_legend.png", bbox_inches='tight', dpi=400)



def behavior_barplot_humans_and_model(df_full, df_1obj, df_2obj, df_model, out_fn="tmp.png", model_perf='Error rate', rt='RT',
                                      model_min=25, min_rt=250, colors=None, tight=True, redo_legend=False, model_hline=True, ncol_legend=1):
    fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(20, 9), sharey='row', sharex='col')
    sns.set(font_scale = 3)
    COLOR_MATCH = GREEN
    COLOR_MISMATCH = RED
    COLOR_PROP_SHAPE, COLOR_PROP_COLOR = LIGHT_BLUE, DARK_BLUE
    COLOR_PROPERTY = BLUE
    COLOR_BINDING, COLOR_RELATION = BROWN, ORANGE
    COLOR_FIRST, COLOR_SECOND = BLUE, BLUE
    COLOR_LEFT, COLOR_RIGHT = BLUE, BLUE
    WIDTH = .385
    # First panel: overall perf for each bock type, split with violation. white facecolor, colored edgecolor
    edgecolors = [COLOR_MATCH]*2 + [COLOR_MISMATCH]*2
    facecolors = [WHITE, COLOR_MATCH, WHITE, COLOR_MISMATCH]
    g = sns.barplot(x='Trial type', y=rt, data=deepcopy(df_full).query("Perf==1"), order=['One object', 'Two objects'],
        hue="Violation", ax=axes[0,0], ci=68, hue_order=("No", "Yes"))
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
        bar.set_width(WIDTH)
    g.set_xlabel('')
    g.legend().set_visible(False)
    g = sns.barplot(x='Trial type', y='Error rate', data=df_full, order=['One object', 'Two objects'],
        hue="Violation", ax=axes[1,0], ci=68, hue_order=("No", "Yes"))
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
        bar.set_width(WIDTH)
    g.set_xlabel('')
    g.legend().set_visible(False)
    g = sns.barplot(x='Trial type', y=model_perf, data=df_model, order=['One object', 'Two objects'],
        hue="Violation", ax=axes[2,0], ci=68, hue_order=("No", "Yes"))
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
        bar.set_width(WIDTH)
    g.legend().set_visible(False)
    if model_hline: g.axhline(0.5, ls='--', c='grey', zorder=-10)
    g.set_ylabel(f'CLIP {model_perf}')

    ## Second panel: feature mismatch for both one and two obj
    # edgecolors = [COLOR_PROP_SHAPE, COLOR_PROP_COLOR, COLOR_PROP_SHAPE, COLOR_PROP_COLOR]
    # facecolors = [WHITE, WHITE, COLOR_PROP_SHAPE, COLOR_PROP_COLOR]
    edgecolors = [LIGHT_BLUE, DARK_BLUE, LIGHT_BLUE, DARK_BLUE]
    facecolors = [WHITE, WHITE, LIGHT_BLUE, DARK_BLUE]
    g = sns.barplot(x='Changed property', y=rt, data=deepcopy(df_full).query("Perf==1"), order=['Shape', 'Color'], ax=axes[0,1], ci=68,
        hue="Trial type", hue_order=['One object', 'Two objects'])
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
        bar.set_width(WIDTH)
    g.set_ylabel(''), g.set_xlabel('') 
    g.legend().set_visible(False)
    g = sns.barplot(x='Changed property', y='Error rate', data=df_full, order=['Shape', 'Color'], ax=axes[1,1], ci=68,
        hue="Trial type", hue_order=['One object', 'Two objects'])
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
        bar.set_width(WIDTH)
    g.set_ylabel(''), g.set_xlabel('')
    g.legend().set_visible(False)
    g = sns.barplot(x='Changed property', y=model_perf, data=df_model, order=['Shape', 'Color'], ax=axes[2,1], ci=68,
        hue="Trial type", hue_order=['One object', 'Two objects'])
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
        bar.set_width(WIDTH)
    g.set_ylabel(''), g.set_xlabel('Mismatch on')
    if model_hline: g.axhline(0.5, ls='--', c='grey', zorder=-10)
    g.legend().set_visible(False)

    ## Third panel: kind of violation for 2 object trials
    edgecolors = [COLOR_PROPERTY, COLOR_BINDING, COLOR_RELATION]
    facecolors = [COLOR_PROPERTY, COLOR_BINDING, COLOR_RELATION]
    g = sns.barplot(x='Violation on', y=rt, data=df_2obj, order=['Property', 'Binding', 'Relation'], ax=axes[0,2], ci=68, color=COLOR_MISMATCH)
    g.set_ylabel(''), g.set_xlabel('')
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
    g = sns.barplot(x='Violation on', y='Error rate', data=df_2obj, order=['Property', 'Binding', 'Relation'], ax=axes[1,2], ci=68, color=COLOR_MISMATCH)
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
    g.set_ylabel(''), g.set_xlabel('')
    g = sns.barplot(x='Violation on', y=model_perf, data=deepcopy(df_model).query("NbObjects==2"), order=['Property', 'Binding', 'Relation'], ax=axes[2,2], ci=68, color=COLOR_MISMATCH)
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
    g.set_ylabel(''), g.set_xlabel('Mismatch on')
    if model_hline: g.axhline(0.5, ls='--', c='grey', zorder=-10)

    ## Fourth panel: ordinal position
    edgecolors = [COLOR_FIRST, COLOR_SECOND]
    facecolors = [COLOR_FIRST, COLOR_SECOND]
    local_df_2obj = deepcopy(df_2obj)
    local_df_2obj["Violation on"] = df_2obj["Violation ordinal position"].apply(lambda x: f"{x} object")
    g = sns.barplot(x='Violation on', y=rt, data=deepcopy(local_df_2obj).query("Perf==1"), order=['First object', 'Second object'], ax=axes[0,3], ci=68, color=COLOR_MISMATCH)
    g.set_ylabel(''), g.set_xlabel('') 
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
    g = sns.barplot(x='Violation on', y='Error rate', data=local_df_2obj, order=['First object', 'Second object'], ax=axes[1,3], ci=68, color=COLOR_MISMATCH)
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
    g.set_ylabel(''), g.set_xlabel('')
    local_df_model = deepcopy(df_model)
    local_df_model["Violation on"] = local_df_model["property_mismatches_order"].apply(lambda x: f"{x} object")
    g = sns.barplot(x='Violation on', y=model_perf, data=local_df_model, order=['First object', 'Second object'], ax=axes[2,3], ci=68, color=COLOR_MISMATCH)
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
    g.set_ylabel(''), g.set_xlabel('Mismatch on')
    if model_hline: g.axhline(0.5, ls='--', c='grey', zorder=-10)

    ## Fifth panel: spatial position
    edgecolors = [COLOR_LEFT, COLOR_RIGHT]
    facecolors = [COLOR_LEFT, COLOR_RIGHT]
    local_df_2obj = deepcopy(df_2obj)
    local_df_2obj["Violation on"] = local_df_2obj["Violation side"].apply(lambda x: f"{x} object")
    g = sns.barplot(x='Violation on', y=rt, data=deepcopy(local_df_2obj).query("Perf==1"), order=['Left object', 'Right object'], ax=axes[0,4], ci=68, color=COLOR_MISMATCH)
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
    g.set_ylabel(''), g.set_xlabel('') 
    g = sns.barplot(x='Violation on', y='Error rate', data=local_df_2obj, order=['Left object', 'Right object'], ax=axes[1,4], ci=68, color=COLOR_MISMATCH)
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
    g.set_ylabel(''), g.set_xlabel('')
    local_df_model = deepcopy(df_model)
    local_df_model["Violation on"] = local_df_model["property_mismatches_side"].apply(lambda x: f"{x} object")
    g = sns.barplot(x='Violation on', y=model_perf, data=local_df_model, order=['Left object', 'Right object'], ax=axes[2,4], ci=68, color=COLOR_MISMATCH)
    for bar, edgecolor, facecolor in zip(g.patches, edgecolors, facecolors):
        bar.set_edgecolor(edgecolor)
        bar.set_facecolor(facecolor)
    g.set_ylabel(''), g.set_xlabel('Mismatch on')
    if model_hline: g.axhline(0.5, ls='--', c='grey', zorder=-10)


    ## Sixth panel: Shared features between objects
    g = sns.pointplot(x='Sharing', y=rt, hue='Violation', hue_order=['No', 'Yes'], data=deepcopy(df_2obj).query("Perf==1"), 
        order=['Both', 'Shape', 'Color', 'None'], ax=axes[0,5], ci=68, jitter=True, dodge=True, palette=[COLOR_MATCH, COLOR_MISMATCH])
    g.set_ylabel(''), g.set_xlabel('')
    # g.legend().set_visible(False)
    g.legend(prop={"size":12}, loc='lower right')
    g = sns.pointplot(x='Sharing', y='Error rate', hue='Violation', hue_order=['No', 'Yes'], data=df_2obj, 
        order=['Both', 'Shape', 'Color', 'None'], ax=axes[1,5], ci=68, jitter=True, dodge=True, palette=[COLOR_MATCH, COLOR_MISMATCH])
    g.set_ylabel(''), g.set_xlabel('')
    g.legend().set_visible(False)
    g = sns.pointplot(x='Sharing', y=model_perf, hue='Violation', hue_order=['No', 'Yes'], data=df_model, 
        order=['Both', 'Shape', 'Color', 'None'], ax=axes[2,5], ci=68, jitter=True, dodge=True, palette=[COLOR_MATCH, COLOR_MISMATCH])
    g.set_ylabel(''), g.set_xlabel('Shared features between objects') 
    g.legend().set_visible(False)
    if model_hline: g.axhline(0.5, ls='--', c='grey', zorder=-10)

    ## Seventh panel: Shared features between objects * kind of violation
    g = sns.pointplot(x='Sharing', y=rt, data=deepcopy(df_2obj).query("Perf==1"), order=['Both', 'Shape', 'Color', 'None'], ax=axes[0,6], ci=68, jitter=0.2, dodge=0.2,
        hue="Violation on detailed", hue_order=['Shape', 'Color', 'Binding', 'Relation'], palette=(COLOR_PROP_SHAPE, COLOR_PROP_COLOR, COLOR_BINDING, COLOR_RELATION)) #, markers=["o", "^", "s"])
    g.set_ylabel(''), g.set_xlabel('')
    g.legend(prop={"size":12}, loc='lower right')
    g = sns.pointplot(x='Sharing', y='Error rate', data=df_2obj, order=['Both', 'Shape', 'Color', 'None'], ax=axes[1,6], ci=68, jitter=0.2, dodge=0.2,
        hue="Violation on detailed", hue_order=['Shape', 'Color', 'Binding', 'Relation'], palette=(COLOR_PROP_SHAPE, COLOR_PROP_COLOR, COLOR_BINDING, COLOR_RELATION)) #, markers=["o", "^", "s"])
    g.set_ylabel(''), g.set_xlabel('')
    g.legend().set_visible(False)
    g = sns.pointplot(x='Sharing', y=model_perf, data=df_model, order=['Both', 'Shape', 'Color', 'None'], ax=axes[2,6], ci=68, jitter=0.2, dodge=0.2,
        hue="Violation on detailed", hue_order=['Shape', 'Color', 'Binding', 'Relation'], palette=(COLOR_PROP_SHAPE, COLOR_PROP_COLOR, COLOR_BINDING, COLOR_RELATION)) #, markers=["o", "^", "s"])
    g.set_ylabel(''), g.set_xlabel('Shared features between objects') 
    g.legend().set_visible(False)
    if model_hline: g.axhline(0.5, ls='--', c='grey', zorder=-10)
     
    # set ymin to maximize visible variance
    for ax in axes[0,:]: ax.set_ylim(min_rt)

    if model_perf and model_perf == "similarity":
        for ax in axes[2,:]: ax.set_ylim(model_min)

    if tight:
        plt.tight_layout()
        # if (ncol > 1 or hue is not None) and legend:
        #     ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=ncol, fontsize=12) # Put the legend out of the figure
    plt.subplots_adjust(hspace=0.07, wspace=0.07)
    plt.savefig(out_fn, transparent=False, bbox_inches='tight', dpi=400)
    plt.close()

    ## Make legend 
    if redo_legend:
        legendfig, legendax = plt.subplots(figsize=(5,5))
        patches = []
        patches.append(mpatches.Patch(edgecolor='k', facecolor='w', label='One object'))
        patches.append(mpatches.Patch(edgecolor='k', facecolor='k', label='Two objects'))
        # r = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False) # filler to align legend columns correctly
        patches.append(mpatches.Patch(color=COLOR_MATCH, label='Match'))
        patches.append(mpatches.Patch(color=COLOR_MISMATCH, label='Mismatch'))
        # patches.append(mpatches.Patch(color=COLOR_FIRST, label='Mismatch on the first object'))
        # patches.append(mpatches.Patch(color=COLOR_SECOND, label='Mismatch on the second object'))
        # patches.append(mpatches.Patch(color=COLOR_LEFT, label='Mismatch on the object on the left'))
        # patches.append(mpatches.Patch(color=COLOR_RIGHT, label='Mismatch on the object on the right'))
        patches.append(mpatches.Patch(color=COLOR_PROP_SHAPE, label='Property mismatch on shape'))
        patches.append(mpatches.Patch(color=COLOR_PROP_COLOR, label='Property mismatch on color'))
        patches.append(mpatches.Patch(color=COLOR_PROPERTY, label='Property mismatch'))
        patches.append(mpatches.Patch(color=COLOR_BINDING, label='Binding mismatch'))
        patches.append(mpatches.Patch(color=COLOR_RELATION, label='Spatial relation mismatch'))

        legendfig.legend(handles=patches, loc='center', ncol=ncol_legend)
        legendax.axis('off') # hide the axes frame and the x/y labels
        plt.tight_layout()
        legendfig.savefig(f"{out_fn[0:-4]}_legend_{ncol_legend}.png", bbox_inches='tight', dpi=400)
        plt.close()

