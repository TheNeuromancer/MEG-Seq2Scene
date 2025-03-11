# -*- coding: utf-8 -*-
"""Utility functions for plotting M/EEG data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Clemens Brunner <clemens.brunner@gmail.com>
#          Daniel McCloy <dan@mccloy.info>
#
# License: Simplified BSD
import mne 
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
import difflib
import webbrowser
import tempfile
import math
import numpy as np
from copy import deepcopy
from distutils.version import LooseVersion
import warnings
from datetime import datetime
import matplotlib

from mne.defaults import _handle_default
from mne.fixes import _get_args
from mne.io import show_fiff, Info
from mne.io.constants import FIFF
from mne.io.pick import (channel_type, channel_indices_by_type, pick_channels,
                       _pick_data_channels, _DATA_CH_TYPES_SPLIT,
                       _DATA_CH_TYPES_ORDER_DEFAULT, _VALID_CHANNEL_TYPES,
                       pick_info, _picks_by_type, pick_channels_cov,
                       _contains_ch_type)
from mne.io.meas_info import create_info
from mne.rank import compute_rank
from mne.io.proj import setup_proj
from mne.utils import (verbose, get_config, warn, _check_ch_locs, _check_option,
                     logger, fill_doc, _pl, _check_sphere, _ensure_int,
                     _validate_type)
from mne.transforms import apply_trans


_channel_type_prettyprint = {'eeg': "EEG channel", 'grad': "Gradiometer",
                             'mag': "Magnetometer", 'seeg': "sEEG channel",
                             'dbs': "DBS channel", 'eog': "EOG channel",
                             'ecg': "ECG sensor", 'emg': "EMG sensor",
                             'ecog': "ECoG channel",
                             'misc': "miscellaneous sensor"}


def plot_ch_scores(info, scores, cmap, kind='topomap', ch_type=None, title=None,
                 show_names=False, ch_groups=None, to_sphere=True, axes=None,
                 block=False, show=True, sphere=None, verbose=None, norm=None, sensor_size=155):
    """Plot sensors positions.
    Parameters
    ----------
    %(info_not_none)s
    kind : str
        Whether to plot the sensors as 3d, topomap or as an interactive
        sensor selection dialog. Available options 'topomap', '3d', 'select'.
        If 'select', a set of channels can be selected interactively by using
        lasso selector or clicking while holding control key. The selected
        channels are returned along with the figure instance. Defaults to
        'topomap'.
    ch_type : None | str
        The channel type to plot. Available options 'mag', 'grad', 'eeg',
        'seeg', 'dbs', 'ecog', 'all'. If ``'all'``, all the available mag,
        grad, eeg, seeg, dbs and ecog channels are plotted. If None (default),
        then channels are chosen in the order given above.
    title : str | None
        Title for the figure. If None (default), equals to
        ``'Sensor positions (%%s)' %% ch_type``.
    show_names : bool | array of str
        Whether to display all channel names. If an array, only the channel
        names in the array are shown. Defaults to False.
    ch_groups : 'position' | array of shape (n_ch_groups, n_picks) | None
        Channel groups for coloring the sensors. If None (default), default
        coloring scheme is used. If 'position', the sensors are divided
        into 8 regions. See ``order`` kwarg of :func:`mne.viz.plot_raw`. If
        array, the channels are divided by picks given in the array.
        .. versionadded:: 0.13.0
    to_sphere : bool
        Whether to project the 3d locations to a sphere. When False, the
        sensor array appears similar as to looking downwards straight above the
        subject's head. Has no effect when kind='3d'. Defaults to True.
        .. versionadded:: 0.14.0
    axes : instance of Axes | instance of Axes3D | None
        Axes to draw the sensors to. If ``kind='3d'``, axes must be an instance
        of Axes3D. If None (default), a new axes will be created.
        .. versionadded:: 0.13.0
    block : bool
        Whether to halt program execution until the figure is closed. Defaults
        to False.
        .. versionadded:: 0.13.0
    show : bool
        Show figure if True. Defaults to True.
    %(topomap_sphere_auto)s
    %(verbose)s
    Returns
    -------
    fig : instance of Figure
        Figure containing the sensor topography.
    selection : list
        A list of selected channels. Only returned if ``kind=='select'``.
    See Also
    --------
    mne.viz.plot_layout
    Notes
    -----
    This function plots the sensor locations from the info structure using
    matplotlib. For drawing the sensors using mayavi see
    :func:`mne.viz.plot_alignment`.
    .. versionadded:: 0.12.0
    """
    # from .evoked import _rgb
    # _check_option('kind', kind, ['topomap', '3d', 'select'])
    if not isinstance(info, Info):
        raise TypeError('info must be an instance of Info not %s' % type(info))
    ch_indices = channel_indices_by_type(info)
    print(ch_indices)
    allowed_types = _DATA_CH_TYPES_SPLIT
    if ch_type is None:
        for this_type in allowed_types:
            if _contains_ch_type(info, this_type):
                ch_type = this_type
                break
        picks = ch_indices[ch_type]
    elif ch_type == 'all':
        picks = list()
        for this_type in allowed_types:
            picks += ch_indices[this_type]
    elif ch_type in allowed_types:
        picks = ch_indices[ch_type]
    else:
        raise ValueError("ch_type must be one of %s not %s!" % (allowed_types,
                                                                ch_type))

    if len(picks) == 0:
        raise ValueError('Could not find any channels of type %s.' % ch_type)

    chs = [info['chs'][pick] for pick in picks]
    # if not _check_ch_locs(chs):
    if not _check_ch_locs(info): # change by TD 14/06/2022
        raise RuntimeError('No valid channel positions found')
    dev_head_t = info['dev_head_t']
    pos = np.empty((len(chs), 3))
    for ci, ch in enumerate(chs):
        pos[ci] = ch['loc'][:3]
        if ch['coord_frame'] == FIFF.FIFFV_COORD_DEVICE:
            if dev_head_t is None:
                warn('dev_head_t is None, transforming MEG sensors to head '
                     'coordinate frame using identity transform')
                dev_head_t = np.eye(4)
            pos[ci] = apply_trans(dev_head_t, pos[ci])
    del dev_head_t

    ch_names = np.array([ch['ch_name'] for ch in chs])
    bads = [idx for idx, name in enumerate(ch_names) if name in info['bads']]
    if ch_groups is None:
        def_colors = _handle_default('color')
        colors = ['red' if i in bads else def_colors[channel_type(info, pick)]
                  for i, pick in enumerate(picks)]
    else:
        if ch_groups in ['position', 'selection']:
            # Avoid circular import
            from ..channels import (read_vectorview_selection, _SELECTIONS,
                                    _EEG_SELECTIONS, _divide_to_regions)

            if ch_groups == 'position':
                ch_groups = _divide_to_regions(info, add_stim=False)
                ch_groups = list(ch_groups.values())
            else:
                ch_groups, color_vals = list(), list()
                for selection in _SELECTIONS + _EEG_SELECTIONS:
                    channels = pick_channels(
                        info['ch_names'],
                        read_vectorview_selection(selection, info=info))
                    ch_groups.append(channels)
            color_vals = np.ones((len(ch_groups), 4))
            for idx, ch_group in enumerate(ch_groups):
                color_picks = [np.where(picks == ch)[0][0] for ch in ch_group
                               if ch in picks]
                if len(color_picks) == 0:
                    continue
                x, y, z = pos[color_picks].T
                color = np.mean(_rgb(x, y, z), axis=0)
                color_vals[idx, :3] = color  # mean of spatial color
        else:
            import matplotlib.pyplot as plt
            colors = np.linspace(0, 1, len(ch_groups))
            color_vals = [plt.cm.jet(colors[i]) for i in range(len(ch_groups))]
        if not isinstance(ch_groups, (np.ndarray, list)):
            raise ValueError("ch_groups must be None, 'position', "
                             "'selection', or an array. Got %s." % ch_groups)
        colors = np.zeros((len(picks), 4))
        for pick_idx, pick in enumerate(picks):
            for ind, value in enumerate(ch_groups):
                if pick in value:
                    colors[pick_idx] = color_vals[ind]
                    break
    ## custom colors
    if norm:
        colors = [cmap(norm(score)) for score in scores]
    else:    
        colors = [cmap(score) for score in scores]
    title = 'Sensor positions (%s)' % ch_type if title is None else title
    fig = _plot_sensors(pos, info, picks, colors, bads, ch_names, title,
                        show_names, axes, show, kind, block,
                        to_sphere, sphere, sensor_size)
    if kind == 'select':
        return fig, fig.lasso.selection
    return fig


def _plot_sensors(pos, info, picks, colors, bads, ch_names, title, show_names,
                  ax, show, kind, block, to_sphere, sphere, sensor_size):
    """Plot sensors."""
    from matplotlib import rcParams
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 analysis:ignore
    from mne.viz.topomap import _get_pos_outlines, _draw_outlines
    sphere = _check_sphere(sphere, info)

    edgecolors = np.repeat(rcParams['axes.edgecolor'], len(colors))
    edgecolors[bads] = 'red'
    axes_was_none = ax is None
    if axes_was_none:
        subplot_kw = dict()
        if kind == '3d':
            subplot_kw.update(projection='3d')
        fig, ax = plt.subplots(
            1, figsize=(max(rcParams['figure.figsize']),) * 2,
            subplot_kw=subplot_kw)
    else:
        fig = ax.get_figure()

    if kind == '3d':
        ax.text(0, 0, 0, '', zorder=1)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], picker=True, c=colors,
                   s=75, edgecolor=edgecolors, linewidth=2)

        ax.azim = 90
        ax.elev = 0
        ax.xaxis.set_label_text('x (m)')
        ax.yaxis.set_label_text('y (m)')
        ax.zaxis.set_label_text('z (m)')
    else:  # kind in 'select', 'topomap'
        ax.text(0, 0, '', zorder=1)

        pos, outlines = _get_pos_outlines(info, picks, sphere,
                                          to_sphere=to_sphere)
        _draw_outlines(ax, outlines)
        pts = ax.scatter(pos[:, 0], pos[:, 1], picker=True, clip_on=False,
                         c=colors, edgecolors=edgecolors, s=sensor_size, lw=.5)
        if kind == 'select':
            fig.lasso = SelectFromCollection(ax, pts, ch_names)
        else:
            fig.lasso = None

        # Equal aspect for 3D looks bad, so only use for 2D
        ax.set(aspect='equal')
        if axes_was_none:  # we'll show the plot title as the window title
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        ax.axis("off")  # remove border around figure
    del sphere

    connect_picker = True
    if show_names:
        if isinstance(show_names, (list, np.ndarray)):  # only given channels
            indices = [list(ch_names).index(name) for name in show_names]
        else:  # all channels
            indices = range(len(pos))
        for idx in indices:
            this_pos = pos[idx]
            if kind == '3d':
                ax.text(this_pos[0], this_pos[1], this_pos[2], ch_names[idx])
            else:
                ax.text(this_pos[0] + 0.0025, this_pos[1], ch_names[idx],
                        ha='left', va='center')
        connect_picker = (kind == 'select')
    if connect_picker:
        picker = partial(_onpick_sensor, fig=fig, ax=ax, pos=pos,
                         ch_names=ch_names, show_names=show_names)
        fig.canvas.mpl_connect('pick_event', picker)
    if axes_was_none:
        _set_window_title(fig, title)
    closed = partial(_close_event, fig=fig)
    fig.canvas.mpl_connect('close_event', closed)
    plt_show(show, block=block)
    return fig


def _onpick_sensor(event, fig, ax, pos, ch_names, show_names):
    """Pick a channel in plot_sensors."""
    if event.mouseevent.inaxes != ax:
        return

    if event.mouseevent.key == 'control' and fig.lasso is not None:
        for ind in event.ind:
            fig.lasso.select_one(ind)

        return
    if show_names:
        return  # channel names already visible
    ind = event.ind[0]  # Just take the first sensor.
    ch_name = ch_names[ind]

    this_pos = pos[ind]

    # XXX: Bug in matplotlib won't allow setting the position of existing
    # text item, so we create a new one.
    ax.texts.pop(0)
    if len(this_pos) == 3:
        ax.text(this_pos[0], this_pos[1], this_pos[2], ch_name)
    else:
        ax.text(this_pos[0], this_pos[1], ch_name)
    fig.canvas.draw()


def _set_window_title(fig, title):
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(title)


def _close_event(event, fig):
    """Listen for sensor plotter close event."""
    if getattr(fig, 'lasso', None) is not None:
        fig.lasso.disconnect()


def plt_show(show=True, fig=None, **kwargs):
    """Show a figure while suppressing warnings.
    Parameters
    ----------
    show : bool
        Show the figure.
    fig : instance of Figure | None
        If non-None, use fig.show().
    **kwargs : dict
        Extra arguments for :func:`matplotlib.pyplot.show`.
    """
    from matplotlib import get_backend
    import matplotlib.pyplot as plt
    if show and get_backend() != 'agg':
        (fig or plt).show(**kwargs)


if __name__ == "main":
    info = mne.create_info(200, sfreq=100)
    plot_sensors(info, ch_type='mag')