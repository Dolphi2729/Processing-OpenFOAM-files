# make sure to define PYTHONPATH
# export
# PYTHONPATH=/media/old_home/palkine/Dropbox/Egor-Rustam/PAPERS/2021-Energies/appearance/

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
import os
import latex_document_parameters as latex
import locale

def matplotlib_header():  # Matplotlib settings

    # locale.setlocale(locale.LC_NUMERIC, ('ru_RU', 'UTF-8')) # Set locale
    plt.rcParams.update({
        # 'axes.formatter.use_locale' = True,
        'mathtext.fontset' : 'cm',
        'font.family': 'serif',
        'font.serif' : str(latex.MATH_FONT),
        'font.sans-serif' : 'Computer Modern Sans serif',
        #'font.size' : str(latex.MATH_FONTSIZE),  # pt
        'text.usetex' : True,
        'xtick.minor.visible' : True,
        'ytick.minor.visible' : True,
        # distance from axis to Mticks label
        #'xaxis.labellocation' : 'left'
        #'yaxis.labellocation' : 'bottom'
        'xtick.major.pad' : 4.20,  # empirical from inkscape
        'ytick.major.pad' : 1.81,  # empirical from inkscape
        'xtick.major.size' : 6,
        'xtick.minor.size' : 4,
        'ytick.major.size' : 6,
        'ytick.minor.size' : 4.,
        'xtick.major.width' : 0.375,
        'xtick.minor.width' : 0.25,
        'ytick.major.width' : 0.375,
        'ytick.minor.width' : 0.25,
        'xtick.direction' : 'in',
        'ytick.direction' : 'in',
        # 'figure.figsize'] # set by set_fig_size()
        'savefig.dpi' : 300,
        'figure.dpi' : 300,
        'image.cmap' : 'jet',
        'legend.frameon' : True,
        'legend.fancybox' : False,
        'legend.framealpha' : 1,
        'legend.edgecolor' : 'k',
        'legend.labelspacing' : 0,
        'legend.handlelength' : 10,
        'legend.handletextpad' : 0.25,  # x pad between marker & label
        'legend.columnspacing' : 0.25,  # y pad labels,
        'legend.borderaxespad' : 0.0,  # pad between corner of bbox_to_anchor
        'legend.borderpad' : 0.1,  # whitespace inside the legend border
        'legend.numpoints' : 3,
        'lines.linewidth' : 0.75,
        'lines.markeredgewidth' : 0.5,
        'lines.markersize' : 4.,
        'axes.linewidth' : 0.2,
        'patch.linewidth' : 0.2,
         # Extra latex packages:
        'text.latex.preamble': '\n'.join([
           r'\usepackage[T1]{fontenc}',
#          '\\usepackage{unicode-math}',   # unicode math setup
           r'\usepackage[utf8]{inputenc}',
           r'\usepackage{mathptmx}',
           r'\usepackage[english, russian]{babel}'
        ])
    })


def linecolor(i):  # Color styles
    const_array = \
        ['none', 'tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:cyan']
    return const_array[i]


def linestyle(i):  # Line styles
    const_array = \
        ['none', 'solid', 'dashed', 'dashdot', (0, (3, 1, 1, 1, 1, 1)), 'dotted']
    return const_array[i]


def set_fig_size(width, fraction=1):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches

    Usage:
    put '\\showthe\\textwidth' in your latex document
    it outputs textwidth
    You should use it as input parameter for this function
    '\\showthe\\font' returns used font
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches [const]
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # golden_ratio = (5**.5 - 1) / 2.
    golden_ratio = 1. / latex.ASPECT_RATIO

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt

    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
