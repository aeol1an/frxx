from matplotlib.figure import Figure
import numpy as np

import xarray as xr

from typing import Union, Tuple, List, Sequence, cast
Number = Union[int, float]

import numpy as np
import numpy.typing as npt

import pyart

from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as pe


def _plotPPI(
    data: npt.NDArray,
    title: str, units: str,
    rangesKM: npt.NDArray, 
    azimuths: npt.NDArray, 
    elevation: Number, 
    vertical: bool = False,
    xlim: Tuple[Number, Number] | None = None, 
    yCenter: Number | None = None,
    cmap: Union[str, Colormap] = 'pyart_Carbone42',
    clims: Tuple[Number, Number, int] | None = None
):
    if not (clims is None):
        vmin, vmax, ticknum = clims
        ticks = np.round(np.linspace(vmin, vmax, ticknum), 2)
    else:
        vmin=None
        vmax=None
        ticks = None

    a = np.sqrt(2)
    if vertical:
        aspectRatio = a
    else:
        aspectRatio = 1/a

    width = 2

    fig = Figure(figsize=(width, width*aspectRatio), dpi=300)
    ax = fig.add_axes((0, 0, 1, 1))
    
    xx, yy, _ = pyart.core.transforms.antenna_vectors_to_cartesian(
        ranges=rangesKM,
        azimuths=azimuths,
        elevations=[elevation],
        edges=False
    )

    if (xlim is None) != (yCenter is None):
        raise ValueError("xlim and yCenter should be set together.")
    if xlim is None:
        xlim = (np.min(xx), np.max(xx))
        yCenter = np.mean(yy)
    yCenter = cast(Number, yCenter)
    dx = xlim[1] - xlim[0]
    dy = 2/3 * dx
    ylim = (yCenter - 0.5*dy, yCenter + 0.5*dy)
    
    plot = ax.pcolormesh(
        xx, yy, data, 
        cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    
    textBorderWidth = 0.5

    cbarAx = inset_axes(ax, width="90%", height="10%", loc='lower center', borderpad=0.25)
    fig.colorbar(
        plot,
        cax = cbarAx,
        ticks = ticks,
        extend="both",
        orientation='horizontal'
    )
    cbarAx.tick_params(labelsize=5, direction='in', pad=-7.5, length = 2)
    labels = [l for l in cbarAx.get_xticklabels() if l.get_text()]
    for i, label in enumerate(labels):
        label.set_color('black')
        label.set_path_effects([pe.withStroke(linewidth=textBorderWidth, foreground='white')])
        
        if i == 0:
            label.set_ha('left')  # left-align first label
        elif i == len(labels) - 1:
            label.set_ha('right')  # right-align last label
    
    unitsText = ax.text(0.01, 0.075, units, transform=ax.transAxes,
                va='center', ha='left', fontsize=5, rotation=90, color='darkblue')
    unitsText.set_path_effects([
        pe.withStroke(linewidth=textBorderWidth, foreground='white')
    ])

    ax.set_title(title, size=8, y=0.85, ha='left', x=0.02, color='darkblue')
    ax.title.set_path_effects([
        pe.withStroke(linewidth=textBorderWidth, foreground='white')
    ])
    ax.title.set_zorder(10)
    
    return fig, ax