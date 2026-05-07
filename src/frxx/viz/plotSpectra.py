from matplotlib.figure import Figure
import numpy as np

from typing import Union, Tuple, cast
Number = Union[int, float]

import numpy as np
import numpy.typing as npt

from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as pe

from ..utils.coordConvert import beamHeightWithRadarHeight

cacheKey = None
cachedHt = None

def plotRangeDoppler(
    data: npt.NDArray,
    title: str, units: str,
    rangesKM: npt.NDArray, 
    velMS: npt.NDArray,
    az: Number, el: Number, radarHt: Number, lat: Number, lon: Number,
    width: Number = 2,
    aspectRatioWH: float = np.sqrt(2),
    cmap: Union[str, Colormap] = 'pyart_Carbone42',
    clims: Tuple[Number, Number, int] | None = None,
    backend: bool = True
):
    #import pyart
    if not (clims is None):
        vmin, vmax, ticknum = clims
        ticks = np.linspace(vmin, vmax, ticknum)
        oom = np.log10(np.abs(ticks).max())
        rounding = 0 if oom >= 1 else  -(round(oom)-1)
        ticks = np.round(ticks, rounding)
    else:
        vmin=None
        vmax=None
        ticks = None
        
    if width < 2:
        raise ValueError("Minimum width of 2 needed for component space.")

    aspectRatio = 1/aspectRatioWH
    scaleRate = 0.5
    lm = 1 + scaleRate*(max(width, aspectRatio*width)/2-1) #width multiplier

    if backend:
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(width, width*aspectRatio), dpi=300)
    else:
        fig = Figure(figsize=(width, width*aspectRatio), dpi=300)
        
    axFrac = 0.75
    start = (1-axFrac)/2
    ax = fig.add_axes((start, start, axFrac, start+axFrac))

    plot = ax.pcolormesh(
        velMS, rangesKM, data, 
        cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.axvline(x=0, color='grey', linestyle=':')
    ax.set_xlabel('Doppler velocity (m/s)', size=4*lm, labelpad=1)
    ax.set_ylabel('Range (km)', size=4*lm, labelpad=0)
    ax.tick_params(axis='both', labelsize=4*lm, length=2, direction='in', pad=1)
    
    xl, xr = ax.get_xlim()
    cbarBuffer = 0.125
    xr += (xr-xl)*cbarBuffer
    ax.set_xlim((xl, xr))

    ax2 = ax.twinx()
    rTicks = np.array(ax.get_yticks(), dtype=np.float64)
    
    global cacheKey, cachedHt
    localCacheKey = (
        tuple(np.round(rTicks).astype(np.int32)),
        round(az, 2),
        round(el, 2),
        round(radarHt, 1),
        round(lat, 3),
        round(lon, 3)
    )
    if localCacheKey == cacheKey:
        htTicks = cachedHt
    else:
        htTicks = beamHeightWithRadarHeight(
            rTicks, az, el, radarHt,
            lat, lon
        )/1000
        cacheKey = localCacheKey
        cachedHt = htTicks
    
    ax2.tick_params(axis='both', labelsize=4*lm, length=2, direction='in', pad=1)
    if np.mean(htTicks) < 0.1:
        ax2.set_yticks(rTicks, np.round(htTicks*1000, decimals=2))
        ax2.set_ylim(ax.get_ylim())
        ax2.set_ylabel('Beam Height AGL (m)', size=4*lm, labelpad=1)
    else:
        ax2.set_yticks(rTicks, np.round(htTicks, decimals=2))
        ax2.set_ylim(ax.get_ylim())
        ax2.set_ylabel('Beam Height AGL (km)', size=4*lm, labelpad=1)
    
    textBorderWidth = 0.5*lm
    
    cbarAx = ax.inset_axes(bounds=(0.875, 0.075, 0.1, 0.85))
    cb = fig.colorbar(
        plot,
        cax = cbarAx,
        ticks = ticks,
        extend="both",
        orientation='vertical'
    )
    cbarAx.tick_params(labelsize=4*lm, direction='in', length = 2, pad=0)
    labels = [l for l in cbarAx.get_yticklabels() if l.get_text()]
    for i, label in enumerate(labels):
        label.set_color('black')
        label.set_path_effects([pe.withStroke(linewidth=textBorderWidth, foreground='white')])
        
        _, y = label.get_position()
        label.set_position((0.8, y))
        label.set_ha('right')
        
        if i == 0:
            label.set_va('bottom')
        elif i == len(labels) - 1:
            label.set_va('top')
    
    unitsText = cbarAx.text(0.5, 1.06, units, transform=cbarAx.transAxes,
                va='bottom', ha='center', fontsize=5*lm, color='darkblue')
    unitsText.set_path_effects([
        pe.withStroke(linewidth=textBorderWidth, foreground='white')
    ])

    fieldTxt = ax.text(
        0.02, 0.98, title, transform=ax.transAxes,
        ha='left', va="top", fontsize=8*lm, color='darkblue'
    )
    fieldTxt.set_path_effects([
        pe.withStroke(linewidth=textBorderWidth, foreground='white')
    ])
    fieldTxt.set_zorder(10)
    
    return fig, ax, plot, cb