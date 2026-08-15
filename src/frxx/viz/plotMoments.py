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
import matplotlib.text as mtext

def elementMultuplier(scaleRate, width, aspectRatioWH):
	return 1 + scaleRate*(max(width, width/aspectRatioWH)/2-1) #width multiplier

def plotPPI(
	data: npt.NDArray,
	title: str, units: str,
	rangesKM: npt.NDArray, 
	azimuths: npt.NDArray, 
	elevation: Number, 
	width: Number = 2,
	aspectRatioWH: float = np.sqrt(2),
	dpi: int = 300,
	xlim: Tuple[Number, Number] | None = None, 
	yCenter: Number | None = None,
	cmap: Union[str, Colormap] = 'Carbone42',
	clims: Tuple[Number, Number, int] | None = None,
	backend: bool = True
):
	import cmweather
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
	lm = elementMultuplier(scaleRate, width, aspectRatioWH)

	if backend:
		from matplotlib import pyplot as plt
		fig = plt.figure(figsize=(width, width*aspectRatio), dpi=dpi)
	else:
		fig = Figure(figsize=(width, width*aspectRatio), dpi=dpi)
	ax = fig.add_axes((0, 0, 1, 1))
	
	from pyart.core.transforms import antenna_vectors_to_cartesian
	xx, yy, _ = antenna_vectors_to_cartesian(
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
	dy = aspectRatio * dx
	ylim = (yCenter - 0.5*dy, yCenter + 0.5*dy)
	
	plot = ax.pcolormesh(
		xx, yy, data, 
		cmap=cmap, vmin=vmin, vmax=vmax,
		zorder = 0
	)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.axis('off')
	
	textBorderWidth = 0.5*lm

	#cbarAx = inset_axes(ax, width=2*0.1*lm, height=aspectRatio*width*0.85, loc='center right', borderpad=0.25)
	cbarAx = ax.inset_axes(bounds=(0.9, 0.075, 0.075, 0.85), zorder = 100)
	cb = fig.colorbar(
		plot,
		cax = cbarAx,
		ticks = ticks,
		extend="both",
		orientation='vertical',
	)
	cbarAx.tick_params(labelsize=4*lm, direction='in', length = 2, pad=0)
	labels = [l for l in cbarAx.get_yticklabels() if l.get_text()]
	for i, label in enumerate(labels):
		label.set_color('black')
		label.set_path_effects([pe.withStroke(linewidth=textBorderWidth, foreground='white')])
		label._frxxBaseFontSize = 4
		
		_, y = label.get_position()
		label.set_position((0.8, y))
		label.set_ha('right')
		
		if i == 0:
			label.set_va('bottom')
		elif i == len(labels) - 1:
			label.set_va('top')
	
	unitsText = cbarAx.text(
		0.5, 1.06, units, transform=cbarAx.transAxes,
		va='bottom', ha='center', fontsize=5*lm, color='darkblue'
	)
	unitsText._frxxBaseFontSize = 5
	unitsText.set_path_effects([
		pe.withStroke(linewidth=textBorderWidth*2, foreground='white')
	])
	unitsText.set_zorder(10)

	fieldTxt = ax.text(
		0.02, 0.98, title, transform=ax.transAxes,
		ha='left', va="top", fontsize=8*lm, color='darkblue'
	)
	fieldTxt._frxxBaseFontSize = 8
	fieldTxt.set_path_effects([
		pe.withStroke(linewidth=textBorderWidth*2, foreground='white')
	])
	fieldTxt.set_zorder(10)

	ax.set_aspect('equal')
	
	return fig, ax, plot, cb, (xx,yy)

def updatePPIAxesText(fig, ax, plot, cb, width, height):
	scaleRate = 0.5
	lm = elementMultuplier(scaleRate, width, width/height)
	for text in fig.findobj(mtext.Text):
		if not hasattr(text, '_frxxBaseFontSize'):
			continue
		text.set_fontsize(text._frxxBaseFontSize * lm)
	fig.canvas.draw_idle()