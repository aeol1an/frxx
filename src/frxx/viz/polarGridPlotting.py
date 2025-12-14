import numpy as np
from matplotlib import pyplot as plt
from ..misc.coordConvert import cart2polar

def rangeRings(ax = None, rint = 3, maxR = 30, xlims = (-30, 30), ylims = (-30, 30), n = 100):
    if ax is None:
        ax = plt.gca()

    rings = np.arange(rint,maxR+rint,rint)

    xx, yy = np.meshgrid(np.linspace(-maxR-rint, maxR+rint, n), np.linspace(-maxR-rint, maxR+rint, n))

    mask = (xx >= xlims[0]) & (xx <= xlims[1]) & (yy >= ylims[0]) & (yy <= ylims[1])

    rows, cols = np.where(mask)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    xx = xx[row_min-1:row_max+2, col_min-1:col_max+2]
    yy = yy[row_min-1:row_max+2, col_min-1:col_max+2]
    rr, _ = cart2polar(xx, yy)

    contour  = ax.contour(xx, yy, rr, rings, colors='k')
    ax.clabel(contour, inline=True)

def azimuthSpiderweb(ax = None, azint = 5, maxR = 30, xlims = (-30, 30), ylims = (-30, 30), n = 500):
    if ax is None:
        ax = plt.gca()

    azSpiderweb = np.arange(0, 360, azint)
    maxAz = azSpiderweb[-1]

    xx, yy = np.meshgrid(np.linspace(-maxR-2, maxR+2, n), np.linspace(-maxR-2, maxR+2, n))

    mask = (xx >= xlims[0]) & (xx <= xlims[1]) & (yy >= ylims[0]) & (yy <= ylims[1])

    rows, cols = np.where(mask)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    xx = xx[row_min-1:row_max+2, col_min-1:col_max+2]
    yy = yy[row_min-1:row_max+2, col_min-1:col_max+2]
    _, az = cart2polar(xx, yy)

    fracLast = (360-maxAz)/4
    az[(az > (maxAz + fracLast)) & (az < (360 - fracLast))] = np.nan
    az[(az >= (360 - fracLast)) & (az <= 360)] -= 360

    contour  = ax.contour(xx, yy, az, azSpiderweb, colors='k')
    ax.clabel(contour, inline=True)