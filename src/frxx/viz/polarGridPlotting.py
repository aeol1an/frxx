import numpy as np
from matplotlib import pyplot as plt
from ..utils.coordConvert import cart2polar

def rangeRings(ax, rint = None, xlim = None, ylim = None, n = 100, lw=1.5):
    if ax is None:
        ax = plt.gca()

    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    xmax = max([abs(i) for i in xlim])
    ymax = max([abs(i) for i in ylim])

    maxR = np.sqrt(xmax**2 + ymax**2)

    if rint is None:
        rint = round(maxR/6)

    rings = np.arange(rint,maxR+rint,rint)

    xx, yy = np.meshgrid(np.linspace(-maxR-rint, maxR+rint, n), np.linspace(-maxR-rint, maxR+rint, n))

    mask = (xx >= xlim[0]) & (xx <= xlim[1]) & (yy >= ylim[0]) & (yy <= ylim[1])

    rows, cols = np.where(mask)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    xx = xx[row_min-1:row_max+2, col_min-1:col_max+2]
    yy = yy[row_min-1:row_max+2, col_min-1:col_max+2]
    rr, _ = cart2polar(xx, yy)

    contour  = ax.contour(xx, yy, rr, rings, colors='k', linewidths=lw)
    ax.clabel(contour, inline=True, fmt='%d km', fontsize=lw*8)

def azimuthSpiderweb(ax, azint = 20, xlim = None, ylim = None, n = 500, lw=1.5):
    if ax is None:
        ax = plt.gca()

    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    xmax = max([abs(i) for i in xlim])
    ymax = max([abs(i) for i in ylim])

    maxR = np.sqrt(xmax**2 + ymax**2)

    azSpiderweb = np.arange(0, 360, azint)
    maxAz = azSpiderweb[-1]

    def concentratedLinspace(limit, n, power=2):
        t = np.linspace(-1, 1, n)
        return np.sign(t) * (np.abs(t) ** power) * limit

    xx, yy = np.meshgrid(concentratedLinspace(maxR+2,n), concentratedLinspace(maxR+2,n))

    mask = (xx >= xlim[0]) & (xx <= xlim[1]) & (yy >= ylim[0]) & (yy <= ylim[1])

    rows, cols = np.where(mask)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    xx = xx[row_min-1:row_max+2, col_min-1:col_max+2]
    yy = yy[row_min-1:row_max+2, col_min-1:col_max+2]
    _, az = cart2polar(xx, yy)

    fracLast = (360-maxAz)/4
    az[(az > (maxAz + fracLast)) & (az < (360 - fracLast))] = np.nan
    az[(az >= (360 - fracLast)) & (az <= 360)] -= 360

    contour  = ax.contour(xx, yy, az, azSpiderweb, colors='k', linewidths=lw)
    ax.clabel(contour, inline=True, fmt='%d$^{\\circ}$', fontsize=lw*8)