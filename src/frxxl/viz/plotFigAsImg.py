import numpy as np

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Sequence, Tuple

def plotFigAsImg(figs: Sequence[Figure], ax: Axes | None = None, marginPx: int = 0, srcVertical: bool = False) -> Tuple[Figure, Axes]:
    numFigs = len(figs)
    if len(figs) < 1 or len(figs) > 4:
        raise ValueError("Too many or too little figs.")

    backends = {type(fig.canvas).__name__ if fig.canvas is not None else None for fig in figs}
    if len(backends) > 1:
        raise ValueError(f"Figures have mismatched backends: {backends}")

    from matplotlib import pyplot as plt
    
    if 'FigureCanvasBase' in backends:
        from matplotlib.backends.registry import backend_registry
        backendName = plt.get_backend()
        backendMod = backend_registry.load_backend_module(backendName)
        FigureCanvas = backendMod.FigureCanvas
        canvases = [FigureCanvas(fig) for fig in figs]
    else:
        canvases = [fig.canvas for fig in figs]

    if ax is None:
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    for canvas in canvases:
        canvas.draw()
    imgs = [np.array(canvas.renderer.buffer_rgba()) for canvas in canvases]

    def applyMargins(img, margin):
        top, bottom, left, right = margin
        return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=0)

    if numFigs == 1:
        result = imgs[0]
    elif numFigs == 2:
        if srcVertical:
            margins = [(0, 0, 0, marginPx), (0, 0, 0, 0)]
            imgs = [applyMargins(imgs[i], margins[i]) for i in range(len(margins))]
            result = np.concatenate(imgs, axis=1)
        else:
            margins = [(0, marginPx, 0, 0), (0, 0, 0, 0)]
            imgs = [applyMargins(imgs[i], margins[i]) for i in range(len(margins))]
            result = np.concatenate(imgs, axis=0)
    elif numFigs == 3:
        if srcVertical:
            margins = [(0, 0, 0, marginPx), (0, 0, 0, marginPx), (0, 0, 0, 0)]
            imgs = [applyMargins(imgs[i], margins[i]) for i in range(len(margins))]
            result = np.concatenate(imgs, axis=1)
        else:
            margins = [(0, marginPx, 0, 0), (0, marginPx, 0, 0), (0, 0, 0, 0)]
            imgs = [applyMargins(imgs[i], margins[i]) for i in range(len(margins))]
            result = np.concatenate(imgs, axis=0)
    else:
        margins = [(0, marginPx, 0, marginPx), (0, marginPx, 0, 0), (0, 0, 0, marginPx), (0, 0, 0, 0)]
        imgs = [applyMargins(imgs[i], margins[i]) for i in range(len(margins))]
        result = np.block([[[imgs[0]], [imgs[1]]],[[imgs[2]], [imgs[3]]]])

    ax.imshow(result)

    return fig, ax
