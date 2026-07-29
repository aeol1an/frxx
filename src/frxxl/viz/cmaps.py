from matplotlib.colors import LinearSegmentedColormap

class cmaps:
    def dmap(n=16):
        cmap_data = [
            (-5, '#000000'),
            (-4, '#404040'),
            (-2, '#9C9C9C'),
            (-0.5, '#C9C9C9'),
            (0, '#8C78B4'),
            (0.25, '#000098'),
            (0.5, '#2398D3'),
            (1, '#44FFD2'),
            (1.5, '#57DB56'),
            (2, '#FFFF60'),
            (2.5, '#FF9045'),
            (3, '#DA0000'),
            (4, '#AE0000'),
            (5, '#F782BE'),
            (6, '#FFFFFF'),
            (8, '#FFFFFF')
        ]

        positions = [(x[0]-(-5))/(8-(-5)) for x in cmap_data]
        colors = [x[1] for x in cmap_data]

        cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)), N=n)
        cmap.set_bad('#FFFFFF', alpha=0)

        return cmap

    def rmap(n=17):
        cmap_data = [
            (0.1, '#000000'),
            (0.2, '#95949C'),
            (0.45, '#16148C'),
            (0.65, '#0902D9'),
            (0.75, '#8987D6'),
            (0.8, '#5CFF59'),
            (0.85, '#8BCF02'),
            (0.9, '#FFFB00'),
            (0.93, '#FFC400'),
            (0.95, '#FF8903'),
            (0.96, '#FF2B00'),
            (0.97, '#E30000'),
            (0.98, '#A10000'),
            (0.99, '#970556'),
            (1.01, '#FAACD1'),
            (1.05, '#FAACD1')
        ]

        positions = [(x[0]-0.1)/(1.05-0.1) for x in cmap_data]
        colors = [x[1] for x in cmap_data]

        cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)), N=n)
        cmap.set_bad('#FFFFFF', alpha=0)

        return cmap