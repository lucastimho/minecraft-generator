import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
from skimage.draw import polygon
from PIL import Image
from noise import snoise3

size = 1024
n = 256
map_seed = 762345
np.random.seed(map_seed)


def voronoi(points, size):
    # add points at edges to eliminate infinite ridges
    edge_points = size*np.array([[-1, -1], [-1, 2], [2, -1], [2, 2]])
    new_points = np.vstack([points, edge_points])
    # calculates voronoi tesselation
    vor = Voronoi(new_points)
    return vor


def voronoi_map(vor, size):
    # calculate voronoi map
    vor_map = np.zeros((size, size), dype=np.uint32)
    for i, region in enumerate(vor.regions):
        # skip empty regions and infinite ridge regions
        if len(region) == 0 or -1 in region:
            continue
        # get polygon vertices
        x, y = np.array([vor.vertices[i][::-1] for i in region]).T
        # get pixels inside polygon
        rr, cc = polygon(x, y)
        # remove pixels out of image bounds
        in_box = np.where((0 <= rr) & (rr < size) & (0 <= cc) & (cc < size))
        rr, cc = rr[in_box], cc[in_box]
        # paint image
        vor_map[rr, cc] = i
    return vor_map


points = np.random.randint(0, size, (514, 2))
vor = voronoi(points, size)
vor_map = voronoi_map(vor, size)

fig = plt.figure(dpi=150, figsize=(4, 4))
plt.scatter(*points.T, s=1)
