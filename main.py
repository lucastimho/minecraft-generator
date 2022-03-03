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
    vor_map = np.zeros((size, size), dtype=np.uint32)
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


def noise_map(size, res, seed, octaves=1, persistence=0.5, lacunarity=2.0):
    scale = size/res
    return np.array([[
        snoise3(
            (x+0.1)/scale,
            y/scale,
            seed+map_seed,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity
        )
        for x in range(size)]
        for y in range(size)
    ])


boundary_displacement = 8
boundary_noise = np.dstack(
    [noise_map(size, 32, 200, octaves=8), noise_map(size, 32, 250, octaves=8)])
boundary_noise = np.indices((size, size)).T + \
    boundary_displacement*boundary_noise
boundary_noise = boundary_noise.clip(0, size-1).astype(np.uint32)

blurred_vor_map = np.zeros_like(vor_map)

for x in range(size):
    for y in range(size):
        j, i = boundary_noise[x, y]
        blurred_vor_map[x, y] = vor_map[i, j]

fig, axes = plt.subplots(1, 2)
fig.set_dpi(150)
fig.set_size_inches(8, 4)
axes[0].imshow(vor_map)
axes[1].imshow(blurred_vor_map)

vor_map = blurred_vor_map

temperature_map = noise_map(size, 2, 10)
precipitation_map = noise_map(size, 2, 20)

fig, axes = plt.subplots(1, 2)
fig.set_dpi(150)
fig.set_size.inches(8, 4)

axes[0].imshow(temperature_map, cmap="rainbow")
axes[0].set.title("Temperature Map")

axes[1].imshow(precipitation_map, cmap="Y1GnBu")
axes[1].set_title("Precipitation Map")
