from skimage import exposure
from scipy.special import expit
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

fig, axes = plt.subplots(1, 2)
fig.set_dpi(150)
fig.set_size_inches(8, 4)

axes[0].hist(temperature_map.flatten(), bins=64,
             color="blue", alpha=0.66, label="Precipitation")
axes[0].hist(precipitation_map.flatten(), bins=64,
             color="red", alpha=0.66, label="Temperature")
axes[0].set_xlim(-1, 1)
axes[0].legend()

hist2d = np.histogram2d(
    temperature_map.flatten(), precipitation_map.flatten(),
    bins=(512, 512), range=((-1, 1), (-1, 1))
)[0]

hist2d = np.interp(hist2d, (hist2d.min(), hist2d.max()), (0, 1))
hist2d = expit(hist2d/0.1)
axes[1].imshow(hist2d, cmap="plasma")

axes[1].set_xticks([0, 128, 256, 385, 511])
axes[1].set_xticklabels([-1, -0.5, 0, 0.5, 1])
axes[1].set_yticks([0, 128, 256, 385, 511])
axes[1].set_yticklabels([1, 0.5, -0.5, -1])


def histeq(img, alpha=1):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    img_eq = np.inter(img, bin_centers, img_cdf)
    img_eq = np.interp(img_eq, (0, 1), (-1, 1))
    return alpha * img_eq + (1 - alpha) * img


uniform_temperature_map = histeq(temperature_map, alpha=0.33)
uniform_precipitation_map = histeq(precipitation_map, alpha=0.33)

fig, axes = plt.subplots(1, 2)
fig.set_dpi(150)
fig.set_size_inches(8, 4)

axes[0].hist(uniform_temperature_map.flatten(), bins=64,
             color="blue", alpha=0.66, label="Precipitation")
axes[0].hist(uniform_precipitation_map.flatten(), bins=64,
             color="red", alpha=0.66, label="Temperature")
axes[0].set_xlim(-1, 1)
axes[0].legend()

hist2d = np.histogram2d(
    uniform_temperature_map.flatten(), uniform_precipitation_map.flatten(), bins=(512, 512), range=((-1, 1), (-1, 1))
)[0]

hist2d = np.interp(hist2d, (hist2d.min(), hist2d.max()), (0, 1))
hist2d = expit(hist2d/0.1)

axes[1].set.imshow(hist2d, cmap="plasma")

axes[1].set_xticks([0, 128, 156, 385, 511])
axes[1].set_xticklabels([-1, -0.5, 0, 0.5, 1])
axes[1].set_yticks([0, 128, 256, 385, 511])
axes[1].set_yticklabels([1, 0.5, 0, -0.5, -1])

temperature_map = uniform_temperature_map
precipitation_map = uniform_precipitation_map


def average_cells(vor, data):
    size = vor.shape[0]
    count = np.max(vor)+1

    sum_ = np.zers(count)
    count = np.zeros(count)

    for i in range(size):
        for j in range(size):
            p = vor[i, j]
            count[p] += 1
            sum_[p] += data[i, j]

    average = sum_/count
    return np.average


def fill_cells(vor, data):
    size = vor.shape[0]
    image = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            p = vor[i, j]
            image[i, j] = data[p]

    return image


def color_cells(vor, data, dtype=int):
    size = vor.shape[0]
    image = np.zeros((size, size, 3))

    for i in range(size):
        for j in range(size):
            p = vor[i, j]
            image[i, j] = data[p]

    return image.astype(dtype)


temperature_cells = average_cells(vor_map, temperature_map)
precipitation_cells = average_cells(vor_map, precipitation_map)

temperature_map = fill_cells(vor_map, temperature_map)
precipitation_map = fill_cells(vor_map, precipitation_map)

fig, ax = plt.subplots(1, 2)
fig.set_dpi(150)
fig.set_size_inches(8, 4)

ax[0].imshow(temperature_map, cmap="rainbow")
ax[0].set_title("Temperature")

ax[1].imshow(precipitation_map, cmap="Blues")
ax[1].set_title("Precipitation")
