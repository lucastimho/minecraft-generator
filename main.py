from scipy.interpolate import interp1d
from scipy import ndimage
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


def quantize(data, n):
    bins = np.linspace(-1, 1, n+1)
    return (np.digitize(data, bins) - 1).clip(0, n-1)


n = 256

quantize_temperature_cells = quantize(temperature_cells, n)
quantize_precipitation_cells = quantize(precipitation_cells, n)

quantize_temperature_map = fill_cells(vor_map, quantize_temperature_cells)
quantize_precipitation_map = fill_cells(vor_map, quantize_precipitation_cells)

temperature_cells = quantize_temperature_cells
precipitation_cells = quantize_precipitation_cells

temperature_map = quantize_temperature_map
precipitation_map = quantize_precipitation_map

im = np.array(Image.open("output/TP_map.png"))[:, :, :3]
biomes = np.zeros((256, 256))

biome_names = [
    "desert",
    "savanna",
    "tropical_woodland",
    "tundra",
    "seasonal_forest",
    "rainforest",
    "temperate_forest",
    "temperate_rainforest",
    "boreal_forest"
]
biome_colors = [
    [255, 255, 178],
    [184, 200, 98],
    [188, 161, 53],
    [190, 255, 242],
    [106, 144, 38],
    [33, 77, 41],
    [86, 179, 106],
    [34, 61, 53],
    [35, 114, 94]
]

for i, color in enumerate(biome_colors):
    indices = np.where(np.all(im == color, axis=-1))
    biomes[indices] = i

biomes = np.flip(biomes, axis=0).T

fig = plt.figure(dpi=150, figsize=(4, 4))
plt.imshow(biomes)
plt.title("Temperature-Precipitation graph")

# Biome Map

n = len(temperature_cells)
biome_cells = np.zeros(n, dtype=np.uint32)

for i in range(n):
    temp, precip = temperature_cells[i], precipitation_cells[i]
    biome_cells[i] = biomes[temp, precip]

biome_map = fill_cells(vor_map, biome_cells).astype(np.uint32)
biome_color_map = color_cells(biome_map, biome_colors)

fig = plt.figure(figsize=(5, 5), dpi=150)
plt.imshow(biome_color_map)

# Height Map

height_map = noise_map(size, 4, 0, octaves=6, persistence=0.5, lacunarity=2)
land_mask = height_map > 0

fig = plt.figure(dpi=150, figsize=(5, 5))
plt.imshow(land_mask, cmap='gray')

sea_color = np.array([12, 14, 255])
land_mask_color = np.repeat(land_mask[:, :, np.newaxis], 3, axis=-1)
masked_biome_color_map = land_mask_color + \
    biome_color_map + (1 - land_mask_color) * sea_color

fig = plt.figure(dpi=150, figsize=(5, 5))
plt.imshow(masked_biome_color_map)


def gradient(im_smooth):
    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.arange(-1, 2).astype(float)
    kernel = - kernel / 2

    gradient_x = ndimage.convolve(gradient_x, kernel[np.newaxis])
    gradient_y = ndimage.convolve(gradient_y, kernel[np.newaxis].T)

    return gradient_x, gradient_y


def sobel(im_smooth):
    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    gradient_x = ndimage.convole(gradient_x, kernel)
    gradient_y = ndimage.convolve(gradient_y, kernel.T)

    return gradient_x, gradient_y


def compute_normal_map(gradient_x, gradient_y, intensity=1):
    width = gradient_x.shape[1]
    height = gradient_x.shape[0]
    max_x = np.max(gradient_x)
    max_y = np.max(gradient_y)

    max_value = max_x

    if max_y > max_x:
        max_value = max_y

    normal_map = np.zeros((height, width, 3), dtype=np.float32)

    intensity = 1 / intensity

    strength = max_value / (max_value * intensity)

    normal_map[..., 0] = gradient_x / max_value
    normal_map[..., 1] = gradient_y / max_value
    normal_map[..., 2] = 1 / strength

    norm = np.sqrt(np.power(normal_map[..., 0], 2) + np.power(
        normal_map[..., 1], 2) + np.power(normal_map[..., 2], 2))

    normal_map[..., 0] /= norm
    normal_map[..., 1] /= norm
    normal_map[..., 2] /= norm

    normal_map *= 0.5
    normal_map += 0.5

    return normal_map


def get_normal_map(im, intensity=1.0):
    sobel_x, sobel_y = sobel(im)
    normal_map = compute_normal_map(sobel_x, sobel_y, intensity)
    return normal_map


def get_normal_light(height_map_):
    normal_map_ = get_normal_map(height_map_)[:, :, 0:2].mean(axis=2)
    normal_map_ = np.interp(normal_map_, (0, 1), (-1, 1))
    return normal_map_


def apply_height_map(im_map, smooth_map, height_map, land_mask):
    normal_map = get_normal_light(height_map)
    normal_map = normal_map*land_mask + smooth_map/2*(~land_mask)

    normal_map = np.interp(normal_map, (-1, 1), (-192, 192))

    normal_map_color = np.repeate(normal_map[:, :, np.newaxis], 3, axis=-1)
    normal_map_color = normal_map_color.astype(int)

    out_map = im_map + normal_map_color
    return out_map, normal_map


biome_height_map, normal_map = apply_height_map(
    masked_biome_color_map, height_map, height_map, land_mask)

fig, ax = plt.subplots(1, 2)
fig.set_dpi(150)
fig.set_size_inches(10, 5)

ax[0].imshow(masked_biome_color_map)
ax[0].set_title("Biomes")

ax[1].imshow(biome_height_map)
ax[1].set_title("Biomes with normal")

height_map = noise_map(size, 4, 0, octaves=6, persistence=0.5, lacunarity=2)
smooth_height_map = noise_map(
    size, 4, 0, octaves=1, persistence=0.5, lacunarity=2)

fig, ax = plt.subplots(1, 2)
fig.set_dpi(150)
fig.set_size_inches(10, 5)

ax[0].imshow(height_map, cmap="gray")
ax[0].set_title("Height Map")

ax[1].imshow(smooth_height_map, cmap="gray")
ax[1].set_title("Smooth Height Map")


def bezier(x1, y1, x2, y2, a):
    p1 = np.array([0, 0])
    p2 = np.array([x1, y1])
    p3 = np.array([x2, y2])
    p4 = np.array([1, a])

    return lambda t: ((1-t)**3 * p1 + 3*(1-t)**2*2*t * p2 + 3*(1-t)*t*t**2 * p3 + t**3 * p4)


def bezier_lut(x1, y1, x2, y2, a):
    t = np.linspace(0, 1, 256)
    f = bezier(x1, y1, x2, y2, a)
    curve = np.array([f(t_) for t_ in t])

    return interp1d(*curve.T)


def filter_map(h_map, smooth_h_map, x1, y1, x2, y2, a, b):
    f = bezier_lut(x1, y1, x2, y2, a)
    output_map = b*h_map + (1-b)*smooth_h_map
    output_map = f(output_map.clip(0, 1))
    return output_map


biome_height_maps = [
    # desert
    filter_map(height_map, smooth_height_map, 0.75, 0.2, 0.95, 0.2, 0.2, 0.5),
    # savanna
    filter_map(height_map, smooth_height_map, 0.5, 0.1, 0.95, 0.1, 0.1, 0.2),
    # tropical woodland
    filter_map(height_map, smooth_height_map,
               0.33, 0.33, 0.95, 0.1, 0.1, 0.75),
    # tundra
    filter_map(height_map, smooth_height_map, 0.5, 1, 0.25, 1, 1, 1),
    # seasonal forest
    filter_map(height_map, smooth_height_map, 0.75, 0.5, 0.4, 0.4, 0.33, 0.2),
    # rainforest
    filter_map(height_map, smooth_height_map, 0.5, 0.25, 0.66, 1, 1, 0.5),
    # temperate forest
    filter_map(height_map, smooth_height_map, 0.75, 0.5, 0.4, 0.4, 0.33, 0.33),
    # temperate rainforest
    filter_map(height_map, smooth_height_map, 0.75, 0.5, 0.4, 0.4, 0.33, 0.33),
    # boreal
    filter_map(height_map, smooth_height_map, 0.8, 0.1, 0.9, 0.05, 0.05, 0.1)
]
