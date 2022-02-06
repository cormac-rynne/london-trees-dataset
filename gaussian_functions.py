import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import h5py
import os
from sortedcontainers import SortedDict
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial
from itertools import islice


def generate_gaussian_kernels(round_decimals=3, sigma_threshold=4, sigma_min=0,
                              sigma_max=20, num_sigmas=801):
    """
    Computing gaussian filter kernel for sigmas in linspace(sigma_min,
    sigma_max, num_sigmas) and saving them to a dictionary. The key-value pair
    is sigma:kernel.
    """

    kernels_dict = dict()
    sigma_space = np.linspace(sigma_min, sigma_max, num_sigmas)
    for sigma in sigma_space:
        sigma = np.round(sigma, decimals=round_decimals)
        kernel_size = (np.ceil(sigma * sigma_threshold).astype(int) * 2) + 1

        img_shape  = (kernel_size, kernel_size)
        img_center = (img_shape[0] // 2, img_shape[1] // 2)

        arr = np.zeros(img_shape)
        arr[img_center] = 1

        arr = gaussian_filter(arr, sigma, mode='constant')
        kernel = arr / arr.sum()
        kernels_dict[sigma] = kernel

    return SortedDict(kernels_dict)


def gaussian_filter_density(non_zero_points, map_h, map_w, distances=None,
                            kernels_dict=None, min_sigma=2, method=1,
                            const_sigma=15):
    """
    Fast gaussian filter implementation : using precomputed distances and kernels
    """
    gt_count = non_zero_points.shape[0]
    density_map = np.zeros((map_h, map_w), dtype=np.float32)

    for i in range(gt_count):
        point_y, point_x = non_zero_points[i]
        sigma = compute_sigma(gt_count, distances[i], min_sigma=min_sigma,
                              method=method, fixed_sigma=const_sigma)
        closest_sigma = find_closest_key(kernels_dict, sigma)
        kernel = kernels_dict[closest_sigma]
        full_kernel_size = kernel.shape[0]
        kernel_size = full_kernel_size // 2

        # get min and max x and y coordinates for kernel around point
        min_img_x = max(0, point_x - kernel_size)
        min_img_y = max(0, point_y - kernel_size)
        max_img_x = min(point_x + kernel_size + 1, map_h - 1)
        max_img_y = min(point_y + kernel_size + 1, map_w - 1)

        # get slice coordinates of kernel if kernal goes over image boundary
        kernel_x_min = kernel_size - point_x if point_x <= kernel_size else 0
        kernel_y_min = kernel_size - point_y if point_y <= kernel_size else 0
        kernel_x_max = kernel_x_min + max_img_x - min_img_x
        kernel_y_max = kernel_y_min + max_img_y - min_img_y

        # Apply kernel
        density_map[min_img_x:max_img_x, min_img_y:max_img_y] += kernel[kernel_x_min:kernel_x_max, kernel_y_min:kernel_y_max]
    return density_map


def compute_sigma(gt_count, distance=None, min_sigma=1, method=1, fixed_sigma=15):
    """
    Compute sigma for gaussian kernel with different methods :
    * method = 1 : sigma = (mean of distance to 3 nearest neighbors) / 10
    * method = 2 : sigma = distance to nearest neighbor
    * method = 3 : sigma = fixed value
    ** if sigma lower than threshold 'min_sigma', then 'min_sigma' will be used
    ** in case of one point on the image sigma = 'fixed_sigma'
    """
    if gt_count > 1 and distance is not None:
        if method == 1:
            sigma = np.mean(distance[1:4]) * 0.1
        elif method == 2:
            sigma = distance[1]
        elif method == 3:
            sigma = fixed_sigma
    else:
        sigma = fixed_sigma
    if sigma < min_sigma:
        sigma = min_sigma
    return sigma


def find_closest_key(sorted_dict, key):
    """
    Find closest key in sorted_dict to 'key'
    """
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))


def save_computed_density(density_map, out_path):
    """
    Save density map to h5py format
    """
    with h5py.File(out_path, 'w') as hf:
        hf['density'] = density_map
    print(f'{out_path} saved.')


def compute_distances(points_dct, n_neighbors = 4, leafsize=1024):
    """
    Approximates the nearest 3 neighbours to each tree
    """
    distances_dct = dict()

    for full_img_path, points in points_dct.items():
        # build kdtree
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)

        # query kdtree
        distances, _ = tree.query(points, k=n_neighbors)

        distances_dct[full_img_path] = distances

    return distances_dct


def generate_points_dct(img_gt_paths):
    """
    Generate dictionary of all points for all images
    """
    points_dct ={}
    for img_path, gt_path in img_gt_paths.copy():
        points = pd.read_csv(gt_path, header=None, index_col=None).values
        points_dct[img_path ] =points

    return points_dct


def generate_gt_den_maps(img_gt_paths, points_dct, distances_dct, kernels_dct, method=3, const_sigma=15,
                         file='h5', min_sigma=2):
    """
    Generates density map file
    """
    for img_filepath, _ in img_gt_paths:
        extension = os.path.splitext(img_filepath)[-1]

        # load img and map
        img = Image.open(img_filepath)
        width, height = img.size
        gt_points = points_dct[img_filepath]
        distances = distances_dct[img_filepath]

        # Generate image density map
        density_map = gaussian_filter_density(
            gt_points, height, width, distances, kernels_dct, min_sigma=min_sigma,
            method=method, const_sigma=const_sigma
        )

        # Save
        den_path = img_filepath.replace('img', 'gt_map_gaus')
        if file == 'csv':
            den_path = den_path.replace(extension, '.csv')
            pd.DataFrame(density_map).to_csv(den_path, index=None, header=None)
        if file == 'h5':
            den_path = den_path.replace(extension, '.h5')
            with h5py.File(den_path, 'w') as hf:
                hf['density'] = density_map

        filename = os.path.split(den_path)[-1]
        print(f'{filename} saved.')