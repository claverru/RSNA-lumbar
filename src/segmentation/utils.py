import colorsys
import math
import random

import torch

import numpy as np


def calc_size(sigma=1.6):
    sigma_6 = 6*sigma
    return int(sigma_6 - sigma_6%2 + 1)


def gaussian_filter2d(sigma=1.6):
    size = calc_size(sigma)
    limit = float(size//2)
    space = torch.linspace(-limit, limit, size)
    g = torch.exp(-space**2 / (2.0 * sigma**2)) / (torch.sqrt(torch.tensor(2 * math.pi)) * sigma**2)
    g2 = torch.tensordot(g, g, dims=0)
    g2 = g2 - g2.min()
    g2 = g2 / g2.amax()
    return g2


def gaussian_filter2d_np(sigma=1.6):
    size = calc_size(sigma)
    limit = float(size//2)
    space = np.linspace(-limit, limit, size)
    g = np.exp(-space**2 / (2.0 * sigma**2)) / (np.sqrt(2 * math.pi) * sigma**2)
    g2 = np.tensordot(g, g, axes=0)
    g2 = g2 - g2.min()
    g2 = g2 / g2.max()
    return g2


def generate_random_colors(n: int, bright: bool = True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = [tuple([int(v * 255) for v in c]) for c in colors]
    random.shuffle(colors)
    return colors


def draw_binary_masks(
    image,
    binary_masks,
    colors = None,
) -> np.ndarray:
    if binary_masks is None:
        return image

    binary_masks = np.stack(binary_masks, 0)[..., None]

    if colors is None:
        colors = generate_random_colors(len(binary_masks))
    colors = np.array(colors)

    color_mask = (binary_masks * colors[:, None, None, :]).sum(0)
    print(color_mask.shape)
    print(binary_masks.shape)
    print(image.shape)
    return np.where(binary_masks.any(0), color_mask * binary_masks.max(0) + (1 - binary_masks.max(0)) * image, image).astype(np.uint8)
