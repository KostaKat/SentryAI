import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image, ImageFilter
import io
import matplotlib.pyplot as plt
import random
from skimage import data
from scipy.ndimage import rotate
from kernels import *


def img_to_patches(img, patch_size=32):
    # Ensure the image dimensions are multiples of patch_size or larger
    width, height = img.size
    # Calculate the number of patches that can fit horizontally and vertically
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size

    patches = [np.asarray(img.crop((j * patch_size, i * patch_size, 
                                    (j + 1) * patch_size, (i + 1) * patch_size))) 
               for i in range(num_patches_y) 
               for j in range(num_patches_x)]

    # Convert patches to grayscale
    grayscale_patches = [cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.int32) for patch in patches]

    return grayscale_patches, patches

def get_pixel_var_degree_for_patch(patch):
    diffs = np.abs(np.diff(patch, axis=0)) + np.abs(np.diff(patch, axis=1))
    diag_diffs = np.abs(np.diff(patch[:-1, :-1], axis=0) - np.diff(patch[1:, 1:], axis=0))
    return np.sum(diffs) + np.sum(diag_diffs)

def extract_rich_and_poor_textures(variance_values, patches):
    threshold = np.mean(variance_values)
    rich_patches = [patch for i, patch in enumerate(patches) if variance_values[i] >= threshold]
    poor_patches = [patch for i, patch in enumerate(patches) if variance_values[i] < threshold]
    return rich_patches, poor_patches

def get_complete_image(patches, grid_size=(8, 8), patch_size=32):
    random.shuffle(patches)
    patches = patches * (grid_size[0] * grid_size[1] // len(patches) + 1)
    grid = np.array(patches[:grid_size[0] * grid_size[1]]).reshape(grid_size + (patch_size, patch_size, -1))
    return np.block([[row] for row in grid])

def smash_n_reconstruct(img, coloured=True):
    grayscale_patches, color_patches = img_to_patches(img)
    pixel_var_degree = [get_pixel_var_degree_for_patch(patch) for patch in grayscale_patches]
    rich_patches, poor_patches = extract_rich_and_poor_textures(pixel_var_degree, color_patches if coloured else grayscale_patches)
    rich_texture = get_complete_image(rich_patches)
    poor_texture = get_complete_image(poor_patches)
    return rich_texture, poor_texture


def apply_high_pass_filter():
    rotated_kernels = []

    for idx, kernel in enumerate(kernels):
        for angle in angles[idx]:
            # Rotate kernel
            rotated_kernel = rotate(kernel, angle, reshape=False)
            # Ensure the kernel is in float32 format
            rotated_kernel = np.round(rotated_kernel).astype(np.float32)
            # Convert to tensor, shape [5, 5]
            tensor_kernel = torch.tensor(rotated_kernel)
            
            # Unsqueeze and repeat to convert to 3-channel, shape [3, 5, 5]
            tensor_kernel = tensor_kernel.unsqueeze(0).repeat(3, 1, 1)
            rotated_kernels.append(tensor_kernel)


    all_kernels = torch.stack(rotated_kernels)

    return all_kernels

# Define the image processing functions


def jpeg_compression(img, quality_range=(70, 100)):
    if random.random() < 0.1:  # 10% probability
        quality = random.randint(*quality_range)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=quality)
        img = Image.open(img_bytes)
    return img


def gaussian_blur(img, sigma_range=(0, 1)):
    if random.random() < 0.1:  # 10% probability
        sigma = random.uniform(*sigma_range)
        img = img.filter(ImageFilter.GaussianBlur(sigma))
    return img


def downsampling(img, scale_range=(0.25, 0.5)):
    if random.random() < 0.1:  # 10% probability
        scale = random.uniform(*scale_range)
        smaller_img = img.resize(
            (int(img.width * scale), int(img.height * scale)), Image.BICUBIC)
        img = smaller_img.resize((img.width, img.height), Image.BICUBIC)
    return img
