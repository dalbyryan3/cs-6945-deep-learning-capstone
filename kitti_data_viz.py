# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: default
#     language: python
#     name: default
# ---

# %% [markdown]
# # KITTI Data Visualization
# This is a notebook to visualize KITTI data

# %%
# Initialization
# %load_ext autoreload
# %autoreload 2

# Basic imports for file manipulation
import sys
import os
import time
import copy
import pickle
import glob
# Matrix manipulation library
import numpy as np
# Pytorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as F
import torch.utils.data


# Standard python random library
import random

# Computer vision libraries for image manipulation
import cv2 as cv 
from PIL import Image
import seaborn as sns

# The following are to do with interactive notebook code
# %matplotlib inline 
import matplotlib
from matplotlib import pyplot as plt # this lets you draw inline pictures in the notebooks
import pylab # this allows you to control figure size 
pylab.rcParams['figure.figsize'] = (10.0, 8.0) # this controls figure size in the notebook

print(f"OpenCV Version: {cv.__version__}")
plt.ion()   # interactive mode

dtype = torch.float
ltype = torch.long
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

print('Torch using device:', device)


# %%
def insert_to_path_if_necessary(path):
    if path not in sys.path:
        sys.path.insert(0, os.path.abspath(path))
        print("Updated Python Path")


# %%
kitti_root_path = os.path.abspath('../')
kitti_depth_data_path = os.path.join(kitti_root_path, 'depth')
kitti_raw_data_path = os.path.join(kitti_root_path, 'raw')
kitti_depthflattened_data_path = os.path.join(kitti_root_path, 'depth_flattened')

print("KITTI root path: {0}\nKITTI depth data path: {1}\nKITTI raw data path: {2}\nKITTI depth flattened data path: {3}".format(kitti_root_path, kitti_depth_data_path, kitti_raw_data_path, kitti_depthflattened_data_path))


# %% [markdown]
# # KITTI Data Usage Points
# ## Difference between "raw" KITTI dataset and "depth" KITTI dataset
# - The difference between this and the raw data is (From: https://github.com/ialhashim/DenseDepth/issues/51#issuecomment-524822202):
#     - KITTI raw refers to the ground truth, which is created by simply projecting one 360° LIDAR scan into the camera image. It is usually created on the fly based on the LIDAR bin files.
#         - http://www.cvlibs.net/datasets/kitti/raw_data.php
#         - https://github.com/nianticlabs/monodepth2/blob/master/datasets/kitti_dataset.py#L65 
#     - KITTI depth refers to a refined ground truth, which was created by Uhrig et. al through aggregation and automatic filtering of KITTI raw. It is only available for cam2 and cam3 and is stored as png.
#         - http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction 
#         - http://www.cvlibs.net/publications/Uhrig2017THREEDV.pdf 
#             - Mentions: “we use the image plane of the KITTI reference camera for all our experiments” 
#
#
# ## Depth map format
# Kitti depth values contained in the depth dataset .png files are in meters (can be converted to as mentioned below) and from "the image plane of the KITTI reference camera for all our experiments" (as per http://www.cvlibs.net/publications/Uhrig2017THREEDV.pdf)
#
# From the depth dev kit readme:
#
# Depth maps (annotated and raw Velodyne scans) are saved as uint16 PNG images,
# which can be opened with either MATLAB, libpng++ or the latest version of
# Python's pillow (from PIL import Image). A 0 value indicates an invalid pixel
# (ie, no ground truth exists, or the estimation algorithm didn't produce an
# estimate for that pixel). Otherwise, the **depth for a pixel can be computed
# in meters** by converting the uint16 value to float and dividing it by 256.0:
#
# disp(u,v)  = ((float)I(u,v))/256.0;
#
# valid(u,v) = I(u,v)>0;

# %%
# From devkit_depth from depth kitti data download
def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth

def depthmap_viz(depthmap_filepath, ax, should_print_size=False):
    d_map = depth_read(depthmap_filepath)
    if should_print_size: 
        print('depth map size: {0}'.format(d_map.shape))
    sns.heatmap(d_map, cmap=sns.color_palette("Spectral_r", as_cmap=True), square=True, ax=ax)
    ax.set_yticks([],[])
    ax.set_xticks([],[])

def depthmap_filepaths(depth_dir_filepath, total_num_viz=3, imgs_per_viz=30):
    num_viz = 0
    filepaths_list = []
    for i, filename in enumerate(sorted(glob.glob(os.path.join(depth_dir_filepath, "*.png")))):
        if num_viz == total_num_viz:
            break
        if (i % imgs_per_viz) != 0:
            continue
        filepaths_list.append(os.path.abspath(filename))
        num_viz +=1
    return filepaths_list



# %%
raw_filepaths = depthmap_filepaths(os.path.join(kitti_depth_data_path, 'data_depth_velodyne/train/2011_09_26_drive_0009_sync/proj_depth/velodyne_raw/image_02'))
depth_filepaths = depthmap_filepaths(os.path.join(kitti_depth_data_path, 'data_depth_annotated/train/2011_09_26_drive_0009_sync/proj_depth/groundtruth/image_02'))
depth_filepaths_basenames_dict = {os.path.basename(filepath):i for i, filepath in enumerate(depth_filepaths)}
# "depth" is the "improved" depth map mentioned in paper
for raw_filepath in raw_filepaths:
    print(raw_filepath)
    basename = os.path.basename(raw_filepath)
    fig, ax = plt.subplots(1,1)
    ax.set_title('{0} data_depth_velodyne'.format(basename))
    depthmap_viz(raw_filepath, ax, should_print_size=True)
    plt.show()
    if basename in depth_filepaths_basenames_dict:
        depth_filepath = depth_filepaths[depth_filepaths_basenames_dict[basename]]
        print(depth_filepath)
        fig, ax = plt.subplots(1,1)
        ax.set_title('{0} data_depth_annotated'.format(basename))
        depthmap_viz(depth_filepath, ax, should_print_size=True)
        plt.show()
    else:
        print('A corresponding {0} was not sampled out of data_depth_annotated'.format(basename))    
        


# %%
depth_filepaths = depthmap_filepaths(os.path.join(kitti_depthflattened_data_path,'depth_images'), total_num_viz=5, imgs_per_viz=3000)
raw_images_filepaths = depthmap_filepaths(os.path.join(kitti_depthflattened_data_path,'raw_images'), total_num_viz=5, imgs_per_viz=3000)

for raw_image_filepath, depth_filepath in zip(raw_images_filepaths, depth_filepaths):
    print('raw: {0}\ndepth: {1}'.format(raw_image_filepath, depth_filepath))
    print('raw image size: {0}'.format(raw_image.shape))
    basename = os.path.basename(raw_image_filepath)
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_title('Raw Image {0}'.format(basename))
    raw_image = plt.imread(raw_image_filepath)
    ax1.imshow(raw_image)
    ax2.set_title('Depth Image {0}'.format(basename))
    depthmap_viz(depth_filepath, ax2, should_print_size=True)
    plt.show()

# %%
raw_filepaths = depthmap_filepaths(os.path.join(kitti_depth_data_path, '/cs6945share/kitti_datasets/depth/data_depth_selection/test_depth_completion_anonymous/image/'))
depth_filepaths = depthmap_filepaths(os.path.join(kitti_depth_data_path, '/cs6945share/kitti_datasets/depth/data_depth_selection/test_depth_completion_anonymous/velodyne_raw/'))
for raw_filepath, depth_filepath in zip(raw_filepaths, depth_filepaths):
    print('raw: {0}\ndepth: {1}'.format(raw_filepath, depth_filepath))
    print('raw image size: {0}'.format(raw_image.shape))
    basename = os.path.basename(raw_filepath)
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_title('Raw Image {0}'.format(basename))
    raw_image = plt.imread(raw_image_filepath)
    ax1.imshow(raw_image)
    ax2.set_title('Depth Image {0}'.format(basename))
    depthmap_viz(depth_filepath, ax2, should_print_size=True)
    plt.show()




# %% [markdown]
# # Depth Completion
# Using PENet: Precise and Efficient Depth Completion
# - https://arxiv.org/abs/2103.00783
# - https://github.com/JUGGHM/PENet_ICRA2021

# %%
os.chdir('PENet_ICRA2021')
pe = torch.load('pe.pth.tar')
os.chdir('..')

# %%
pe_model = pe['model']


# %%
pe_net_src_root = 'PENet_ICRA2021'
insert_to_path_if_necessary(pe_net_src_root)

# %%
os.chdir('PENet_ICRA2021')
# !python main.py -b 1 -n pe --evaluate pe.pth.tar --data-folder /cs6945share/kitti_datasets/depth/ --data-folder-save /cs6945share/kitti_datasets/depth/data_pe_completed_depth --data-folder-rgb /cs6945share/kitti_datasets/raw/ --test
os.chdir('..')

# %%
os.chdir('PENet_ICRA2021')
# !python main.py -b 1 -n pe --evaluate pe.pth.tar --data-folder /cs6945share/kitti_datasets/depth_flattened --data-folder-save /cs6945share/kitti_datasets/depth/data_pe_completed_depth --test --flattened
os.chdir('..')

# %% [markdown]
# # Monocular Depth Estimation
# Using AdaBins:
# - https://arxiv.org/abs/2011.14141
# - https://github.com/shariqfarooq123/AdaBins

# %%
adabins_src_root = 'AdaBins'
insert_to_path_if_necessary(adabins_src_root)

# %%
