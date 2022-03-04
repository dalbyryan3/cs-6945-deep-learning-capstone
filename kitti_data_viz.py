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
#     display_name: Python 3
#     language: python
#     name: python3
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
# Matrix manipulation library
import numpy as np
# Pytorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as F

# Standard python random library
import random

# OpenCV computer vision library for image manipulation
import cv2 as cv 

# The following are to do with interactive notebook code
# %matplotlib inline 
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
kitti_root_path = os.path.abspath('./')
kitti_depth_data_path = os.path.join(kitti_root_path, 'depth')
    
print("KITTI root path: {0}\nKITTI depth data path: {1}".format(kitti_root_path, kitti_depth_data_path))

# %%
