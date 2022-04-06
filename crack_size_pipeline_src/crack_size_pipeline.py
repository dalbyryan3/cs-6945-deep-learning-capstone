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
# # Crack Size Estimation Pipeline

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
# Seaborn for statistics and plotting 
import seaborn as sns

# The following are to do with interactive notebook code
# %matplotlib inline 
import matplotlib
from matplotlib import pyplot as plt # this lets you draw inline pictures in the notebooks
import matplotlib.patches as patches
import pylab # this allows you to control figure size 
pylab.rcParams['figure.figsize'] = (10.0, 8.0) # this controls figure size in the notebook

print(f"OpenCV Version: {cv.__version__}")
plt.ion()   # interactive mode

dtype = torch.float
ltype = torch.long
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

print('Torch using device:', device)


# %% [markdown]
# ## Blyncsy Data

# %%
blyncsy_data_path = '/cs6945share/blyncsy_data'
blyncsy_images_path = os.path.join(blyncsy_data_path, 'udot/sample1/images')
blyncsy_geojson_path = os.path.join(blyncsy_data_path, 'udot/sample1/sample1.geojson')

blyncsy_image_filepaths = [os.path.join(blyncsy_images_path, f) for f in sorted(os.listdir(blyncsy_images_path))]

# %%
r = random.choices(list(range(len(blyncsy_image_filepaths))), k=20)
for i in r:
    print(i)
    plt.figure()
    plt.imshow(plt.imread(blyncsy_image_filepaths[i]))
    plt.show()

# %%
plt.figure()
plt.imshow(plt.imread(blyncsy_image_filepaths[155]))
plt.show()
plt.figure()
plt.imshow(plt.imread(blyncsy_image_filepaths[264]))
plt.show()
plt.figure()
plt.imshow(plt.imread(blyncsy_image_filepaths[286]))
plt.show()
plt.figure()
plt.imshow(plt.imread(blyncsy_image_filepaths[526]))
plt.show()


# %% [markdown]
# ## Utility Functions

# %%
def insert_to_path_if_necessary(path):
    if path not in sys.path:
        sys.path.insert(0, os.path.abspath(path))
        print("Updated Python Path")


# %%
def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


# %%
def extract_images_from_video(vid_path, vid_extract_start_ms, vid_extract_end_ms, skip_num=1, preview=True, preview_and_save=False):
    extracted_image_list = []
    vid = cv.VideoCapture(vid_path)
    i = 0
    while vid.isOpened():
        image_exists, image = vid.read()
        current_ms = vid.get(cv.CAP_PROP_POS_MSEC)
        if (current_ms < vid_extract_start_ms):
            continue
        if (current_ms > vid_extract_end_ms):
            break
        if not image_exists:
            continue
        i += 1
        if i != skip_num:
            continue
        i = 0
        image_color_conv = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        extracted_image_list.append(image_color_conv)
        if preview:
            plt.figure()
            plt.title('Image timestamp = {0}ms'.format(current_ms))
            plt.imshow(image_color_conv)
            if preview_and_save:
                save_path = os.path.join(os.getcwd(),'img_{0}.png'.format(current_ms))
                cv.imwrite(save_path, image)
                print('Saved image to {0}'.format(save_path))
            plt.show()
    vid.release()
    return extracted_image_list


# %%
def plot_p1_p2_collection(vid_image, p1_collection, p2_collection):
    plt.figure()
    plt.imshow(vid_image)
    for p1,p2 in zip(p1_collection, p2_collection):
        color = np.random.rand(3,)
        plt.plot(p1[0], p1[1], c=color, marker='.')
        plt.plot(p2[0], p2[1], c=color, marker='x')
    plt.show()

# # e.g.
# p1_collection = np.array([[598, 1024], [865, 1120]]) # Denoted as .
# p2_collection = np.array([[742, 1003], [965, 1073]]) # Denoted as x

def plotting_points_of_interest(fig, ax1, ax2, p1_collection, p2_collection, evaluation_results, ax1_title_text=None, ax2_title_text=None):
    legend_list = []
    for i,(p1,p2) in enumerate(zip(p1_collection, p2_collection)):
        predicted_distances, ground_truth, difference = evaluation_results
        color = np.random.rand(3,)
        ax1.plot(p1[0], p1[1], c=color, marker='.')
        ax1.plot(p2[0], p2[1], c=color, marker='x')
        ax2.plot(p1[0], p1[1], c=color, marker='.')
        ax2.plot(p2[0], p2[1], c=color, marker='x')
        patch = patches.Patch(color=color, label='Predicted distances pred dist = {0}m, GT = {1}m, diff={2}m'.format(predicted_distances[i], ground_truth[i], difference[i]))
        legend_list.append(patch)
    ax1.legend(handles=legend_list)
    ax2.legend(handles=legend_list)
    if ax1_title_text is not None:
        ax1.set_title(ax1_title_text)
    if ax2_title_text is not None:
        ax2.set_title(ax2_title_text)

def ada_bins_evalaute_against_ground_truth(dmap, ground_truth, p1_collection, p2_collection):
    dmap_array = np.array(dmap)
    predicted_distances = []
    for i,(p1,p2) in enumerate(zip(p1_collection, p2_collection)):
        predicted_distance = np.abs(dmap_array[tuple(p2)] - dmap_array[tuple(p1)])
        predicted_distances.append(predicted_distance)
        
    predicted_distances_arr = np.array(predicted_distances)
    difference = predicted_distances_arr - ground_truth
    return predicted_distances_arr, ground_truth, difference



# %%
def viz_boxes(img_filepath, list_tup, n_classes=6):
    plt.figure()
    plt.imshow(plt.imread(img_filepath))
    cmap = get_cmap(n_classes, name='hsv_r')
    for tup in list_tup:
        bb, cls, prob = tup 

        color = cmap(cls)
        x_min, y_min, x_max, y_max = bb
        # bb is [x_min, y_min, x_max, y_max]
        width = x_max-x_min
        height = y_max-y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x_max-18, y_max, cls, fontweight='heavy', color=color)
    plt.show()

def filter_list_tup(list_tup, prob_thresh):
    filtered_list_tup = []
    for tup in list_tup:
        bb, cls, prob = tup 
        if prob < prob_thresh:
            continue
        filtered_list_tup.append(tup)
    return filtered_list_tup


# %% [markdown]
# # Naive Size Estimation Approach
#
# Monocular Image -> (YOLO/FasterRCNN) -> Bounding Boxed Cracks -> (AdabBins), (LaneDetector) -> Depth Map, Detected Lane -> Crack Estimated Dimensions
#
#
# Will naively estimate the "height" of crack as difference in depth between the top and bottom depth values of the bounding box. (Can mess with exactly what depth values are used, e.g. an average over the top and an average over bottom depth values etc.)
#
# Will naively estimate the "width" of a crack in terms of the lane width (ratio of width of crack bounding box to the width of the lane) then given a lane size of the road can give the "width" of the crack.

# %% [markdown]
# ## Crack Detection

# %% [markdown]
# ### YOLOv5

# %%
yolo_src_root = '../yolov5x weighted model'
yolo_model_path = os.path.join(yolo_src_root, 'yolov5x.pt') # Also'yolov5s.pt'

# %%
yolo_model = torch.load(yolo_model_path)
# Should load onto a yolo model object (I think this was saved as a state dict)

# %% [markdown]
# ### FasterRCNN

# %%
fasterrcnn_src_root = '../adithya_models/models'
fasterrcnn_model_path = os.path.join(fasterrcnn_src_root, 'v30-new-metrics-sample-6k_full.pth')


# %%
# a "list_tup" is a list of tupes containing tuples of bounding boxes of [x_min, y_min, x_max, y_max], a classification, and a probability/softmax output for the classification
def img_filepath_to_FasterRCNN_input_tensor(img_filepath, device):
    img = Image.open(img_filepath)
    img_arr = np.asarray(img, dtype="float")/255
    return torch.unsqueeze(torch.tensor(img_arr, device=device).movedim(2,0).float(),0)

def FasterRCNN_output_to_list_tup(out_full):
    out = out_full[0]
    boxes = out['boxes'].cpu().detach().numpy()
    labels = out['labels'].cpu().detach().numpy()
    scores = out['scores'].cpu().detach().numpy()
    list_tup = []
    for b,l,s in zip(boxes, labels, scores):
        list_tup.append((b,l,s))
    return list_tup
    


# %%
fasterrcnn_model = torch.load(fasterrcnn_model_path)
fasterrcnn_model.eval()
# print(fasterrcnn_model)

# %% [markdown]
# ## Monocular Depth Estimation

# %% [markdown]
# ### AdaBins

# %%
# AdaBins Monocular Depth for Monocular Depth Estimation
adabins_src_root = './AdaBins'
insert_to_path_if_necessary(adabins_src_root)
from infer import InferenceHelper


# %%
def adabins_predict_pil_anysize(infer_helper, pil_image, prediction_dimensions):
    original_size = pil_image.size
    pil_image_resize = pil_image.resize(prediction_dimensions)
    bin_centers, predicted_depth = infer_helper.predict_pil(pil_image_resize)
    d_map_pred = predicted_depth[0][0]
    d_map_pred_upsample = Image.fromarray(d_map_pred).resize(original_size) # Performs bicubic interpolation
    return d_map_pred_upsample

def adabins_predict_and_viz(pil_image, dmap_pred, extra_plotting_func=None, cbar_min=None, cbar_max=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,5))

    ax1.imshow(pil_image)
    ax1.set_title('Monocular Dashcam Image')

    sns.heatmap(d_map_pred, cmap=sns.color_palette("Spectral_r", as_cmap=True), square=True, ax=ax2, vmin=cbar_min, vmax=cbar_max)
    ax2.set_title('AdaBins Predicted Depth Map (in meters)')
    
    if extra_plotting_func is not None:
        extra_plotting_func(fig, ax1, ax2)
    
    ax1.set_yticks([],[])
    ax1.set_xticks([],[])
    ax2.set_yticks([],[])
    ax2.set_xticks([],[])
    fig.tight_layout()
    fig.show()
    plt.show()

# e.g.:
# prev_dir = os.getcwd()
# try:
#     os.chdir(adabins_src_root)
#     infer_helper = InferenceHelper(dataset='kitti') # dataset='kitti' sets some parameters for training and things like min depth, max depth, and saving factor, only matters for inference so values beyond 10 meters can be predicted
    
#     d_map_pred = adabins_predict_pil_anysize(infer_helper, pil_img, (1241, 376))
#     adabins_predict_and_viz(pil_img, d_map_pred, extra_plotting_func=lambda fig, ax1, ax2:ax2.set_title('AdaBins Predicted Depth Map (in meters) (resizing to kitti size of (1241, 376) for inference)'), cbar_min=0, cbar_max=60)
        
# finally:
#     os.chdir(prev_dir)
    


# %% [markdown]
# ## Lane Detection
# ### LaneDet
# Open source lane detection toolbox based on PyTorch that aims to pull together a wide variety of state-of-the-art lane detection models: https://github.com/Turoad/lanedet

# %%
# LaneDet 

# %% [markdown]
# ## Pipeline

# %%
test_blyncsy_image_filepath = blyncsy_image_filepaths[286]
plt.figure()
plt.imshow(plt.imread(test_blyncsy_image_filepath))
plt.show()

with torch.no_grad():
    test_tensor = img_filepath_to_FasterRCNN_input_tensor(test_blyncsy_image_filepath, device)
    fasterrcnn_model.eval()
    out = fasterrcnn_model(test_tensor)
    list_tup = FasterRCNN_output_to_list_tup(out)
    filtered_list_tup = filter_list_tup(list_tup, 0.4)
    viz_boxes(test_blyncsy_image_filepath,filtered_list_tup,n_classes=6)

prev_dir = os.getcwd()
try:
    os.chdir(adabins_src_root)
    infer_helper = InferenceHelper(dataset='kitti') # dataset='kitti' sets some parameters for training and things like min depth, max depth, and saving factor, only matters for inference so values beyond 10 meters can be predicted
    
    pil_img = Image.open(test_blyncsy_image_filepath)
    d_map_pred = adabins_predict_pil_anysize(infer_helper, pil_img, (1241, 376))
    adabins_predict_and_viz(pil_img, d_map_pred, extra_plotting_func=lambda fig, ax1, ax2:ax2.set_title('AdaBins Predicted Depth Map (in meters) (resizing to kitti size of (1241, 376) for inference)'), cbar_min=0, cbar_max=60)
        
finally:
    os.chdir(prev_dir)
