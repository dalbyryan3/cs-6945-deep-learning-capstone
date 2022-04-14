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
from copy import deepcopy
# Matrix manipulation library
import numpy as np
# Pytorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch.utils.data
# Standard python libraries
import random
import math
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
plt.figure()
plt.imshow(plt.imread(blyncsy_image_filepaths[1148]))
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
def viz_boxes(pil_img, list_tup_bbox, n_classes=6):
    plt.figure()
    plt.imshow(np.array(pil_img))
    cmap = get_cmap(n_classes, name='hsv_r')
    for tup_bbox in list_tup_bbox:
        bb, cls, prob = tup_bbox 

        color = cmap(cls)
        x_min, y_min, x_max, y_max = bb
        # bb is [x_min, y_min, x_max, y_max]
        width = x_max-x_min
        height = y_max-y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x_max-18, y_max, cls, fontweight='heavy', color=color)
    plt.show()

def filter_list_tup_bbox(list_tup_bbox, prob_thresh):
    filtered_list_tup_bbox = []
    for tup_bbox in list_tup_bbox:
        bb, cls, prob = tup_bbox 
        if prob < prob_thresh:
            continue
        filtered_list_tup_bbox.append(tup_bbox)
    return filtered_list_tup_bbox


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
# yolo_src_root = '../yolov5x weighted model'
# yolo_model_path = os.path.join(yolo_src_root, 'yolov5x.pt') # Also'yolov5s.pt'

# %%
# yolo_model = torch.load(yolo_model_path)
# Should load onto a yolo model object (I think this was saved as a state dict)

# %% [markdown]
# ### FasterRCNN

# %%
fasterrcnn_src_root = '../adithya_models/models'
fasterrcnn_model_path = os.path.join(fasterrcnn_src_root, 'v30-new-metrics-sample-6k_full.pth')


# %%
# a "list_tup_bbox" is a list of tupes containing tuples of bounding boxes of [x_min, y_min, x_max, y_max], a classification, and a probability/softmax output for the classification
def pil_to_FasterRCNN_input_tensor(pil_img, device):
    img_arr = np.asarray(pil_img, dtype="float")/255
    FasterRCNN_input_tensor = torch.unsqueeze(torch.tensor(img_arr, device=device).movedim(2,0).float(),0)
    return FasterRCNN_input_tensor

def FasterRCNN_output_to_list_tup_bbox(out_full):
    out = out_full[0]
    boxes = out['boxes'].cpu().detach().numpy()
    labels = out['labels'].cpu().detach().numpy()
    scores = out['scores'].cpu().detach().numpy()
    list_tup_bbox = []
    for b,l,s in zip(boxes, labels, scores):
        list_tup_bbox.append((b,l,s))
    return list_tup_bbox
    


# %% [markdown]
# ## Monocular Depth Estimation

# %% [markdown]
# ### AdaBins

# %%
# AdaBins Monocular Depth for Monocular Depth Estimation
adabins_src_root = './AdaBins'
insert_to_path_if_necessary(adabins_src_root)
from models.unet_adaptive_bins import UnetAdaptiveBins
from model_io import load_checkpoint


# %%
def adabins_predict(model, image_tensor, device, min_depth, max_depth):
    bins, pred = model(image_tensor)
    pred = np.clip(pred.cpu().numpy(), min_depth, max_depth)

    # Flip
    flip_image_tensor = torch.Tensor(np.array(image_tensor.cpu().numpy())[..., ::-1].copy()).to(device)
    pred_lr = model(flip_image_tensor)[-1]
    pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], min_depth, max_depth)

    # Take average of original and mirror
    final = 0.5 * (pred + pred_lr)
    final = nn.functional.interpolate(torch.Tensor(final), image_tensor.shape[-2:],mode='bilinear', align_corners=True).cpu().numpy()

    final[final < min_depth] = min_depth
    final[final > max_depth] = max_depth
    final[np.isinf(final)] = max_depth
    final[np.isnan(final)] = min_depth

    centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
    centers = centers.cpu().squeeze().numpy()
    centers = centers[centers > min_depth]
    centers = centers[centers < max_depth]

    return centers, final

def adabins_predict_pil_anysize(model, normalizer, pil_image, prediction_dimensions, device, min_depth, max_depth):
    original_size = pil_image.size
    pil_image_resize = pil_image.resize(prediction_dimensions)
    image_resize = np.asarray(pil_image_resize) / 255.
    image_tensor = normalizer(torch.from_numpy(image_resize.transpose((2, 0, 1)))).unsqueeze(0).float().to(device)
    bin_centers, predicted_depth = adabins_predict(model, image_tensor, device, min_depth, max_depth)
    dmap_pred = predicted_depth[0][0]
    dmap_pred_upsample = Image.fromarray(dmap_pred).resize(original_size) # Performs bicubic interpolation
    return dmap_pred_upsample

def adabins_predict_pil_images_anysize(model, pil_images, prediction_dimensions, device, min_depth, max_depth):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dmaps = []
    for pil_image in pil_images:
        dmaps.append(adabins_predict_pil_anysize(model, normalizer, pil_image, prediction_dimensions, device, min_depth, max_depth))
    return dmaps

def adabins_viz(pil_image, dmap_pred, extra_plotting_func=None, cbar_min=None, cbar_max=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,5))

    ax1.imshow(pil_image)
    ax1.set_title('Monocular Dashcam Image')

    sns.heatmap(dmap_pred, cmap=sns.color_palette("Spectral_r", as_cmap=True), square=True, ax=ax2, vmin=cbar_min, vmax=cbar_max)
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


# %%
# test_adabins_images = [Image.open(blyncsy_image_filepaths[0]), Image.open(blyncsy_image_filepaths[1]), Image.open(blyncsy_image_filepaths[2])]

# from models.unet_adaptive_bins import UnetAdaptiveBins
# from model_io import load_checkpoint


# min_depth = 1e-3
# max_depth = 80
# model = UnetAdaptiveBins.build(n_bins=256, min_val=min_depth, max_val=max_depth)
# model, _, _ = load_checkpoint('./AdaBins/pretrained/AdaBins_kitti.pt', model)
# model.eval()
# model = model.to(device)

# with torch.no_grad():
#     dmaps = adabins_predict_pil_images_anysize(model, test_adabins_images, (1241, 376), device, min_depth, max_depth)
#     for img,dmap in zip(test_adabins_images,dmaps):
#         adabins_viz(img, dmap, extra_plotting_func=None, cbar_min=None, cbar_max=None)
        
        

# %% [markdown]
# ## Lane Detection
#

# %% [markdown]
# ### LaneDet
# Open source lane detection toolbox based on PyTorch that aims to pull together a wide variety of state-of-the-art lane detection models: https://github.com/Turoad/lanedet
# - It was found that there are some incompatibilities with the GPU because of the required pytorch verison, this can be kind of forced to work but I am unsure about the resulting performance

# %% [markdown]
# ### PINet

# %%
PINet_src_root = './PINet_new/TuSimple'
insert_to_path_if_necessary(PINet_src_root)
from hourglass_network import lane_detection_network


# %% [markdown]
# ## On mapping from PINet output (lane line points) to lane lines
#
# - Clustering can be problematic for getting lane lines from lane line points since points from different lines are often clustered together. It was then necessary to perform linear regression on the clusters to get lines.
#
# - Linear regression can be problematic for getting lane lines from lane line points since outlier points can greatly effect the line. This also meant that it was necessary to rely on the classification of lane lines by PINet (or clustering) (unlike the hough transform which I just made PINet output all points as the same class then found lines directly) which I found to be inconsitent and highly dependent on the PINet prediction parameters.
#

# %%
class PINet_params:
    def __init__(self):
        self.x_size = 512
        self.y_size = 256
        self.threshold_instance = 0.08 
        self.resize_ratio = 8    
        self.grid_x = self.x_size//self.resize_ratio                                                                                  
        self.grid_y = self.y_size//self.resize_ratio 
        self.grid_location = np.zeros((self.grid_y, self.grid_x, 2)) 
        self.threshold_point = 0.1 #0.35 #0.5 #0.57 #0.64 #0.35  
        
        for y in range(self.grid_y):                                                                                                    
            for x in range(self.grid_x):                                                                                                
                self.grid_location[y][x][0] = x                                                                                         
                self.grid_location[y][x][1] = y  
        self.color = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(100,255,0),(100,0,255),    (255,100,0),(0,100,255),(255,0,100),(0,255,100)] 

def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   
def generate_result(confidance, offsets,instance, p):
    mask = confidance > p.threshold_point

    grid = p.grid_location[mask]
    offset = offsets[mask]
    feature = instance[mask]

    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                flag = 0
                index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= p.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                elif len(lane_feature) < 12:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                
    return x, y

def sort_along_y(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(j, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

def draw_points(x, y, image, p):
    color_index = 0
    for i, j in zip(x, y):
        color_index += 1
        if color_index > 12:
            color_index = 12
        for index in range(len(i)):
            image = cv.circle(image, (int(i[index]), int(j[index])), 10, p.color[color_index], -1)
    return image

def PINet_predict_pil_anysize(pin_model, pil_img, device, p=PINet_params(), give_result_image=False):
    # PINet input should be:
#     sample_batch_size = 1
#     channel = 3
#     height = 256
#     width = 512
#     dummy_input = torch.randn(sample_batch_size, channel, height, width)
    img_arr = np.array(pil_img)
    orig_xy_dims = tuple(reversed(img_arr.shape[:2])) # Flip dims to be in "xy" order
    new_xy_dims = (512, 256)
    pil_img_resize = pil_img.resize(new_xy_dims)
    img_arr_resize = np.asarray(pil_img_resize, dtype="float")/255
    PINet_input_tensor = torch.unsqueeze(torch.tensor(img_arr_resize, device=device).movedim(2,0).float(),0)
    
    torch.cuda.synchronize()
    outputs, features = pin_model(PINet_input_tensor)
    confidences, offsets, instances = outputs[-1]     

    confidence = confidences[0].view(p.grid_y, p.grid_x).cpu().data.numpy()

    offset = offsets[0].cpu().data.numpy()
    offset = np.rollaxis(offset, axis=2, start=0)
    offset = np.rollaxis(offset, axis=2, start=0)
        
    instance = instances[0].cpu().data.numpy()
    instance = np.rollaxis(instance, axis=2, start=0)
    instance = np.rollaxis(instance, axis=2, start=0)

    # generate point and cluster
    raw_x, raw_y = generate_result(confidence, offset, instance, p)

    # eliminate fewer points
    in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
                
    # sort points along y 
    in_x, in_y = sort_along_y(in_x, in_y)  
    

    # rescale in_x and in_y to be on original image dimensions
    scale_factor_x = orig_xy_dims[0]/new_xy_dims[0]
    scale_factor_y = orig_xy_dims[1]/new_xy_dims[1]
    
    scale_in_x = [[int(j*scale_factor_x) for j in i] for i in in_x]
    scale_in_y = [[int(j*scale_factor_y) for j in i] for i in in_y]
    
    if give_result_image:
        result_image = draw_points(scale_in_x, scale_in_y, deepcopy(img_arr), p)
        return scale_in_x, scale_in_y, result_image

    return scale_in_x, scale_in_y

from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from skimage.transform import hough_line, hough_line_peaks,probabilistic_hough_line
from skimage.feature import canny
from skimage.color import rgb2gray
from matplotlib import cm


def get_lane_lines_linear_regression(x, y, y_min_max_tup=None):
    line_parameters_list = []
    for i, j in zip(x, y):
        x_arr = np.array(i).reshape(-1, 1)
        y_arr = np.array(j)
        if y_min_max_tup is not None:
            selected_idxs = np.logical_and(y_arr>y_min_max_tup[0], y_arr<y_min_max_tup[1])
            x_arr = x_arr[selected_idxs]
            y_arr = y_arr[selected_idxs]
            if not x_arr.size > 0:
                continue


#         cluster_arr = np.array((i,j)).T
#         # Compute DBSCAN
#         cluster_arr_norm = StandardScaler().fit_transform(cluster_arr)
#         db = DBSCAN(eps=0.3, min_samples=2).fit(cluster_arr_norm)
#         core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#         core_samples_mask[db.core_sample_indices_] = True
#         labels = db.labels_
#         # Number of clusters in labels, ignoring noise if present.
#         n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#         n_noise_ = list(labels).count(-1)
#         unique_labels = set(labels)
    
    
        model = LinearRegression().fit(x_arr, y_arr)  
        line_parameters_list.append((model.coef_.item(),model.intercept_.item()))
    return line_parameters_list

def get_lane_lines_hough_transform(x, y, lane_img_shape, y_min_max_tup=None, line_abs_slope_min_cutoff=1e-5, line_abs_slope_max_cutoff=10000, ht_peaks_min_distance=200, ht_peaks_min_angle=30, ht_peaks_threshold=100):
    line_parameters_list = []
    for i, j in zip(x, y):
        x_arr = np.array(i).reshape(-1, 1)
        y_arr = np.array(j)
        if y_min_max_tup is not None:
            selected_idxs = np.logical_and(y_arr>y_min_max_tup[0], y_arr<y_min_max_tup[1])
            x_arr = x_arr[selected_idxs]
            y_arr = y_arr[selected_idxs]
            if not x_arr.size > 0:
                continue
        background = np.zeros(lane_img_shape)
        for index in range(len(y_arr)):
            background = cv.circle(background, (int(x_arr[index,0]), int(y_arr[index])), 10, (255,255,255), -1)
        background = rgb2gray(background)
#         plt.figure()
#         plt.imshow(background, cmap=cm.gray)
        h, theta, d = hough_line(background)
        for accum, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=ht_peaks_min_distance, min_angle=ht_peaks_min_angle, threshold=ht_peaks_threshold)):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - lane_img_shape[1] * np.cos(angle)) / np.sin(angle)
#             plt.plot((0, lane_img_shape[1]), (y0, y1), '-r')
            m = (y1-y0)/lane_img_shape[1]
            # Continue if slope doesn't meet constraints
            if np.abs(m) < line_abs_slope_min_cutoff or np.abs(m) > line_abs_slope_max_cutoff:
                continue
            b = y0
            line_parameters_list.append((m,b))
#         plt.show()

    return line_parameters_list


# %%
# test_blyncsy_image_filepath = blyncsy_image_filepaths[286]
# test_blyncsy_image_filepath = blyncsy_image_filepaths[155]
# test_blyncsy_image_filepath = blyncsy_image_filepaths[526]
# test_blyncsy_image_filepath = blyncsy_image_filepaths[524]
test_blyncsy_image_filepath = blyncsy_image_filepaths[1148]

test_blyncsy_pil_image = Image.open(test_blyncsy_image_filepath)
PINet_p = PINet_params()
PINet_p.threshold_point = 0.1
PINet_p.threshold_instance = 1 # Set this to 1 so hough transform determines line instances
pin_model = lane_detection_network().to(device)
pin_state_dict = torch.load(os.path.join(PINet_src_root, 'savefile/0_tensor(0.5242)_lane_detection_network.pkl'))
pin_model.load_state_dict(pin_state_dict)
with torch.no_grad():
    pin_model.eval()
    in_x, in_y, lane_detect_result_image = PINet_predict_pil_anysize(pin_model, test_blyncsy_pil_image, device, p=PINet_p, give_result_image=True)

# plt.figure()
# plt.imshow(lane_detect_result_image)
# lane_lines = get_lane_lines_linear_regression(in_x,in_y, y_min_max_tup=(400, lane_detect_result_image.shape[0]-1))
# x_vals = np.arange(lane_detect_result_image.shape[1], dtype=np.int)
# for line in lane_lines:
#     y_vals = line[0]*x_vals + line[1]
#     idx_in_range = np.logical_and(y_vals < lane_detect_result_image.shape[0]-1, y_vals > 0)
#     plt.plot(x_vals[idx_in_range], y_vals[idx_in_range])
# plt.show()

plt.figure()
plt.imshow(lane_detect_result_image)
lane_lines = get_lane_lines_hough_transform(in_x,in_y,lane_detect_result_image.shape,  y_min_max_tup=(400, lane_detect_result_image.shape[0]-1), line_abs_slope_min_cutoff=0.1, line_abs_slope_max_cutoff=100, ht_peaks_min_distance=200, ht_peaks_min_angle=30, ht_peaks_threshold=170)
x_vals = np.arange(lane_detect_result_image.shape[1], dtype=np.int)
for line in lane_lines:
    y_vals = line[0]*x_vals + line[1]
    idx_in_range = np.logical_and(y_vals < lane_detect_result_image.shape[0]-1, y_vals > 0)
    plt.plot(x_vals[idx_in_range], y_vals[idx_in_range])
plt.show()


# %% [markdown]
# ## Pipeline

# %%
def sample_dmap_along_line(dmap_arr, coord1_min, coord1_max, coord2_const, coord1_samples):
    coord1 = np.round(np.linspace(coord1_min, coord1_max-1, num=coord1_samples)).astype(int)
    coord2 = np.full(coord1.shape, np.round(coord2_const)).astype(int)
    points = np.stack((coord1, coord2),axis=-1)
    dvals = dmap_arr[points[:,1],points[:,0]]
    return dvals

# Get height and width of a crack
# Will return a -1 value for a height or width if it cannot be found using this method
def naive_crack_size_estimation(list_tup_bbox, dmap, lane_lines, lane_img_shape, lane_width_m=3, top_bottom_depth_samples=5):
    width_estimation_list = []
    width_estimation_lane_lines_list = []
    height_estimation_list = []

    # a "list_tup_bbox" is a list of tupes containing tuples of bounding boxes of [x_min, y_min, x_max, y_max], a classification, and a probability/softmax output for the classification
    for tup in list_tup_bbox: # For each bounding box
        dmap_arr = np.array(dmap)
        (x_min, y_min, x_max, y_max), cls, prob = tup  
        
        # Width estimation
        x_mid = round((y_min+y_max)/2)
        y_mid = round((y_min+y_max)/2)
        left_lane_line_x = None
        left_line = None
        right_lane_line_x = None
        right_line = None
#         print("x_min:{0}, x_max:{1}".format(x_min, x_max))
        for line in lane_lines:
            x = (y_mid-line[1])/line[0]
#             print(line[0])
#             print(x)
            if (x < 0) or (x>lane_img_shape[1]): # shape[1] is x
                # The x is not on the image, skip this lane line
                continue
            # Now know that x is on image x domain
            if x < x_min:
                if left_lane_line_x is None or x > left_lane_line_x:
                    left_lane_line_x = x
                    left_line = line
            elif x > x_max:
                if right_lane_line_x is None or x < right_lane_line_x:
                    right_lane_line_x = x
                    right_line = line
            else: # x>=x_min and x<=x_max
                # lane goes through bbox, consider these lines automatically the closest to either 
                if x>=x_mid:
                    right_lane_line_x = x
                    right_line = line
                else: # x<x_mid
                    left_lane_line_x = x
                    left_line = line     
#             print("l is: {0}, r is {1}\n\n".format(left_lane_line_x, right_lane_line_x))
        lane_width_px = None
        if (left_lane_line_x is None) or (right_lane_line_x is None):
            width_estimation_list.append(-1) # Could not find two lines to give lane width
            width_estimation_lane_lines_list.append(None)
        else:
            lane_width_px = right_lane_line_x - left_lane_line_x
            bbox_width_px = x_max - x_min
            bbox_width_in_terms_of_lane_width = bbox_width_px/lane_width_px
            bbox_width_m = bbox_width_in_terms_of_lane_width * lane_width_m
            width_estimation_list.append(bbox_width_m)
            width_estimation_lane_lines_list.append((left_line, right_line))

        # Height estimation
        top_dvals = sample_dmap_along_line(dmap_arr, x_min, x_max, y_max, top_bottom_depth_samples) # m
        bottom_dvals = sample_dmap_along_line(dmap_arr, x_min, x_max, y_min, top_bottom_depth_samples) # m
        height_estimation_m = np.abs(np.mean(top_dvals) - np.mean(bottom_dvals)) # m
        height_estimation_list.append(height_estimation_m)
        
    return height_estimation_list, width_estimation_list, width_estimation_lane_lines_list
    


# %%
def plot_lane_line_in_range(x_vals, line, img_shape, ax, **plot_kwargs):
    y_vals = line[0]*x_vals + line[1]
    idx_in_range = np.logical_and(y_vals < img_shape[0]-1, y_vals > 0)
    ax.plot(x_vals[idx_in_range], y_vals[idx_in_range], **plot_kwargs)
    
def viz_pipeline(pil_img, list_tup_bbox, dmap, lane_detect_result_image, lane_lines, pred_heights, pred_widths, width_estimation_lane_lines, n_classes=6, cbar_min=None, cbar_max=None, **subplots_kwargs):
    lane_detect_result_image_x_vals = np.arange(lane_detect_result_image.shape[1], dtype=np.int)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, **subplots_kwargs)

    img_arr = np.array(pil_img)
    ax1.imshow(img_arr)
    ax1.set_title('Input Image')
    
    ax2.imshow(img_arr)
    cmap = get_cmap(len(list_tup_bbox), name='hsv_r')
    legend_list = []
    for i, tup_bbox in enumerate(list_tup_bbox):
        bb, cls, prob = tup_bbox
        color = cmap(i)
        x_min, y_min, x_max, y_max = bb
        # bb is [x_min, y_min, x_max, y_max]
        width = x_max-x_min
        height = y_max-y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x_max-18, y_max, cls, fontweight='heavy', color=color)
        
        if width_estimation_lane_lines[i] is not None:
            plot_lane_line_in_range(lane_detect_result_image_x_vals, width_estimation_lane_lines[i][0], lane_detect_result_image.shape, ax2, color=color, alpha=0.7, ls='--')
            plot_lane_line_in_range(lane_detect_result_image_x_vals, width_estimation_lane_lines[i][1], lane_detect_result_image.shape, ax2, color=color, alpha=0.7, ls='--')

        patch = patches.Patch(color=color, label='Predicted Height = {0}m\nPredicted Width = {1}m\n'.format(pred_heights[i], pred_widths[i]))
        legend_list.append(patch)
    ax2.legend(handles=legend_list, loc=(1.04,0))
    ax2.set_title('Size Estimation\ndetected crack bounding boxes and estimated height and width (-1 if unable to detect)\nIf width could be found, the lane lines used for lane width are plotted')
    
    sns.heatmap(dmap, cmap=sns.color_palette("Spectral_r", as_cmap=True), square=True, ax=ax3, vmin=cbar_min, vmax=cbar_max)
#     ax3.set_yticks([],[])
#     ax3.set_xticks([],[])
    ax3.set_title('Adabins Depth Map')
    
    
    ax4.imshow(lane_detect_result_image)
    for line in lane_lines:
        plot_lane_line_in_range(lane_detect_result_image_x_vals, line, lane_detect_result_image.shape, ax4)
    ax4.set_title('PINet Detected Lane Line Points with Hough Transform Detected Lane Lanes')

    fig.tight_layout()
    fig.show()
    plt.show()

# %%
test_blyncsy_image_filepath = blyncsy_image_filepaths[286]
# test_blyncsy_image_filepath = blyncsy_image_filepaths[155]
# test_blyncsy_image_filepath = blyncsy_image_filepaths[526]
# test_blyncsy_image_filepath = blyncsy_image_filepaths[524]
# test_blyncsy_image_filepath = random.choice(blyncsy_image_filepaths)
print(test_blyncsy_image_filepath)

test_blyncsy_pil_image = Image.open(test_blyncsy_image_filepath)

# FasterRCNN
fasterrcnn_model = torch.load(fasterrcnn_model_path)
fasterrcnn_model.eval()
with torch.no_grad():
    test_tensor = pil_to_FasterRCNN_input_tensor(test_blyncsy_pil_image, device)
    fasterrcnn_model.eval()
    out = fasterrcnn_model(test_tensor)
    list_tup_bbox = FasterRCNN_output_to_list_tup_bbox(out)
    filtered_list_tup_bbox = filter_list_tup_bbox(list_tup_bbox, 0.3)
#     print(filtered_list_tup_bbox)
#     viz_boxes(test_blyncsy_pil_image,filtered_list_tup_bbox,n_classes=6)

# AdaBins
min_depth = 1e-3
max_depth = 80
model = UnetAdaptiveBins.build(n_bins=256, min_val=min_depth, max_val=max_depth)
model, _, _ = load_checkpoint('./AdaBins/pretrained/AdaBins_kitti.pt', model)
model = model.to(device)
with torch.no_grad():
    model.eval()
    dmaps = adabins_predict_pil_images_anysize(model, [test_blyncsy_pil_image], (1241, 376), device, min_depth, max_depth)
    dmap_pred = dmaps[0]
    
# PINet
PINet_p = PINet_params()
PINet_p.threshold_point = 0.05
pin_model = lane_detection_network().to(device)
pin_state_dict = torch.load(os.path.join(PINet_src_root, 'savefile/0_tensor(0.5242)_lane_detection_network.pkl'))
pin_model.load_state_dict(pin_state_dict)
with torch.no_grad():
    pin_model.eval()
    in_x, in_y, lane_detect_result_image = PINet_predict_pil_anysize(pin_model, test_blyncsy_pil_image, device, p=PINet_p, give_result_image=True)
    lane_lines = get_lane_lines_hough_transform(in_x,in_y,lane_detect_result_image.shape,  y_min_max_tup=(400, lane_detect_result_image.shape[0]-1), line_abs_slope_min_cutoff=0.1, line_abs_slope_max_cutoff=100, ht_peaks_min_distance=200, ht_peaks_min_angle=30, ht_peaks_threshold=170)



# %%
pred_heights, pred_widths, width_estimation_lane_lines = naive_crack_size_estimation(filtered_list_tup_bbox, dmap_pred, lane_lines, lane_detect_result_image.shape, lane_width_m=3, top_bottom_depth_samples=5)



# %%
viz_pipeline(test_blyncsy_pil_image, filtered_list_tup_bbox, dmap_pred, lane_detect_result_image, lane_lines, pred_heights, pred_widths, width_estimation_lane_lines, n_classes=6, cbar_min=0, cbar_max=60, figsize=(40,5))


# %% [markdown]
# TODO: 
# - Batch prediction of images
# - Add perspective transform
# - Try YOLO, new FasterRCNN model
# - Vectorize Adabins prediction (will have to look at the prediction utility) and the entire pipeline, likely will have to create Datasets for loading images into tensor
#
# - Although don't have enough time, it would be smart to truly integrate each part of the pipeline using Pytorch and make each step ready for a batch, this would involve changing some of the post-processing operations to work with tensors but would in turn speed up pipeline and wrapping it in a Pytorch model would give a much quicker pipeline, this should be done once the pipeline is more concrete 

# %%
