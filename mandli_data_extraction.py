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

# %%
import os
import xml.etree.ElementTree as ET
import glob
import matplotlib.pyplot as plt
import numpy as np

mandli_data_test_dir_path = '../0008_+'

# %% [markdown]
# # Misc Notes
# - I believe F_* means full size image while R_* means reduced size image
# - LCMS is Laser Crack Measurement System which is what was used to collect road distress and related information (here: Laser Crack Measurement System)
#     - The manual was obtained directly from Mandli, could not see a direct way to get hold of the manual on their website, I used a manual titled '202108_LCMS_DataAnalyser_Manual_v4.71.5.pdf'
# - I believe that the "Type" of a node may be this: 
# "The pavement type, as defined by the user. This setting will be ignored if the automatic pavement type detection is enabled. Possible values are: 1: Asphalt
# 2: Concrete
# 3: Grooved concrete (transversally)
# 4: Grooved concrete (longitudinally)
# 5: Highly textured (or porous)
# 6: Concrete CRCP (CRCP = Continuously Reinforced Concrete Pavement). *Concrete CRCP is concrete pavement with no transversal joints. The library is looking only for longitudinal joints when the pavement type is set"
#

# %%
os.listdir(mandli_data_test_dir_path)

# Front view monolcular images are contained here
front_view_test_dir_path = os.path.join(mandli_data_test_dir_path, '0008_+/Front')
print(os.listdir(front_view_test_dir_path))

LCMS_distress_test_dir_path = os.path.join(mandli_data_test_dir_path, 'LCMSDistress')
print(os.listdir(LCMS_distress_test_dir_path))

LCMS_visual_rng_test_dir_path = os.path.join(mandli_data_test_dir_path, '0008_+/RngOverlay_Down')
print(os.listdir(LCMS_visual_rng_test_dir_path))

LCMS_rng_test_dir_path = os.path.join(mandli_data_test_dir_path, '0008_+/Rng_Down')
print(os.listdir(LCMS_rng_test_dir_path))


# %% [markdown]
# # Visualizing Mandli Data
# Below shows how Mandli data front dashcam images and LCMS data line up, looking at the manhole cover it can be seen that the LCMS rng data is much closer to the bottom of the dashcam images and past a certain distance it ends up in the back of the LCMS data for the next frame.
#
# The horizontal crack at the bottom of the image of 00228 shows how the bottom of the image frame lines up with the rng LCMS data.
#
# The Manhole cover of the image of 00227 almost appears on 00227's LCMS data at the very top but appears in the LCMS rng data of 00228.
#
# In summary, roughly the bottom half of a mandli front view image lines up with the top half of the corresponding LCMS rng data.
# Translation to visualization given same size both front and rng images (and assuming camera perspective is fixed for all images) is **approximately (upon manual inspection)**:
#
# front image reduced size: y=0 ====> rng image full size: y=763
#
# front image reduced size: y=310 ====> rng image full size: y=2012

# %%
def get_front_img_filepaths(mandli_data_dir_path, name):
    return glob.glob(os.path.join(os.path.join(mandli_data_dir_path, '0008_+/Front'), '*/R_{0}.jpg'.format(name)), recursive=True)

def get_LCMS_distress_filepaths(mandli_data_dir_path, name):
    return glob.glob(os.path.join(os.path.join(mandli_data_dir_path, 'LCMSDistress'), '*_0{0}.xml'.format(name)), recursive=True)

def get_LCMS_visual_rng_filepaths(mandli_data_dir_path, name):
    return glob.glob(os.path.join(os.path.join(mandli_data_dir_path, '0008_+/RngOverlay_Down'), '*/F_{0}.jpg'.format(name)), recursive=True)

def get_LCMS_rng_filepaths(mandli_data_dir_path, name):
    return glob.glob(os.path.join(os.path.join(mandli_data_dir_path, '0008_+/Rng_Down'), '*/F_{0}.jpg'.format(name)), recursive=True)

def plot_front_and_visual_rng_and_rng(filenames, mandli_data_dir_path, rng_extra_plotting_func=None, print_shape=False, figsize=(20,20), near_viz_color='lime', far_viz_color='magenta', viz_linestyle= '-', viz_linewidth=3):
    for filename in filenames:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=figsize)
        file_front_img = get_front_img_filepaths(mandli_data_dir_path, filename)[0]
        front_img = np.flipud(plt.imread(file_front_img))
        ax1.imshow(front_img, origin='lower')
        ax1.axhline(y=310, color=far_viz_color, linestyle=viz_linestyle, linewidth=viz_linewidth, label='far')
        ax1.axhline(y=0, color=near_viz_color, linestyle=viz_linestyle, linewidth=viz_linewidth, label='near')

        file_LCMS_visual_rng_img = get_LCMS_visual_rng_filepaths(mandli_data_dir_path, filename)[0]
        LCMS_visual_rng_img = np.flipud(plt.imread(file_LCMS_visual_rng_img))
        ax2.imshow(LCMS_visual_rng_img, origin='lower')
        ax2.axhline(y=2012, color=far_viz_color, linestyle=viz_linestyle, linewidth=viz_linewidth, label='far')
        ax2.axhline(y=763, color=near_viz_color, linestyle=viz_linestyle, linewidth=viz_linewidth, label='near')

        file_LCMS_rng_img = get_LCMS_rng_filepaths(mandli_data_dir_path, filename)[0]
        LCMS_rng_img = np.flipud(plt.imread(file_LCMS_rng_img))
        ax3.imshow(LCMS_rng_img, origin='lower')
        ax3.axhline(y=2012, color=far_viz_color, linestyle=viz_linestyle, linewidth=viz_linewidth, label='far')
        ax3.axhline(y=763, color=near_viz_color, linestyle=viz_linestyle, linewidth=viz_linewidth, label='near')
        
        if rng_extra_plotting_func is not None:
            rng_extra_plotting_func(filename, fig, ax1, ax2, ax3)
        
        if print_shape:
            print('Front img shape: {0}'.format(front_img.shape))
            print('LCMS visual img shape: {0}'.format(LCMS_visual_rng_img.shape))
            print('LCMS img shape: {0}'.format(LCMS_rng_img.shape))
        
        ax1.legend()
        ax2.legend()
        ax3.legend()
        fig.suptitle(filename)
        fig.show()
        plt.show()
    
test_files_names = ['00227', '00228', '00229']
test_files_names.reverse() # Reverse so viz appears as driving forward

plot_front_and_visual_rng_and_rng(test_files_names, mandli_data_test_dir_path, figsize=(15,15), print_shape=True)


# %% [markdown]
# # Parse XML
# Useful info the manual: 
#
# - "The actual crack detection information is in the \<CrackList> element. The crack list may contain any number of \<Crack>. A \<Crack> has an identifier (\<CrackID>), a crack length (\<Length>) and a list of connected \<Node>. A \<Crack> element always has at least two \<Node>. A \<Node> is a point along a crack. It is defined by two coordinates (\<X> and \<Y>), a \<Width> value and a \<Depth> value. The \<Width> and \<Depth> values are the average severity and depth of the crack between two successive nodes. The node coordinates \<X> and \<Y> are given with respect to the lower-left corner of the result images (see LcmsGetResultImage documentation in Section 7.27). In order to determine the exact match between the \<X> and \<Y> coordinates of a crack from the XML file and the pixel coordinates of the same crack in the result images, one must use the GeneralParam_ResultImageResolution_mm parameter (see Section 6.1)."
#
# - All (X,Y) coordinates are returned with respect to the lower-left corner of the road section result image
#
# - In order to determine the exact match between the position of a joint in the XML file and its coordinates in pixel in the result images, one must use the GeneralParam_ResultImageResolution_mm parameter.
#     - Note: I assume this GeneralParam_ResultImageResolution_mm parameter is the mm of the width an height of an image pixel 

# %%
def LCMS_crack_data_extraction(LCMS_distress_filepath, should_give_crack_node_info=True):
    root = ET.parse(LCMS_distress_filepath).getroot()
    
    ProcessingParameters_tree = root.find('ProcessingInformation').find('ProcessingParameters')
    image_pixel_resolution_in_mm = float(ProcessingParameters_tree.find('GeneralParam_ResultImageResolution_mm').text)
    
    CrackInformation_tree = root.findall('CrackInformation')
    if not (len(CrackInformation_tree) == 1):
        print('No cracks found for {0}'.format(LCMS_distress_filepath))
        return
    CrackInformation_tree = CrackInformation_tree[0]

    Unit_tree = CrackInformation_tree.find('Unit')
    unit_dict = {}
    desired_crack_unit_info={'X', 'Y', 'Width', 'Depth', 'Length'}
    for child in Unit_tree:
        if not child.tag in desired_crack_unit_info:
            continue
        unit_dict[child.tag] = child.text
        
    CrackList_tree = CrackInformation_tree.find('CrackList')
    
    crack_dict = {}
    for crack_tree in CrackList_tree.findall('Crack'):
        crack_info = {}
        crack_info['Length'] = float(crack_tree.find('Length').text)
        crack_info['WeightedDepth'] = float(crack_tree.find('WeightedDepth').text)
        crack_info['WeightedWidth'] = float(crack_tree.find('WeightedWidth').text)
        if should_give_crack_node_info:
            crack_info['X'], crack_info['Y'], crack_info['Width'], crack_info['Depth'], crack_info['Type'] = extract_crack_node_info(crack_tree)
            crack_info['X_pixel'] = np.rint(crack_info['X']/image_pixel_resolution_in_mm)
            crack_info['Y_pixel'] = np.rint(crack_info['Y']/image_pixel_resolution_in_mm)

        crack_dict[crack_tree.find('CrackID').text] = crack_info

    return image_pixel_resolution_in_mm, unit_dict, crack_dict
    
def extract_crack_node_info(crack_tree):
    x_list = []
    y_list = []
    width_list =[] 
    depth_list = []
    type_list = []
    for node_tree in crack_tree.findall('Node'):
        x_list.append(node_tree.find('X').text)
        y_list.append(node_tree.find('Y').text)
        width_list.append(node_tree.find('Width').text)
        depth_list.append(node_tree.find('Depth').text)
        type_list.append(node_tree.find('Type').text)
    return np.array(x_list, dtype=np.float), np.array(y_list, dtype=np.float), np.array(width_list, dtype=np.float), np.array(depth_list, dtype=np.float), np.array(type_list, dtype=np.float)
    


# %% [markdown]
# I think '2': {'Length': '3.37',
#    'WeightedDepth': '5.37',
#    'WeightedWidth': '14.28',
#    'random_x_on_crack_mm': '410.0',
#    'random_y_on_crack_mm': '3320.0'}, is the long crack, bottom of image, middle of LCMS data crack in 0028
#    
#    
#    
#    
# End goal is to get some (front view) images labelled (can use segmenting tool) and associate segments with ground truth crack dimensions.
#
# Just need to find some good examples, so can be a rough heuristic, maybe just look at cracks that are over a certain length and fall on the bottom y values of the scan (because of how images line up with scan) and maybe also calculate statistics over the ``nodes''
#
#
#
# QUESTIONS:
#
# - How much data (in folder structure given) can I get?
# - What exaactly is "WeightedDepth" and "WeightedWidth" of a Crack, and what is "Type" of a Node?

# %%
def viz_crack(filename, fig, ax1, ax2, ax3, crack_info):
    x_points = crack_info['X_pixel']
    y_points = crack_info['Y_pixel']
    for x,y in zip(x_points,y_points):
        ax3.plot(x,y,'rx')
    


# %%
# test_files_names = ['00227', '00228', '00229']
test_files_names = ['00228']

test_files_names.reverse() # Reverse so viz appears as driving forward



mandli_data_dir_path = mandli_data_test_dir_path
filenames = test_files_names

for filename in filenames:
    file_LCMS_distress_xml = get_LCMS_distress_filepaths(mandli_data_dir_path, filename)[0]
    image_pixel_resolution_in_mm, unit_dict, crack_dict = LCMS_crack_data_extraction(file_LCMS_distress_xml)
    longest_crack_key = max(crack_dict, key=lambda k: crack_dict[k].get('Length', float('-inf')))
    longest_crack_viz_crack = lambda filename, fig, ax1, ax2, ax3: viz_crack(filename, fig, ax1, ax2, ax3, crack_dict[longest_crack_key])
#     plot_front_and_visual_rng_and_rng([filename], mandli_data_dir_path, figsize=(15,15), rng_extra_plotting_func=longest_crack_viz_crack)
    for crack_key in crack_dict:
        longest_crack_viz_crack = lambda filename, fig, ax1, ax2, ax3: viz_crack(filename, fig, ax1, ax2, ax3, crack_dict[crack_key])
        plot_front_and_visual_rng_and_rng([filename], mandli_data_dir_path, figsize=(15,15), rng_extra_plotting_func=longest_crack_viz_crack)

# %%

# %%
