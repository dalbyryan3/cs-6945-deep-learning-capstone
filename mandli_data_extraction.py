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

mandli_data_test_dir_path = '../0008_+'

# %% [markdown]
# # Misc Notes
# - I believe F_* means full size image while R_* means reduced size image
# - LCMS is Laser Crack Measurement System which is what was used to collect road distress and related information (here: Laser Crack Measurement System)
#     - The manual was obtained directly from Mandli, could not see a direct way to get hold of the manual on their website, I used a manual titled '202108_LCMS_DataAnalyser_Manual_v4.71.5.pdf'
#

# %%
os.listdir(mandli_data_test_dir_path)

# Front view monolcular images are contained here
front_view_test_dir_path = os.path.join(mandli_data_test_dir_path, '0008_+/Front')
print(os.listdir(front_view_test_dir_path))

LCMS_distress_test_dir_path = os.path.join(mandli_data_test_dir_path, 'LCMSDistress')
print(os.listdir(LCMS_distress_test_dir_path))

LCMS_visual_test_dir_path = os.path.join(mandli_data_test_dir_path, '0008_+/RngOverlay_Down')
print(os.listdir(LCMS_visual_test_dir_path))


# %% [markdown]
# # Visualizing Mandli Data
# Below shows how Mandli data front dashcam images and LCMS data line up, looking at the manhole cover it can be seen that the LCMS data is much closer to the bottom of the dashcam images and past a certain distance it ends up in the back of the LCMS data for the next frame.
#
# The horizontal crack at the bottom of the image of 00228 shows how the bottom of the image frame lines up with the LCMS data.
#
# The Manhole cover of the image of 00227 almost appears on 00227's LCMS data at the very top but appears in the LCMS data of 00228.
#
# In summary, roughly the bottom half of a mandli front view image lines up with the top half of the corresponding LCMS data.

# %%
def get_front_img_filepaths(front_dir_path, name):
    return glob.glob(os.path.join(front_dir_path, '*/R_{0}.jpg'.format(name)), recursive=True)

def get_LCMS_distress_filepaths(LCMS_distress_dir_path, name):
    return glob.glob(os.path.join(LCMS_distress_dir_path, '*_0{0}.xml'.format(name)), recursive=True)

def get_LCMS_visual_filepaths(LCMS_visual_dir_path, name):
    return glob.glob(os.path.join(LCMS_visual_dir_path, '*/F_{0}.jpg'.format(name)), recursive=True)

test_files_names = ['00227', '00228', '00229']
test_files_names.reverse() # Reverse so viz appears as driving forward

for test_file_name in test_files_names:
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
    test_file_front_img = get_front_img_filepaths(front_view_test_dir_path, test_file_name)[0]
#     print(test_file_front_img)
    ax1.imshow(plt.imread(test_file_front_img))

    test_file_LCMS_distress_xml = get_LCMS_distress_filepaths(LCMS_distress_test_dir_path, test_file_name)[0]
#     print(test_file_LCMS_distress_xml)

    test_file_LCMS_visual_img = get_LCMS_visual_filepaths(LCMS_visual_test_dir_path, test_file_name)[0]
#     print(test_file_LCMS_visual_img)
    ax2.imshow(plt.imread(test_file_LCMS_visual_img))

    fig.suptitle(test_file_name)
    fig.show()
    plt.show()


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
def LCMS_crack_data_extraction(LCMS_distress_filepath, desired_crack_info={'X', 'Y', 'Width', 'Depth', 'Length'}):
    root = ET.parse(LCMS_distress_filepath).getroot()
    
    ProcessingParameters_tree = root.find('ProcessingInformation').find('ProcessingParameters')
    image_pixel_mm_resolution = ProcessingParameters_tree.find('GeneralParam_ResultImageResolution_mm').text
    
    CrackInformation_tree = root.findall('CrackInformation')
    if not (len(CrackInformation_tree) == 1):
        print('No cracks found for {0}'.format(LCMS_distress_filepath))
        return
    CrackInformation_tree = CrackInformation_tree[0]

    Unit_tree = CrackInformation_tree.find('Unit')
    unit_dict = {}
    for child in Unit_tree:
        if not child.tag in desired_crack_info:
            continue
        unit_dict[child.tag] = child.text
        
    CrackList_tree = CrackInformation_tree.find('CrackList')
    
    crack_dict = {}
    for crack_tree in CrackList_tree.findall('Crack'):
        crack_info = {}
        crack_info['Length'] = crack_tree.find('Length').text
        crack_info['WeightedDepth'] = crack_tree.find('WeightedDepth').text
        crack_info['WeightedWidth'] = crack_tree.find('WeightedWidth').text
        crack_info['random_x_on_crack_mm'] = crack_tree.find('Node').find('X').text
        crack_info['random_y_on_crack_mm'] = crack_tree.find('Node').find('Y').text



        crack_dict[crack_tree.find('CrackID').text] = crack_info

    
    return image_pixel_mm_resolution, unit_dict, crack_dict
    
test_file_name = '00228'
test_file_LCMS_distress_xml = get_LCMS_distress_filepaths(LCMS_distress_test_dir_path, test_file_name)[0]
LCMS_crack_data_extraction(test_file_LCMS_distress_xml)


# %% [markdown]
# I think '2': {'Length': '3.37',
#    'WeightedDepth': '5.37',
#    'WeightedWidth': '14.28',
#    'random_x_on_crack_mm': '410.0',
#    'random_y_on_crack_mm': '3320.0'}, is the long crack, bottom of image, middle of LCMS data crack in 0028

# %%
