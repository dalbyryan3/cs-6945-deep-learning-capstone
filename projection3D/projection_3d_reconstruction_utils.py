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

# %% id="Hyw70gGBll_x"
from skimage import io
from skimage import color
from skimage import exposure
from skimage.segmentation import flood, flood_fill
import matplotlib.pyplot as plt
import numpy as np 


# %% [markdown] id="rGHzWjAt9orN"
# ## Load image

# %% colab={"base_uri": "https://localhost:8080/", "height": 252} id="gjecbmN5lwV6" outputId="7166e49c-5b72-40b3-dfc4-f9f904c7bc7a"
# filename = "image1.jpg"
# img = io.imread(filename)
# plt.imshow(img)
# plt.title("sample image")
# plt.show()

# %% [markdown] id="t8lYvcR19sLV"
# ## Initialize 2d points
# We pass in control_point2D as an array. [x,y,depth] Depth value came from adabins. \\
# Ideally, this array is 4*3. (four corners of boundary box * (x,y,depth)) \\
# Currently, all defualt depth is one

# %% colab={"base_uri": "https://localhost:8080/", "height": 252} id="jabzvQecqJGd" outputId="1a367a6c-7230-4bb4-f670-71056c4155c2"
# Control point for 2D
# control_point2D = [[330,531,1],[354,531,1],[384,531,1],[407,532,1],[437,532,1],[459,532,1],[331,589,1],[356,587,1],[385,588,1],[409,587,1],[332,646,1],[356,646,1],[491,641,1],[512,641,1],[297,511,1],[622,515,1],[299,722,1],[624,703,1],[271,852,1],[290,857,1],[310,863,1],[607,821,1],[630,824,1],[657,827,1]]
# for i in range(len(control_point2D)):
#   control_point2D[i][0]*=control_point2D[i][2]
#   control_point2D[i][1]*=control_point2D[i][2]
# control_point2D = np.array(control_point2D)
# fig = plt.figure()

# plt.imshow(img)
# plt.title("2d points on image")
# for (x,y,z) in control_point2D:
#   plt.scatter(x,y,c='#ff7f0e',s=2)
# plt.show()

# %% [markdown] id="8mq4WXB5-tzN"
# ## Transform functions
# It used for setting up the svd transformation with the camera parameter

# %% id="GvzefqQo-me0"
from random import sample 
from random import randint
import numpy as np

# Input pair: x1,y1, source
# Output pair: x'1,y'1, target
def build_up_matrix(source_pair,target_pair,number_of_points):
  N = len(source_pair)
  source_pair = np.array(source_pair)

  source_pair_sample = []
  target_pair_sample = []
  index_array = np.arange(N)
  # if too many coorespondences, do a sampling
  if N>number_of_points:
    for i in range(N-number_of_points):
      remove_index = randint(0, len(source_pair)-1)
      source_pair = np.delete(source_pair,remove_index,axis=0)
      target_pair = np.delete(target_pair,remove_index,axis=0)
    N=number_of_points

  height = N*3
  width = 11
  # source 2d point
  # taget 3d point
  P = np.zeros((height,width))
  b = np.zeros(height)
  for i in range(N):
    P[i][0] = source_pair[i][0]
    P[i][1] = source_pair[i][1]
    P[i][2] = 1

    P[i+N][3] = source_pair[i][0]
    P[i+N][4] = source_pair[i][1]
    P[i+N][5] = 1

    P[i+2*N][6] = source_pair[i][0]
    P[i+2*N][7] = source_pair[i][1]
    P[i+2*N][8] = 1
  for i in range(height):
    if i<N:
      P[i][9] = -1*source_pair[i][0]*target_pair[i][0]
      P[i][10] = -1*source_pair[i][1]*target_pair[i][0]
      b[i] = target_pair[i][0]
    elif i<2*N:
      P[i][9] = -1*source_pair[i-N][0]*target_pair[i-N][1]
      P[i][10] = -1*source_pair[i-N][1]*target_pair[i-N][1]
      b[i] = target_pair[i-N][1]
    elif i<3*N:
      P[i][9] = -1*source_pair[i-2*N][0]*target_pair[i-2*N][1]
      P[i][10] = -1*source_pair[i-2*N][1]*target_pair[i-2*N][1]
      b[i] = target_pair[i-2*N][1]
  return np.asarray(P),np.asarray(b)


# %% id="TH4R1thX-n0k"
def svd(A,b):
  #print(A)
  #print(b.shape)
  try:
     u,s,vh = np.linalg.svd(A)
  except:
      print("non_invertable")
      empty_np = np.zeros(10)
      return empty_np
  inver_s = np.zeros((vh.shape[0],u.shape[0]))
  for i in range(len(s)):
    inver_s[i][i] = 1/s[i]
    if inver_s[i][i] > 9999999:
        inver_s[i][i] = 0

  inverse_A = np.matmul(np.matrix.transpose(vh) ,np.matmul(inver_s, np.matrix.transpose(u)))
  #print(inverse_A)
  #print(b.shape)
  x = np.matmul(inverse_A,b)
  return x
  


# %% [markdown] id="yeEHVVyO-7G3"
# ## Initialize the camera paramemter

# %% colab={"base_uri": "https://localhost:8080/"} id="8auYKoFZ0Lhf" outputId="579883d6-2a07-4fa4-cdb9-5d9454ecd17c"
# P is a 3*3 camera Intrinsic Matrix

# IntrinsicMatrix = [722.7379,0,0,0,726.1398,0,663.8862,335.0579,1] 
# IntrinsicMatrix = np.reshape(IntrinsicMatrix,(3,3))

# # Elements in IntrinsicMatrix: [fx,0,cx,0,fy,cy,0,0,1]
# IntrinsicMatrix = np.transpose(IntrinsicMatrix)

# # Extrinsic Matrix, we assume the camera is (0,0,0). So the matrix is identity + [0,0,0] translation
# ExtrinsicMatrix = [1,0,0,0,0,1,0,0,0,0,1,0]
# ExtrinsicMatrix = np.reshape(ExtrinsicMatrix,(3,4))

# print(IntrinsicMatrix)
# print(ExtrinsicMatrix)

# Perspective_matrix = np.matmul(IntrinsicMatrix,ExtrinsicMatrix)
# print(Perspective_matrix)

#3*1 = (3*4)*(4*1)

# %% id="ieCrJKnT-pWj"
def perspective_transform(x,y,p):
  # P is a 4*3 matrix 
  #print(p)
  new_x = (p[0][0]*x+p[0][1]*y+p[0][2])/(p[3][0]*x+p[3][1]*y+p[3][2])
  new_y = (p[1][0]*x+p[1][1]*y+p[1][2])/(p[3][0]*x+p[3][1]*y+p[3][2])
  new_z = (p[2][0]*x+p[2][1]*y+p[2][2])/(p[3][0]*x+p[3][1]*y+p[3][2])
  return (new_x,new_y,new_z)
def perspective_transform_matlab(x,y,p):
  # P is 3*3
  new_x = p[0][0]*x+p[0][1]*y+p[0][2]
  new_y = p[1][0]*x+p[1][1]*y+p[1][2]
  new_z = p[2][0]*x+p[2][1]*y+p[2][2]
  return (new_x,new_y,new_z)
  


# %% id="kBSnvT1BYiOe"
# conrtol_point2D is N*3 (x,y,depth), return 3 list (x,y,z)
# return x_list[N], y_list [N], z_list[N]
def transform(control_point2D, perspective_matrix):
  x_3d_list = []
  y_3d_list = []
  z_3d_list = []

  for i in range(len(control_point2D)):
    #2d = P*3d, x will be 4d, b will be 3d
    temp_3d = []
    temp_3d.append(control_point2D[i][0]) # x in image2d
    temp_3d.append(control_point2D[i][1]) # y in image2d
    temp_3d.append(control_point2D[i][2]) # depth value in imag
    x = svd(perspective_matrix,np.array(temp_3d))
    # x will be a 4d point, represnt for x,y,z,w(homogenous coordinate)
    # But if camera is the center, x[3] will always be zero
    x_3d_list.append(x[0])
    y_3d_list.append(x[1])
    z_3d_list.append(x[2])
  return x_3d_list,y_3d_list,z_3d_list


# %% [markdown] id="UFBkter5CTVl"
# ## Run the transformation

# %% id="-QhGJz8n_9jc"
# This will transform all the 2d point to 3d
# x_3d_list,y_3d_list,z_3d_list = transform(control_point2D)

# # %% colab={"base_uri": "https://localhost:8080/", "height": 665} id="58iCzP9FZnlh" outputId="255759a5-41dc-4a76-bb97-8a44741d5a03"
# for i in range(24):
#   print(x_3d_list[i],y_3d_list[i],z_3d_list[i])
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(x_3d_list,y_3d_list,z_3d_list, marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
