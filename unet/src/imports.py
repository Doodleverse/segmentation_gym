# Written by Dr Daniel Buscombe, Marda Science LLC
# for  the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE zSOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from model_imports import *

# import os, shutil, json
# from skimage.io import imsave, imread
# from skimage.filters.rank import median
# from skimage.morphology import disk
# from scipy.ndimage import rotate

from glob import glob
import matplotlib.pyplot as plt
import numpy as np


##========================================================
def fromhex(n):
    """ hexadecimal to integer """
    return int(n, base=16)

##========================================================
def label_to_colors(
    img,
    mask,
    alpha,#=128,
    colormap,#=class_label_colormap, #px.colors.qualitative.G10,
    color_class_offset,#=0,
    do_alpha,#=True
):
    """
    Take MxN matrix containing integers representing labels and return an MxNx4
    matrix where each label has been replaced by a color looked up in colormap.
    colormap entries must be strings like plotly.express style colormaps.
    alpha is the value of the 4th channel
    color_class_offset allows adding a value to the color class index to force
    use of a particular range of colors in the colormap. This is useful for
    example if 0 means 'no class' but we want the color of class 1 to be
    colormap[0].
    """


    colormap = [
        tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
        for h in [c.replace("#", "") for c in colormap]
    ]

    cimg = np.zeros(img.shape[:2] + (3,), dtype="uint8")
    minc = np.min(img)
    maxc = np.max(img)

    for c in range(minc, maxc + 1):
        cimg[img == c] = colormap[(c + color_class_offset) % len(colormap)]

    cimg[mask==1] = (0,0,0)

    if do_alpha is True:
        return np.concatenate(
            (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
        )
    else:
        return cimg



##========================================================
def rescale(dat,
    mn,
    mx):
    '''
    rescales an input dat between mn and mx
    '''
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx-mn)*(dat-m)/(M-m)+mn

##====================================
def standardize(img):
    #standardization using adjusted standard deviation

    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0/np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    del m, s, N
    #
    if np.ndim(img)==2:
        img = np.dstack((img,img,img))

    return img

# ##========================================================
def inpaint_nans(im):
    ipn_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]]) # kernel for inpaint_nans
    nans = np.isnan(im)
    while np.sum(nans)>0:
        im[nans] = 0
        vNeighbors = convolve2d((nans==False),ipn_kernel,mode='same',boundary='symm')
        im2 = convolve2d(im,ipn_kernel,mode='same',boundary='symm')
        im2[vNeighbors>0] = im2[vNeighbors>0]/vNeighbors[vNeighbors>0]
        im2[vNeighbors==0] = np.nan
        im2[(nans==False)] = im[(nans==False)]
        im = im2
        nans = np.isnan(im)
    return im

#-----------------------------------
def plot_seg_history_iou(history, train_hist_fig):
    """
    "plot_seg_history_iou(history, train_hist_fig)"
    This function plots the training history of a model
    INPUTS:
        * history [dict]: the output dictionary of the model.fit() process, i.e. history = model.fit(...)
        * train_hist_fig [string]: the filename where the plot will be printed
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (figure printed to file)
    """
    n = len(history.history['val_loss'])

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.plot(np.arange(1,n+1), history.history['mean_iou'], 'b', label='train accuracy')
    plt.plot(np.arange(1,n+1), history.history['val_mean_iou'], 'k', label='validation accuracy')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Mean IoU Coefficient', fontsize=10)
    plt.legend(fontsize=10)

    plt.subplot(122)
    plt.plot(np.arange(1,n+1), history.history['loss'], 'b', label='train loss')
    plt.plot(np.arange(1,n+1), history.history['val_loss'], 'k', label='validation loss')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)

    # plt.show()
    plt.savefig(train_hist_fig, dpi=200, bbox_inches='tight')
#
# #-----------------------------------
# def crf_refine(label, img, nclasses = 2, theta_col=100, theta_spat=3, compat=120):
#     """
#     "crf_refine(label, img)"
#     This function refines a label image based on an input label image and the associated image
#     Uses a conditional random field algorithm using spatial and image features
#     INPUTS:
#         * label [ndarray]: label image 2D matrix of integers
#         * image [ndarray]: image 3D matrix of integers
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS: label [ndarray]: label image 2D matrix of integers
#     """
#
#     gx,gy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
#     # print(gx.shape)
#     img = np.dstack((img,gx,gy))
#
#     H = label.shape[0]
#     W = label.shape[1]
#     U = unary_from_labels(1+label,nclasses,gt_prob=0.51)
#     d = dcrf.DenseCRF2D(H, W, nclasses)
#     d.setUnaryEnergy(U)
#
#     # to add the color-independent term, where features are the locations only:
#     d.addPairwiseGaussian(sxy=(theta_spat, theta_spat),
#                  compat=3,
#                  kernel=dcrf.DIAG_KERNEL,
#                  normalization=dcrf.NORMALIZE_SYMMETRIC)
#     feats = create_pairwise_bilateral(
#                           sdims=(theta_col, theta_col),
#                           schan=(2,2,2),
#                           img=img,
#                           chdim=2)
#
#     d.addPairwiseEnergy(feats, compat=compat,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
#     Q = d.inference(20)
#     kl1 = d.klDivergence(Q)
#     return np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8), kl1
# #
#
# ###############################################################
# ### DATA FUNCTIONS
# ###############################################################
#
#
# #-----------------------------------
# def seg_file2tensor(f):
#     """
#     "seg_file2tensor(f)"
#     This function reads a jpeg image from file into a cropped and resized tensor,
#     for use in prediction with a trained segmentation model
#     INPUTS:
#         * f [string] file name of jpeg
#     OPTIONAL INPUTS: None
#     OUTPUTS:
#         * image [tensor array]: unstandardized image
#     GLOBAL INPUTS: TARGET_SIZE
#     """
#     bits = tf.io.read_file(f)
#     if 'jpg' in f:
#         image = tf.image.decode_jpeg(bits)
#     elif 'png' in f:
#         image = tf.image.decode_png(bits)
#
#     w = tf.shape(image)[0]
#     h = tf.shape(image)[1]
#     tw = TARGET_SIZE[0]
#     th = TARGET_SIZE[1]
#     resize_crit = (w * th) / (h * tw)
#     image = tf.cond(resize_crit < 1,
#                   lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
#                   lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
#                  )
#
#     nw = tf.shape(image)[0]
#     nh = tf.shape(image)[1]
#     image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
#     # image = tf.cast(image, tf.uint8) #/ 255.0
#
#     return image
#
#
