
# Written by Dr Daniel Buscombe, Marda Science LLC
# for  the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021-22, Marda Science LLC
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


import os, time

import numpy as np
from skimage.filters.rank import median
from skimage.morphology import disk
from tkinter import filedialog
from tkinter import *
from tkinter import messagebox
import json
from skimage.io import imsave, imread
from numpy.lib.stride_tricks import as_strided as ast
from glob import glob

from joblib import Parallel, delayed
from skimage.morphology import remove_small_holes, remove_small_objects
from scipy.ndimage import maximum_filter
from skimage.transform import resize
from tqdm import tqdm
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

import tensorflow as tf #numerical operations on gpu
import tensorflow.keras.backend as K

from imports import standardize, label_to_colors, fromhex

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# #-----------------------------------
def seg_file2tensor_ND(f, TARGET_SIZE):#, resize):
    """
    "seg_file2tensor(f)"
    This function reads a NPZ image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of npz
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """

    with np.load(f) as data:
        bigimage = data['arr_0'].astype('uint8')

    smallimage = resize(bigimage,(TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True)
    #smallimage=bigimage.resize((TARGET_SIZE[1], TARGET_SIZE[0]))
    smallimage = np.array(smallimage)
    smallimage = tf.cast(smallimage, tf.uint8)

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return smallimage, w, h, bigimage

# #-----------------------------------
def seg_file2tensor_3band(f, TARGET_SIZE):#, resize):
    """
    "seg_file2tensor(f)"
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """

    bigimage = imread(f)#Image.open(f)
    smallimage = resize(bigimage,(TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True)
    #smallimage=bigimage.resize((TARGET_SIZE[1], TARGET_SIZE[0]))
    smallimage = np.array(smallimage)
    smallimage = tf.cast(smallimage, tf.uint8)

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return smallimage, w, h, bigimage

# =========================================================
def do_seg(f, M, metadatadict,sample_direc,NCLASSES,N_DATA_BANDS,TARGET_SIZE,TESTTIMEAUG):

    if f.endswith('jpg'):
    	segfile = f.replace('.jpg', '_predseg.png')
    elif f.endswith('png'):
    	segfile = f.replace('.png', '_predseg.png')
    elif f.endswith('npz'):# in f:
    	segfile = f.replace('.npz', '_predseg.png')

    metadatadict['input_file'] = f

    segfile = os.path.normpath(segfile)
    segfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'out'))

    try:
    	os.mkdir(os.path.normpath(sample_direc+os.sep+'out'))
    except:
    	pass

    metadatadict['nclasses'] = NCLASSES
    metadatadict['n_data_bands'] = N_DATA_BANDS

    if NCLASSES==1:

        if N_DATA_BANDS<=3:
            image, w, h, bigimage = seg_file2tensor_3band(f, TARGET_SIZE)
        else:
            image, w, h, bigimage = seg_file2tensor_ND(f, TARGET_SIZE)

        image = standardize(image.numpy()).squeeze()

        E0 = []; E1 = [];

        for counter,model in enumerate(M):
            #heatmap = make_gradcam_heatmap(tf.expand_dims(image, 0) , model)

            est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()

            if TESTTIMEAUG == True:
                #return the flipped prediction
                est_label2 = np.flipud(model.predict(tf.expand_dims(np.flipud(image), 0), batch_size=1).squeeze())
                est_label3 = np.fliplr(model.predict(tf.expand_dims(np.fliplr(image), 0), batch_size=1).squeeze())
                est_label4 = np.flipud(np.fliplr(model.predict(tf.expand_dims(np.flipud(np.fliplr(image)), 0), batch_size=1).squeeze()))

                #soft voting - sum the softmax scores to return the new TTA estimated softmax scores
                est_label = est_label + est_label2 + est_label3 + est_label4
                del est_label2, est_label3, est_label4

            E0.append(resize(est_label[:,:,0],(w,h), preserve_range=True, clip=True))
            E1.append(resize(est_label[:,:,1],(w,h), preserve_range=True, clip=True))
            del est_label

        #heatmap = resize(heatmap,(w,h), preserve_range=True, clip=True)
        K.clear_session()

        e0 = np.average(np.dstack(E0), axis=-1)#, weights=np.array(MW))

        del E0

        e1 = np.average(np.dstack(E1), axis=-1)#, weights=np.array(MW))
        del E1

        est_label = (e1+(1-e0))/2

        metadatadict['av_prob_stack'] = est_label

        del e0, e1

        thres = threshold_otsu(est_label)
        # print("Class threshold: %f" % (thres))
        est_label = (est_label>thres).astype('uint8')
        metadatadict['otsu_threshold'] = thres

    else: ###NCLASSES>1

        if N_DATA_BANDS<=3:
        	image, w, h, bigimage = seg_file2tensor_3band(f,TARGET_SIZE)#, resize=True)
        	w = w.numpy(); h = h.numpy()
        else:
            image, w, h, bigimage = seg_file2tensor_ND(f, TARGET_SIZE)

        #image = tf.image.per_image_standardization(image)
        image = standardize(image.numpy())
        #return the base prediction
        if N_DATA_BANDS==1:
            image = image[:,:,0]
            bigimage = np.dstack((bigimage,bigimage,bigimage))

        est_label = np.zeros((TARGET_SIZE[0],TARGET_SIZE[1], NCLASSES))
        for counter,model in enumerate(M):
            # heatmap = make_gradcam_heatmap(tf.expand_dims(image, 0) , model)

            est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()


            if TESTTIMEAUG == True:
                #return the flipped prediction
                est_label2 = np.flipud(model.predict(tf.expand_dims(np.flipud(image), 0), batch_size=1).squeeze())
                est_label3 = np.fliplr(model.predict(tf.expand_dims(np.fliplr(image), 0), batch_size=1).squeeze())
                est_label4 = np.flipud(np.fliplr(model.predict(tf.expand_dims(np.flipud(np.fliplr(image)), 0), batch_size=1).squeeze()))

                #soft voting - sum the softmax scores to return the new TTA estimated softmax scores
                est_label = est_label + est_label2 + est_label3 + est_label4
                del est_label2, est_label3, est_label4

            K.clear_session()

        est_label /= counter+1
        est_label = resize(est_label,(w,h))

        metadatadict['av_prob_stack'] = est_label

        est_label = np.argmax(est_label, -1)
        metadatadict['otsu_threshold'] = np.nan

    #heatmap = resize(heatmap,(w,h), preserve_range=True, clip=True)

    class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477','#66AA00','#B82E2E', '#316395']
    #add classes for more than 10 classes

    if NCLASSES>1:
        class_label_colormap = class_label_colormap[:NCLASSES]
    else:
        class_label_colormap = class_label_colormap[:2]

    metadatadict['color_segmentation_output'] = segfile

    try:
        color_label = label_to_colors(est_label, bigimage.numpy()[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
    except:
        color_label = label_to_colors(est_label, bigimage[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)

    if 'jpg' in f:
        imsave(segfile, (color_label).astype(np.uint8), check_contrast=False)
    elif 'png' in f:
        imsave(segfile, (color_label).astype(np.uint8), check_contrast=False)

    metadatadict['color_segmentation_output'] = segfile

#   segfile = segfile.replace('_meta.npz','_res.npz')
    segfile = segfile.replace('_predseg.png','_res.npz')

    # datadict['color_label'] = color_label
    metadatadict['grey_label'] = est_label
    # datadict['image_fullsize'] = bigimage
    # datadict['image_targetsize'] = image

    np.savez_compressed(segfile, **metadatadict)

    segfile = segfile.replace('_res.npz','_overlay.png')

    if N_DATA_BANDS<=3:
        plt.imshow(bigimage)
    else:
        plt.imshow(bigimage[:,:,:3])

    plt.imshow(color_label, alpha=0.5);
    plt.axis('off')
    # plt.show()
    plt.savefig(segfile, dpi=200, bbox_inches='tight')
    plt.close('all')


#--------------------------------------------------------
def make_gradcam_heatmap(image, model):

    # Remove last layer's softmax
    model.layers[-2].activation = None

    last_conv_layer_name = model.layers[-39].name
    # print(last_conv_layer_name)

   # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output], trainable=False
    )

    #then gradient of the output with respect to the output feature map of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(image)

        grads = tape.gradient(preds, last_conv_layer_output)
    #mean intensity of the gradient
    # importance of each channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    #normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    heatmap = heatmap.numpy().squeeze()

    #plt.imshow(image.numpy().squeeze()); plt.imshow(heatmap, cmap='bwr',alpha=0.5); plt.savefig('tmp.png')

    return heatmap



###### for patches-based segmentation

def window2d(window_func, window_size, **kwargs):
    '''
    Generates a 2D square image (of size window_size) containing a 2D user-defined
    window with values ranging from 0 to 1.
    It is possible to pass arguments to the window function by setting kwargs.
    All available windows: https://docs.scipy.org/doc/scipy/reference/signal.windows.html
    '''
    window = np.matrix(window_func(M=window_size, sym=False, **kwargs))
    return window.T.dot(window)

def generate_corner_windows(window_func, window_size, **kwargs):
    step = window_size >> 1
    window = window2d(window_func, window_size, **kwargs)
    window_u = np.vstack([np.tile(window[step:step+1, :], (step, 1)), window[step:, :]])
    window_b = np.vstack([window[:step, :], np.tile(window[step:step+1, :], (step, 1))])
    window_l = np.hstack([np.tile(window[:, step:step+1], (1, step)), window[:, step:]])
    window_r = np.hstack([window[:, :step], np.tile(window[:, step:step+1], (1, step))])
    window_ul = np.block([
        [np.ones((step, step)), window_u[:step, step:]],
        [window_l[step:, :step], window_l[step:, step:]]])
    window_ur = np.block([
        [window_u[:step, :step], np.ones((step, step))],
        [window_r[step:, :step], window_r[step:, step:]]])
    window_bl = np.block([
        [window_l[:step, :step], window_l[:step, step:]],
        [np.ones((step, step)), window_b[step:, step:]]])
    window_br = np.block([
        [window_r[:step, :step], window_r[:step, step:]],
        [window_b[step:, :step], np.ones((step, step))]])
    return np.array([
        [ window_ul, window_u, window_ur ],
        [ window_l,  window,   window_r  ],
        [ window_bl, window_b, window_br ],
    ])


def generate_patch_list(image_width, image_height, window_func, window_size, overlapping=False):
    patch_list = []
    if overlapping:
        step = window_size >> 1
        windows = generate_corner_windows(window_func, window_size)
        max_height = int(image_height/step - 1)*step
        max_width = int(image_width/step - 1)*step
    else:
        step = window_size
        windows = np.ones((window_size, window_size))
        max_height = int(image_height/step)*step
        max_width = int(image_width/step)*step
    for i in range(0, max_height, step):
        for j in range(0, max_width, step):
            if overlapping:
                # Close to border and corner cases
                # Default (1, 1) is regular center window
                border_x, border_y = 1, 1
                if i == 0: border_x = 0
                if j == 0: border_y = 0
                if i == max_height-step: border_x = 2
                if j == max_width-step: border_y = 2
                # Selecting the right window
                current_window = windows[border_x, border_y]
            else:
                current_window = windows
            # The patch is cropped when the patch size is not
            # a multiple of the image size.
            patch_height = window_size
            if i+patch_height > image_height:
                patch_height = image_height - i
            patch_width = window_size
            if j+patch_width > image_width:
                patch_width = image_width - j
            # Adding the patch
            patch_list.append(
                (j, i, patch_width, patch_height, current_window[:patch_height, :patch_width])
            )
    return patch_list


# =========================================================
def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')

# =========================================================
# Return a sliding window over a in any number of dimensions
# version with no memory mapping
def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    '''
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)

    return a.reshape(dim), newshape
