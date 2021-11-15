
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

#-----------------------------------
def seg_file2tensor_4band(f, fir, TARGET_SIZE,resize):
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
    bits = tf.io.read_file(f)

    if 'jpg' in f:
        bigimage = tf.image.decode_jpeg(bits)
    elif 'png' in f:
        bigimage = tf.image.decode_png(bits)

    bits = tf.io.read_file(fir)
    if 'jpg' in fir:
        nir = tf.image.decode_jpeg(bits)
    elif 'png' in f:
        nir = tf.image.decode_png(bits)

    bigimage = tf.concat([bigimage, nir],-1)[:,:,:N_DATA_BANDS]

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    if resize:

        tw = TARGET_SIZE[0]
        th = TARGET_SIZE[1]
        resize_crit = (w * th) / (h * tw)
        image = tf.cond(resize_crit < 1,
                      lambda: tf.image.resize(bigimage, [w*tw/w, h*tw/w]), # if true
                      lambda: tf.image.resize(bigimage, [w*th/h, h*th/h])  # if false
                     )

        nw = tf.shape(image)[0]
        nh = tf.shape(image)[1]
        image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
        # image = tf.cast(image, tf.uint8) #/ 255.0

        return image, w, h, bigimage

    else:
        return None, w, h, bigimage


# =========================================================
def do_seg(f, M, metadatadict,sample_direc,NCLASSES,N_DATA_BANDS,TARGET_SIZE,temp=0):

    if 'jpg' in f:
    	segfile = f.replace('.jpg', '_predseg.png')
    elif 'png' in f:
    	segfile = f.replace('.png', '_predseg.png')

    metadatadict['input_file'] = f

    segfile = os.path.normpath(segfile)
    segfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'out'))

    try:
    	os.mkdir(os.path.normpath(sample_direc+os.sep+'out'))
    except:
    	pass

    metadatadict['nclasses'] = NCLASSES
    metadatadict['n_data_bands'] = N_DATA_BANDS

    datadict={}

    if NCLASSES==1:

        if N_DATA_BANDS<=3:
            image, w, h, bigimage = seg_file2tensor_3band(f, TARGET_SIZE)#, resize=True)
        if image is None:
            image = bigimage#/255
            #bigimage = bigimage#/255
            w = w.numpy(); h = h.numpy()
        else:
            image, w, h, bigimage = seg_file2tensor_4band(f, f.replace('aug_images', 'aug_nir'), TARGET_SIZE,resize=True )
            if image is None:
                image = bigimage#/255
                w = w.numpy(); h = h.numpy()

        print("Working on %i x %i image" % (w,h))

        image = standardize(image.numpy()).squeeze()

        E0 = []; E1 = [];

        for counter,model in enumerate(M):
            heatmap = make_gradcam_heatmap(tf.expand_dims(image, 0) , model)

            est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
            print('Model {} applied'.format(counter))
            E0.append(resize(est_label[:,:,0],(w,h), preserve_range=True, clip=True))
            E1.append(resize(est_label[:,:,1],(w,h), preserve_range=True, clip=True))
            del est_label

        heatmap = resize(heatmap,(w,h), preserve_range=True, clip=True)
        K.clear_session()

        e0 = np.average(np.dstack(E0), axis=-1)#, weights=np.array(MW))

        del E0

        e1 = np.average(np.dstack(E1), axis=-1)#, weights=np.array(MW))
        del E1

        est_label = (e1+(1-e0))/2

        datadict['av_prob_stack'] = est_label

        del e0, e1

        thres = threshold_otsu(est_label)+temp
        print("Class threshold: %f" % (thres))
        est_label = (est_label>thres).astype('uint8')
        metadatadict['otsu_threshold'] = thres

    else: ###NCLASSES>1

        if N_DATA_BANDS<=3:
        	image, w, h, bigimage = seg_file2tensor_3band(f,TARGET_SIZE)#, resize=True)
        	image = image#/255
        	bigimage = bigimage#/255
        	w = w.numpy(); h = h.numpy()
        else:
        	image, w, h, bigimage = seg_file2tensor_4band(f, f.replace('aug_images', 'aug_nir'), TARGET_SIZE, resize=True )
        	image = image#/255
        	bigimage = bigimage#/255
        	w = w.numpy(); h = h.numpy()

        print("Working on %i x %i image" % (w,h))

        #image = tf.image.per_image_standardization(image)
        image = standardize(image.numpy())


        est_label = np.zeros((TARGET_SIZE[0],TARGET_SIZE[1], NCLASSES))
        for counter,model in enumerate(M):
            heatmap = make_gradcam_heatmap(tf.expand_dims(image, 0) , model)

            # est_label = model.predict(tf.expand_dims(image, 0 , batch_size=1).squeeze()
            est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
            est_label += resize(est_label,(TARGET_SIZE[0],TARGET_SIZE[1]))
            K.clear_session()

        est_label /= counter+1
        est_label = resize(est_label,(w,h))

        datadict['av_prob_stack'] = est_label

        est_label = np.argmax(est_label, -1)
        metadatadict['otsu_threshold'] = np.nan

    heatmap = resize(heatmap,(w,h), preserve_range=True, clip=True)

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

    segfile = segfile.replace('.png','_meta.npz')

    np.savez_compressed(segfile, **metadatadict)

    segfile = segfile.replace('_meta.npz','_res.npz')

    # datadict['color_label'] = color_label
    datadict['grey_label'] = est_label
    # datadict['image_fullsize'] = bigimage
    # datadict['image_targetsize'] = image

    np.savez_compressed(segfile, **datadict)

    segfile = segfile.replace('_res.npz','_overlay.png')

    plt.imshow(bigimage); plt.imshow(color_label, alpha=0.5);
    plt.axis('off')
    # plt.show()
    plt.savefig(segfile, dpi=200, bbox_inches='tight')
    plt.close('all')

    segfile = segfile.replace('_overlay.png','_gradcam.png')

    plt.imshow(bigimage); plt.imshow(heatmap, cmap='bwr', alpha=0.5);
    plt.axis('off')
    # plt.show()
    plt.savefig(segfile, dpi=200, bbox_inches='tight')
    plt.close('all')



def make_gradcam_heatmap(image, model):

    # Remove last layer's softmax
    model.layers[-2].activation = None

    last_conv_layer_name = model.layers[-39].name
    print(last_conv_layer_name)

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
