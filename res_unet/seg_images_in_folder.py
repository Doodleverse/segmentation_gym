# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020, Marda Science LLC
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
USE_GPU = False #True

if USE_GPU == True:
   ##use the first available GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#utils
#keras functions for early stopping and model weights saving
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf #numerical operations on gpu

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import tensorflow.keras.backend as K
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
from skimage.filters.rank import median
from skimage.morphology import disk

from tkinter import filedialog
from tkinter import *
import json
from skimage.io import imsave
from skimage.transform import resize


#-----------------------------------
def crf_refine(label, img, nclasses,theta_col=100, mu=120, theta_spat=3, mu_spat=3):
    """
    "crf_refine(label, img)"
    This function refines a label image based on an input label image and the associated image
    Uses a conditional random field algorithm using spatial and image features
    INPUTS:
        * label [ndarray]: label image 2D matrix of integers
        * image [ndarray]: image 3D matrix of integers
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: label [ndarray]: label image 2D matrix of integers
    """

    H = label.shape[0]
    W = label.shape[1]
    U = unary_from_labels(1+label,nclasses,gt_prob=0.51)
    d = dcrf.DenseCRF2D(H, W, nclasses)
    d.setUnaryEnergy(U)

    # to add the color-independent term, where features are the locations only:
    d.addPairwiseGaussian(sxy=(theta_spat, theta_spat),
                 compat=mu_spat,
                 kernel=dcrf.DIAG_KERNEL,
                 normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats = create_pairwise_bilateral(
                          sdims=(theta_col, theta_col),
                          schan=(2,2,2),
                          img=img,
                          chdim=2)

    d.addPairwiseEnergy(feats, compat=mu,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    #kl1 = d.klDivergence(Q)
    return np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8)#, kl1

#-----------------------------------
def mean_iou(y_true, y_pred):
    """
    mean_iou(y_true, y_pred)
    This function computes the mean IoU between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

    INPUTS:
        * y_true: true masks, one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
        * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * IoU score [tensor]
    """
    yt0 = y_true[:,:,:,0]
    yp0 = tf.keras.backend.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou

#-----------------------------------
def dice_coef(y_true, y_pred):
    """
    dice_coef(y_true, y_pred)

    This function computes the mean Dice coefficient between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

    INPUTS:
        * y_true: true masks, one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
        * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * Dice score [tensor]
    """
    smooth = 1.
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

###############################################################
### MODEL FUNCTIONS
###############################################################
#-----------------------------------
def batchnorm_act(x):
    """
    batchnorm_act(x)
    This function applies batch normalization to a keras model layer, `x`, then a relu activation function
    INPUTS:
        * `z` : keras model layer (should be the output of a convolution or an input layer)
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * batch normalized and relu-activated `x`
    """
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)

#-----------------------------------
def conv_block(x, filters, kernel_size = (7,7), padding="same", strides=1):
    """
    conv_block(x, filters, kernel_size = (7,7), padding="same", strides=1)
    This function applies batch normalization to an input layer, then convolves with a 2D convol layer
    The two actions combined is called a convolutional block

    INPUTS:
        * `filters`: number of filters in the convolutional block
        * `x`:input keras layer to be convolved by the block
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the batch normalized convolution
    """
    conv = batchnorm_act(x)
    return tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)

#-----------------------------------
def bottleneck_block(x, filters, kernel_size = (7,7), padding="same", strides=1):
    """
    bottleneck_block(x, filters, kernel_size = (7,7), padding="same", strides=1)

    This function creates a bottleneck block layer, which is the addition of a convolution block and a batch normalized/activated block
    INPUTS:
        * `filters`: number of filters in the convolutional block
        * `x`: input keras layer
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the addition between convolutional and bottleneck layers
    """
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    bottleneck = batchnorm_act(bottleneck)

    return tf.keras.layers.Add()([conv, bottleneck])

#-----------------------------------
def res_block(x, filters, kernel_size = (7,7), padding="same", strides=1):
    """
    res_block(x, filters, kernel_size = (7,7), padding="same", strides=1)

    This function creates a residual block layer, which is the addition of a residual convolution block and a batch normalized/activated block
    INPUTS:
        * `filters`: number of filters in the convolutional block
        * `x`: input keras layer
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the addition between residual convolutional and bottleneck layers
    """
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    bottleneck = batchnorm_act(bottleneck)

    return tf.keras.layers.Add()([bottleneck, res])

#-----------------------------------
def upsamp_concat_block(x, xskip):
    """
    upsamp_concat_block(x, xskip)
    This function takes an input layer and creates a concatenation of an upsampled version and a residual or 'skip' connection
    INPUTS:
        * `xskip`: input keras layer (skip connection)
        * `x`: input keras layer
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the addition between residual convolutional and bottleneck layers
    """
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    return tf.keras.layers.Concatenate()([u, xskip])

#-----------------------------------
def iou(obs, est):
    IOU=0
    for n in range(1,5):
        component1 = obs==n
        component2 = est==n
        overlap = component1*component2 # Logical AND
        union = component1 + component2 # Logical OR
        calc = overlap.sum()/float(union.sum())
        if not np.isnan(calc):
            IOU += calc
        if IOU>1:
            IOU=IOU/n
    return IOU

#-----------------------------------
def res_unet(sz, f, nclasses=1):
    """
    res_unet(sz, f, nclasses=1)
    This function creates a custom residual U-Net model for image segmentation
    INPUTS:
        * `sz`: [tuple] size of input image
        * `f`: [int] number of filters in the convolutional block
        * flag: [string] if 'binary', the model will expect 2D masks and uses sigmoid. If 'multiclass', the model will expect 3D masks and uses softmax
        * nclasses [int]: number of classes
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras model
    """
    inputs = tf.keras.layers.Input(sz)

    ## downsample
    e1 = bottleneck_block(inputs, f); f = int(f*2)
    e2 = res_block(e1, f, strides=2); f = int(f*2)
    e3 = res_block(e2, f, strides=2); f = int(f*2)
    e4 = res_block(e3, f, strides=2); f = int(f*2)
    _ = res_block(e4, f, strides=2)

    ## bottleneck
    b0 = conv_block(_, f, strides=1)
    _ = conv_block(b0, f, strides=1)

    ## upsample
    _ = upsamp_concat_block(_, e4)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e3)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e2)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e1)
    _ = res_block(_, f)

    ## classify
    if nclasses==1:
        outputs = tf.keras.layers.Conv2D(nclasses, (1, 1), padding="same", activation="sigmoid")(_)
    else:
        outputs = tf.keras.layers.Conv2D(nclasses, (1, 1), padding="same", activation="softmax")(_)

    #model creation
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model


#-----------------------------------
def seg_file2tensor_noresize(f):
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
    image = tf.image.decode_jpeg(bits)

    return image


#-----------------------------------
def seg_file2tensor(f):
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
    image = tf.image.decode_jpeg(bits)

    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE[0]
    th = TARGET_SIZE[1]
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                 )

    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    # image = tf.cast(image, tf.uint8) #/ 255.0

    return image


#====================================================

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/weights",title = "Select file",filetypes = (("weights file","*.h5"),("all files","*.*")))
weights = root.filename
print(weights)
root.withdraw()


root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/samples",title = "Select directory of images to segment")
sample_direc = root.filename
print(sample_direc)
root.withdraw()


configfile = weights.replace('.h5','.json')

with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')

do_crf_refine


#=======================================================
model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], 3), BATCH_SIZE, nclasses)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])

model.load_weights(weights)


### predict
print('.....................................')
print('Using model for prediction on jpeg images ...')

sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))
print('Number of samples: %i' % (len(sample_filenames)))

for counter,f in enumerate(sample_filenames):
    image = seg_file2tensor(f)/255

    est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
    if nclasses>1:
        est_label = tf.argmax(est_label, axis=-1)
    else:
        est_label[est_label<1] = 0

    image = seg_file2tensor_noresize(f)/255
    nx,ny,nz = image.shape

    est_label = np.round(resize(est_label/nclasses,(nx,ny))*nclasses).astype(np.uint8)

    if do_crf_refine:
        est_label = crf_refine(est_label, (255*image.numpy().astype(np.uint8)),
                                theta_col=40, mu=40, theta_spat=1, mu_spat=1, nclasses = nclasses+1)

    if nclasses==1:
        imsave(f.replace('.jpg', '_predseg.png'), (est_label*255).astype(np.uint8))
        cmd = 'convert '+f+' \( '+f.replace('.jpg', '_predseg.png')+' -normalize +level 0,50% \) -compose screen -composite '+f.replace('.jpg', '_segoverlay.png')
        os.system(cmd)
    else:
        imsave(f.replace('.jpg', '_predseg.png'), (est_label).astype(np.uint8))
