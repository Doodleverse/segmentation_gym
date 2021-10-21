# Written by Dr Daniel Buscombe, Marda Science LLC
# for  the USGS Coastal Change Hazards Program
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
# OUT OF OR IN CONNECTION WITH THE zSOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from skimage.io import imsave, imread
from glob import glob
from skimage.filters.rank import median
from skimage.morphology import disk
from scipy.ndimage import rotate

import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import os, shutil, json
#keras functions for early stopping and model weights saving
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())

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
    #img = rescale(img, 0, 1)
    del m, s, N
    #
    # if np.ndim(img)!=3:
    #     img = np.dstack((img,img,img))

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
#

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
def iou(obs, est,nclasses):
    IOU=0
    for n in range(1,nclasses):
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
def res_unet(sz, f, nclasses=1, kernel_size=(7,7)):
    """
    res_unet(sz, f, nclasses=1)
    This function creates a custom residual U-Net model for image segmentation
    INPUTS:
        * `sz`: [tuple] size of input image
        * `f`: [int] number of filters in the convolutional block
        * flag: [string] if 'binary', the model will expect 2D masks and uses sigmoid. If 'multiclass', the model will expect 3D masks and uses softmax
        * nclasses [int]: number of classes
    OPTIONAL INPUTS:
        * `kernel_size`=(7, 7): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras model
    """
    inputs = tf.keras.layers.Input(sz)

    ## downsample
    e1 = bottleneck_block(inputs, f);
    f = int(f*2)
    e2 = res_block(e1, f, strides=2,  kernel_size = kernel_size)
    f = int(f*2)
    e3 = res_block(e2, f, strides=2,  kernel_size = kernel_size)
    f = int(f*2)
    e4 = res_block(e3, f, strides=2,  kernel_size = kernel_size)
    f = int(f*2)
    _ = res_block(e4, f, strides=2, kernel_size = kernel_size)

    ## bottleneck
    b0 = conv_block(_, f, strides=1,  kernel_size = kernel_size)
    _ = conv_block(b0, f, strides=1,  kernel_size = kernel_size)

    ## upsample
    _ = upsamp_concat_block(_, e4)
    _ = res_block(_, f, kernel_size = kernel_size)
    f = int(f/2)

    _ = upsamp_concat_block(_, e3)
    _ = res_block(_, f, kernel_size = kernel_size)
    f = int(f/2)

    _ = upsamp_concat_block(_, e2)
    _ = res_block(_, f, kernel_size = kernel_size)
    f = int(f/2)

    _ = upsamp_concat_block(_, e1)
    _ = res_block(_, f, kernel_size = kernel_size)

    # ## classify
    if nclasses==1:
        outputs = tf.keras.layers.Conv2D(nclasses, (1, 1), padding="same", activation="sigmoid")(_)
    else:
        outputs = tf.keras.layers.Conv2D(nclasses, (1, 1), padding="same", activation="softmax")(_)

    #model creation
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model



def conv2d_block(inputs,use_batch_norm=True,dropout=0.2,dropout_type="spatial",
                 filters=16,kernel_size=(3, 3),activation="relu",
                 kernel_initializer="he_normal",padding="same"):

    if dropout_type == "spatial":
        DO = tf.keras.layers.SpatialDropout2D
    elif dropout_type == "standard":
        DO = tf.keras.layers.Dropout
    else:
        raise ValueError(f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}")

    c = tf.keras.layers.Conv2D(filters, kernel_size,activation=activation,
        kernel_initializer=kernel_initializer, padding=padding,
        use_bias=not use_batch_norm)(inputs)

    if use_batch_norm:
        c = tf.keras.layers.BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)

    c = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation,
        kernel_initializer=kernel_initializer, padding=padding,
        use_bias=not use_batch_norm)(c)

    if use_batch_norm:
        c = tf.keras.layers.BatchNormalization()(c)
    return c


def upsample_conv(filters, kernel_size, strides, padding):
    return tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def upsample_simple(filters, kernel_size, strides, padding):
    return tf.keras.layers.UpSampling2D(strides)


def custom_unet(input_shape, kernel = (2,2), num_classes=1,activation="relu",use_batch_norm=True,
    upsample_mode="deconv",
    dropout=0.1, dropout_change_per_layer=0.0, dropout_type="spatial",
    use_dropout_on_upsampling=False,
    filters=16,
    num_layers=4,
    output_activation="sigmoid"):

    """
    Customisable UNet architecture (Ronneberger et al. 2015 https://arxiv.org/abs/1505.04597)

    input_shape: shape (x, y, num_channels)

    num_classes (int): 1 for binary segmentation

    activation (str): A keras.activations.Activation to use. ReLu by default.

    use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutions

    upsample_mode ( "deconv" or "simple"): transposed convolutions or simple upsampling in the decoder

    dropout (float , 0. and 1.): dropout after the first convolutional block. 0. = no dropout

    dropout_change_per_layer (float , 0. and 1.): Factor to add to the Dropout after each convolutional block

    dropout_type (one of "spatial" or "standard"): Spatial is recommended  by  https://arxiv.org/pdf/1411.4280.pdf

    use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network

    filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block

    num_layers (int): Number of total layers in the encoder not including the bottleneck layer

    output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation

    """

    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # Build U-Net model
    inputs = tf.keras.layers.Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )
        down_layers.append(x)
        x =  tf.keras.layers.MaxPooling2D(kernel)(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(
        inputs=x,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        activation=activation,
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, kernel, strides=(2, 2), padding="same")(x)
        x = tf.keras.layers.concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters,
            use_batch_norm=use_batch_norm, dropout=dropout,
            dropout_type=dropout_type, activation=activation)

    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

##========================================================================

def bn_conv_relu(input, filters, bachnorm_momentum, **conv2d_args):
    x = tf.keras.layers.BatchNormalization(momentum=bachnorm_momentum)(input)
    x = tf.keras.layers.Conv2D(filters, **conv2d_args)(x)
    return x

def bn_upconv_relu(input, filters, bachnorm_momentum, **conv2d_trans_args):
    x = tf.keras.layers.BatchNormalization(momentum=bachnorm_momentum)(input)
    x = tf.keras.layers.Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x


def sat_unet(input_shape, num_classes=1, output_activation='sigmoid', num_layers=4):

    inputs = tf.keras.layers.Input(input_shape)

    filters = 16 #64
    upconv_filters = 24 #96

    kernel_size = (3,3)
    activation = 'relu'
    strides = (1,1)
    padding = 'same'
    kernel_initializer = 'he_normal'

    conv2d_args = {
        'kernel_size':kernel_size,
        'activation':activation,
        'strides':strides,
        'padding':padding,
        'kernel_initializer':kernel_initializer
        }

    conv2d_trans_args = {
        'kernel_size':kernel_size,
        'activation':activation,
        'strides':(2,2),
        'padding':padding,
        'output_padding':(1,1)
        }

    bachnorm_momentum = 0.01

    pool_size = (2,2)
    pool_strides = (2,2)
    pool_padding = 'valid'

    maxpool2d_args = {
        'pool_size':pool_size,
        'strides':pool_strides,
        'padding':pool_padding,
        }

    x = tf.keras.layers.Conv2D(filters, **conv2d_args)(inputs)
    c1 = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(c1, filters, bachnorm_momentum, **conv2d_args)
    x = tf.keras.layers.MaxPooling2D(**maxpool2d_args)(x)

    down_layers = []

    for l in range(num_layers):
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        down_layers.append(x)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = tf.keras.layers.MaxPooling2D(**maxpool2d_args)(x)

    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    for conv in reversed(down_layers):
        x = tf.keras.layers.concatenate([x, conv])
        x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    x = tf.keras.layers.concatenate([x, c1])
    x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)

    outputs = tf.keras.layers.Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), activation=output_activation, padding='valid') (x)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model
## https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/



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

#---------------------------------------------------
def dice_coef_loss(y_true, y_pred):
    """
    dice_coef_loss(y_true, y_pred)

    This function computes the mean Dice loss (1 - Dice coefficient) between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

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
        * Dice loss [tensor]
    """
    return 1.0 - dice_coef(y_true, y_pred)




###############################################################
### DATA FUNCTIONS
###############################################################


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
    if 'jpg' in f:
        image = tf.image.decode_jpeg(bits)
    elif 'png' in f:
        image = tf.image.decode_png(bits)

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




#-----------------------------------
#-----------------------------------
# def recompress_seg_image(image, label):
#     """
#     "recompress_seg_image"
#     This function takes an image and label encoded as a byte string
#     and recodes as an 8-bit jpeg
#     INPUTS:
#         * image [tensor array]
#         * label [tensor array]
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS:
#         * image [tensor array]
#         * label [tensor array]
#     """
#     image = tf.cast(image, tf.uint8)
#     image = tf.image.encode_png(image, compression=0) #jpeg(image, optimize_size=False, chroma_downsampling=False, quality=100, x_density=1000, y_density=1000)
#
#     label = tf.cast(label, tf.uint8)
#     label = tf.image.encode_png(label, compression=0) ##, optimize_size=False, chroma_downsampling=False, quality=100, x_density=400, y_density=400)
#
#     return image, label


# #-----------------------------------
# def write_seg_records(dataset, tfrecord_dir, root_string):
#     """
#     "write_seg_records(dataset, tfrecord_dir)"
#     This function writes a tf.data.Dataset object to TFRecord shards
#     INPUTS:
#         * dataset [tf.data.Dataset]
#         * tfrecord_dir [string] : path to directory where files will be written
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS: None (files written to disk)
#     """
#     for shard, (image, label) in enumerate(dataset):
#       shard_size = image.numpy().shape[0]
#       filename = tfrecord_dir+os.sep+root_string + "{:02d}-{}.tfrec".format(shard, shard_size)
#
#       with tf.io.TFRecordWriter(filename) as out_file:
#         for i in range(shard_size):
#           example = to_seg_tfrecord(image.numpy()[i].flatten().tobytes(),label.numpy()[i].flatten().tobytes())
#           out_file.write(example.SerializeToString())
#         print("Wrote file {} containing {} records".format(filename, shard_size))

# #-----------------------------------
# def write_seg_records_4bands(dataset, tfrecord_dir, root_string):
#     """
#     "write_seg_records(dataset, tfrecord_dir)"
#     This function writes a tf.data.Dataset object to TFRecord shards
#     INPUTS:
#         * dataset [tf.data.Dataset]
#         * tfrecord_dir [string] : path to directory where files will be written
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS: None (files written to disk)
#     """
#     for shard, (image, nir, label) in enumerate(dataset):
#       shard_size = image.numpy().shape[0]
#       filename = tfrecord_dir+os.sep+root_string + "{:02d}-{}.tfrec".format(shard, shard_size)
#
#       with tf.io.TFRecordWriter(filename) as out_file:
#         for i in range(shard_size):
#           example = to_seg_tfrecord_4bands(image.numpy()[i].flatten().tobytes(),nir.numpy()[i].flatten().tobytes(),label.numpy()[i].flatten().tobytes())
#           out_file.write(example.SerializeToString())
#         print("Wrote file {} containing {} records".format(filename, shard_size))
#
# #-----------------------------------
# def _bytestring_feature(list_of_bytestrings):
#     """
#     "_bytestring_feature"
#     cast inputs into tf dataset 'feature' classes
#     INPUTS:
#         * list_of_bytestrings
#     OPTIONAL INPUTS:
#     GLOBAL INPUTS:
#     OUTPUTS: tf.train.Feature example
#     """
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))
#
# #-----------------------------------
# def to_seg_tfrecord(img_bytes, label_bytes):
#     """
#     "to_seg_tfrecord"
#     This function creates a TFRecord example from an image byte string and a label feature
#     INPUTS:
#         * img_bytes
#         * label_bytes
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS: tf.train.Feature example
#     """
#     feature = {
#       "image": _bytestring_feature([img_bytes]), # one image in the list
#       "label": _bytestring_feature([label_bytes]), # one label image in the list
#               }
#     return tf.train.Example(features=tf.train.Features(feature=feature))
#
#
# #-----------------------------------
# def to_seg_tfrecord_4bands(img_bytes, nir_bytes, label_bytes):
#     """
#     "to_seg_tfrecord"
#     This function creates a TFRecord example from an image byte string and a label feature
#     INPUTS:
#         * img_bytes
#         * label_bytes
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS: tf.train.Feature example
#     """
#     feature = {
#       "image": _bytestring_feature([img_bytes]), # one image in the list
#       "nir": _bytestring_feature([nir_bytes]), # one nir image in the list
#       "label": _bytestring_feature([label_bytes]), # one label image in the list
#               }
#     return tf.train.Example(features=tf.train.Features(feature=feature))
