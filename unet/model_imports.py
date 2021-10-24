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

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

#keras functions for early stopping and model weights saving
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())


###############################################################
### MODEL ARCHITECTURES
###############################################################

#-----------------------------------
def custom_resunet(input_shape,
    kernel = (2,2),
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",
    dropout=0.1,
    dropout_change_per_layer=0.0,
    dropout_type="standard",
    use_dropout_on_upsampling=False,
    filters=8,
    num_layers=4,
    strides=(1,1)):

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
        x = res_conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
            strides=strides,#(1,1),
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
        strides=strides,#(1,1),
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, kernel, strides=(2,2), padding="same")(x)#(2, 2)
        x = tf.keras.layers.concatenate([x, conv])
        x = res_conv2d_block(inputs=x, filters=filters,
            use_batch_norm=use_batch_norm, dropout=dropout,
            dropout_type=dropout_type, activation=activation,strides=strides)#(1,1))
    #outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    # ## classify
    if num_classes==1:
        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid")(x) #
    else:
        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), padding="same", activation="softmax")(x) #(1, 1)


    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model


#-----------------------------------
def custom_unet(input_shape,
    kernel = (2,2),
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",
    dropout=0.1,
    dropout_change_per_layer=0.0,
    dropout_type="standard",
    use_dropout_on_upsampling=False,
    filters=8,
    num_layers=4,
    strides=(1,1)):

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
            strides=strides,#(1,1),
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
        strides=strides,#(1,1),
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, kernel, strides=(2,2), padding="same")(x)#(2, 2)
        x = tf.keras.layers.concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters,
            use_batch_norm=use_batch_norm, dropout=dropout,
            dropout_type=dropout_type, activation=activation)

    #outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    # ## classify
    if num_classes==1:
        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid")(x)
    else:
        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), padding="same", activation="softmax")(x)


    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

##========================================================================


#-----------------------------------
def custom_satunet(input_shape,
    kernel = (2,2),
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",
    dropout=0.1,
    dropout_change_per_layer=0.0,
    dropout_type="standard",
    use_dropout_on_upsampling=False,
    filters=8,
    num_layers=4,
    strides=(1,1)):

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

    """

    upconv_filters = int(1.5*filters)

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
            strides=strides,#(1,1),
        )
        down_layers.append(x)
        x =  tf.keras.layers.MaxPooling2D(kernel)(x)
        dropout += dropout_change_per_layer
        #filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(
        inputs=x,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        activation=activation,
        strides=strides,#(1,1),
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, kernel, strides=(2,2), padding="same")(x)#(2, 2)
        x = tf.keras.layers.concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=upconv_filters,
            use_batch_norm=use_batch_norm, dropout=dropout,
            dropout_type=dropout_type, activation=activation)

    #outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    # ## classify
    if num_classes==1:
        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid")(x)
    else:
        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), padding="same", activation="softmax")(x)


    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

##========================================================================


#-----------------------------------
def sat_unet(input_shape, num_classes=1, num_layers=4):

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

    # outputs = tf.keras.layers.Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), activation=output_activation, padding='valid') (x)

    # ## classify
    if num_classes==1:
        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1),  strides=(1,1), padding="valid", activation="sigmoid")(x)
    else:
        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1),  strides=(1,1), padding="valid", activation="softmax")(x)


    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model
## https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/



###############################################################
### MODEL SUBFUNCTIONS
###############################################################
#-----------------------------------
def bn_conv_relu(input, filters, bachnorm_momentum, **conv2d_args):
    x = tf.keras.layers.BatchNormalization(momentum=bachnorm_momentum)(input)
    x = tf.keras.layers.Conv2D(filters, **conv2d_args)(x)
    return x

#-----------------------------------
def bn_upconv_relu(input, filters, bachnorm_momentum, **conv2d_trans_args):
    x = tf.keras.layers.BatchNormalization(momentum=bachnorm_momentum)(input)
    x = tf.keras.layers.Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x


#-----------------------------------
def conv2d_block(inputs,use_batch_norm=True,dropout=0.1,dropout_type="standard",
                 filters=16,kernel_size=(2, 2),activation="relu", strides=(1,1),
                 kernel_initializer="he_normal",padding="same"):

    if dropout_type == "spatial":
        DO = tf.keras.layers.SpatialDropout2D
    elif dropout_type == "standard":
        DO = tf.keras.layers.Dropout
    else:
        raise ValueError(f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}")

    c = tf.keras.layers.Conv2D(filters, kernel_size,activation=activation,
        kernel_initializer=kernel_initializer, padding=padding, strides=strides,
        use_bias=not use_batch_norm)(inputs)

    if use_batch_norm:
        c = tf.keras.layers.BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)

    c = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation,
        kernel_initializer=kernel_initializer, padding=padding, strides=strides,
        use_bias=not use_batch_norm)(c)

    if use_batch_norm:
        c = tf.keras.layers.BatchNormalization()(c)
    return c


#-----------------------------------
def res_conv2d_block(inputs,use_batch_norm=True,dropout=0.1,dropout_type="standard",
                 filters=16,kernel_size=(2, 2),activation="relu", strides=(1,1),
                 kernel_initializer="he_normal",padding="same"):

    res = conv2d_block(
        inputs=inputs,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        strides=strides,
        kernel_initializer="he_normal",padding="same",
    )

    res = conv2d_block(
        inputs=res,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        strides=(1,1),
        kernel_initializer="he_normal",padding="same",
    )

    bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides)(inputs) ##(1,1)
    bottleneck = batchnorm_act(bottleneck)

    return tf.keras.layers.Add()([bottleneck, res])

#-----------------------------------
def upsample_conv(filters, kernel_size, strides, padding):
    return tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

#-----------------------------------
def upsample_simple(filters, kernel_size, strides, padding):
    return tf.keras.layers.UpSampling2D(strides)

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



###############################################################
### LOSSES AND METRICS
###############################################################

#-----------------------------------
# def mean_iou_eval(y_true, y_pred):
#     """
#     mean_iou(y_true, y_pred)
#     This function computes the mean IoU between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions
#
#     INPUTS:
#         * y_true: true masks, one-hot encoded.
#             * Inputs are B*W*H*N tensors, with
#                 B = batch size,
#                 W = width,
#                 H = height,
#                 N = number of classes
#         * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
#             * Inputs are B*W*H*N tensors, with
#                 B = batch size,
#                 W = width,
#                 H = height,
#                 N = number of classes
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS:
#         * IoU score [tensor]
#     """
#     yt0 = tf.keras.backend.cast(y_true, 'float32')#[:,:,:,0]
#     yp0 = tf.keras.backend.cast(y_pred.numpy()> 0.5, 'float32')
#     inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
#     union = tf.math.count_nonzero(tf.add(yt0, yp0))
#     iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
#     return iou


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


#-----------------------------------
def iou(obs, est,nclasses):
    IOU=0
    smooth = 1.
    if nclasses>1:
        for n in range(nclasses):
            component1 = obs==n
            component2 = est==n
            overlap = component1*component2 # Logical AND
            union = component1 + component2 # Logical OR
            calc = overlap.sum()/(float(union.sum())+smooth)
            if not np.isnan(calc):
                IOU += calc
    else:
        if len(np.unique(obs))>1:
            for n in range(nclasses):
                component1 = obs==n
                component2 = est==n
                overlap = component1*component2 # Logical AND
                union = component1 + component2 # Logical OR
                calc = overlap.sum()/(float(union.sum())+smooth)
                if not np.isnan(calc):
                    IOU += calc
        else:
            u=np.unique(obs)[0]
            IOU = np.sum(est==u)/np.sum(obs==u)

    if IOU>1:
        IOU=IOU/n

    return IOU



#
# #-----------------------------------
# def upsamp_concat_block(x, xskip):
#     """
#     upsamp_concat_block(x, xskip)
#     This function takes an input layer and creates a concatenation of an upsampled version and a residual or 'skip' connection
#     INPUTS:
#         * `xskip`: input keras layer (skip connection)
#         * `x`: input keras layer
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS:
#         * keras layer, output of the addition between residual convolutional and bottleneck layers
#     """
#     u = tf.keras.layers.UpSampling2D((2, 2))(x)
#     return tf.keras.layers.Concatenate()([u, xskip])
#

#
# #-----------------------------------
# def res_unet(sz, f, nclasses=1, kernel_size=(7,7)):
#     """
#     res_unet(sz, f, nclasses=1)
#     This function creates a custom residual U-Net model for image segmentation
#     INPUTS:
#         * `sz`: [tuple] size of input image
#         * `f`: [int] number of filters in the convolutional block
#         * flag: [string] if 'binary', the model will expect 2D masks and uses sigmoid. If 'multiclass', the model will expect 3D masks and uses softmax
#         * nclasses [int]: number of classes
#     OPTIONAL INPUTS:
#         * `kernel_size`=(7, 7): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
#         * `padding`="same":  see tf.keras.layers.Conv2D
#         * `strides`=1: see tf.keras.layers.Conv2D
#     GLOBAL INPUTS: None
#     OUTPUTS:
#         * keras model
#     """
#     inputs = tf.keras.layers.Input(sz)
#
#     ## downsample
#     e1 = bottleneck_block(inputs, f);
#     f = int(f*2)
#     e2 = res_block(e1, f, strides=2,  kernel_size = kernel_size)
#     f = int(f*2)
#     e3 = res_block(e2, f, strides=2,  kernel_size = kernel_size)
#     f = int(f*2)
#     e4 = res_block(e3, f, strides=2,  kernel_size = kernel_size)
#     f = int(f*2)
#     _ = res_block(e4, f, strides=2, kernel_size = kernel_size)
#
#     ## bottleneck
#     b0 = conv_block(_, f, strides=1,  kernel_size = kernel_size)
#     _ = conv_block(b0, f, strides=1,  kernel_size = kernel_size)
#
#     ## upsample
#     _ = upsamp_concat_block(_, e4)
#     _ = res_block(_, f, kernel_size = kernel_size)
#     f = int(f/2)
#
#     _ = upsamp_concat_block(_, e3)
#     _ = res_block(_, f, kernel_size = kernel_size)
#     f = int(f/2)
#
#     _ = upsamp_concat_block(_, e2)
#     _ = res_block(_, f, kernel_size = kernel_size)
#     f = int(f/2)
#
#     _ = upsamp_concat_block(_, e1)
#     _ = res_block(_, f, kernel_size = kernel_size)
#
#     # ## classify
#     if nclasses==1:
#         outputs = tf.keras.layers.Conv2D(nclasses, (1, 1), padding="same", activation="sigmoid")(_)
#     else:
#         outputs = tf.keras.layers.Conv2D(nclasses, (1, 1), padding="same", activation="softmax")(_)
#
#     #model creation
#     model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
#     return model


#
# #-----------------------------------
# def conv_block(x, filters, kernel_size = (7,7), padding="same", strides=1):
#     """
#     conv_block(x, filters, kernel_size = (7,7), padding="same", strides=1)
#     This function applies batch normalization to an input layer, then convolves with a 2D convol layer
#     The two actions combined is called a convolutional block
#
#     INPUTS:
#         * `filters`: number of filters in the convolutional block
#         * `x`:input keras layer to be convolved by the block
#     OPTIONAL INPUTS:
#         * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
#         * `padding`="same":  see tf.keras.layers.Conv2D
#         * `strides`=1: see tf.keras.layers.Conv2D
#     GLOBAL INPUTS: None
#     OUTPUTS:
#         * keras layer, output of the batch normalized convolution
#     """
#     conv = batchnorm_act(x)
#     return tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
#
# #-----------------------------------
# def bottleneck_block(x, filters, kernel_size = (2,2), padding="same", strides=1):
#     """
#     bottleneck_block(x, filters, kernel_size = (7,7), padding="same", strides=1)
#
#     This function creates a bottleneck block layer, which is the addition of a convolution block and a batch normalized/activated block
#     INPUTS:
#         * `filters`: number of filters in the convolutional block
#         * `x`: input keras layer
#     OPTIONAL INPUTS:
#         * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
#         * `padding`="same":  see tf.keras.layers.Conv2D
#         * `strides`=1: see tf.keras.layers.Conv2D
#     GLOBAL INPUTS: None
#     OUTPUTS:
#         * keras layer, output of the addition between convolutional and bottleneck layers
#     """
#     conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
#     conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
#
#     bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
#     bottleneck = batchnorm_act(bottleneck)
#
#     return tf.keras.layers.Add()([conv, bottleneck])


# def res_block_orig(x, filters, kernel_size = (7,7), padding="same", strides=1):
#     """
#     res_block(x, filters, kernel_size = (7,7), padding="same", strides=1)
#     This function creates a residual block layer, which is the addition of a residual convolution block and a batch normalized/activated block
#     INPUTS:
#         * `filters`: number of filters in the convolutional block
#         * `x`: input keras layer
#     OPTIONAL INPUTS:
#         * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
#         * `padding`="same":  see tf.keras.layers.Conv2D
#         * `strides`=1: see tf.keras.layers.Conv2D
#     GLOBAL INPUTS: None
#     OUTPUTS:
#         * keras layer, output of the addition between residual convolutional and bottleneck layers
#     """
#     res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
#     res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
#
#     bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
#     bottleneck = batchnorm_act(bottleneck)
#
#     return tf.keras.layers.Add()([bottleneck, res])
#
# #-----------------------------------
# def res_block(x, filters,
#               kernel_size = (2,2),
#               activation='relu',
#               padding="same",
#               strides=(1,1),
#               use_batch_norm=True,
#               dropout=0.1,
#               dropout_change_per_layer=0.0,
#               dropout_type="standard",
#               use_dropout_on_upsampling=False):
#     """
#     res_block(x, filters, kernel_size = (7,7), padding="same", strides=1)
#
#     This function creates a residual block layer, which is the addition of a residual convolution block and a batch normalized/activated block
#     INPUTS:
#         * `filters`: number of filters in the convolutional block
#         * `x`: input keras layer
#     OPTIONAL INPUTS:
#         * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
#         * `padding`="same":  see tf.keras.layers.Conv2D
#         * `strides`=1: see tf.keras.layers.Conv2D
#     GLOBAL INPUTS: None
#     OUTPUTS:
#         * keras layer, output of the addition between residual convolutional and bottleneck layers
#     """
#     # res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
#     # res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
#
#     res = conv2d_block(
#         inputs=x,
#         use_batch_norm=use_batch_norm,
#         dropout=dropout,
#         dropout_type=dropout_type,
#         filters=filters,
#         kernel_size=kernel_size,
#         activation=activation,
#         strides=strides,
#         kernel_initializer="he_normal",padding="same",
#     )
#
#     res = conv2d_block(
#         inputs=res,
#         use_batch_norm=use_batch_norm,
#         dropout=dropout,
#         dropout_type=dropout_type,
#         filters=filters,
#         kernel_size=kernel_size,
#         activation=activation,
#         strides=(1,1),
#         kernel_initializer="he_normal",padding="same",
#     )
#
#
#     bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
#     bottleneck = batchnorm_act(bottleneck)
#
#     return tf.keras.layers.Add()([bottleneck, res])

# #-----------------------------------

# #
# #-----------------------------------
# def custom_resunet(input_shape,
#     kernel = (2,2),
#     num_classes=1,
#     activation="relu",
#     use_batch_norm=True,
#     upsample_mode="deconv",
#     dropout=0.1,
#     dropout_change_per_layer=0.0,
#     dropout_type="standard",
#     use_dropout_on_upsampling=False,
#     filters=8,
#     num_layers=4,
#     strides=(1,1)):
#
#     inputs = tf.keras.layers.Input(input_shape)#sz)
#
#     ## downsample
#     e1 = bottleneck_block(inputs, filters)
#     filters = int(filters*2)
#     #e2 = res_block_orig(e1, filters, strides=2,  kernel_size = kernel_size)
#     e2 = res_block(e1, filters, kernel_size = kernel,activation=activation,
#                     padding="same", strides=strides, use_batch_norm=use_batch_norm, dropout=dropout,
#                     dropout_change_per_layer=dropout_change_per_layer,
#                     dropout_type=dropout_type, use_dropout_on_upsampling=use_dropout_on_upsampling)
#     #e2.shape
#     filters = int(filters*2)
#     #e3 = res_block_orig(e2, filters, strides=2,  kernel_size = kernel_size)
#     e3 = res_block(e2, filters, kernel_size = kernel,activation=activation,
#                     padding="same", strides=strides, use_batch_norm=use_batch_norm, dropout=dropout,
#                     dropout_change_per_layer=dropout_change_per_layer,
#                     dropout_type=dropout_type, use_dropout_on_upsampling=use_dropout_on_upsampling)
#     #e3.shape
#     filters = int(filters*2)
#     #e4 = res_block_orif(e3, filters, strides=2,  kernel_size = kernel_size)
#     e4 = res_block(e3, filters, kernel_size = kernel,activation=activation,
#                     padding="same", strides=strides, use_batch_norm=use_batch_norm, dropout=dropout,
#                     dropout_change_per_layer=dropout_change_per_layer,
#                     dropout_type=dropout_type, use_dropout_on_upsampling=use_dropout_on_upsampling)
#     #e4.shape
#     filters = int(filters*2)
#     #_ = res_block(e4, filters, strides=2, kernel_size = kernel_size)
#     _ = res_block(e4, filters, kernel_size = kernel,activation=activation,
#                     padding="same", strides=strides, use_batch_norm=use_batch_norm, dropout=dropout,
#                     dropout_change_per_layer=dropout_change_per_layer,
#                     dropout_type=dropout_type, use_dropout_on_upsampling=use_dropout_on_upsampling)
#     #_.shape
#     ## bottleneck
#     #b0 = conv_block(_, filters, strides=1,  kernel_size = kernel_size)
#     #_ = conv_block(b0, filters, strides=1,  kernel_size = kernel_size)
#
#     b0 = conv2d_block(
#         inputs=_,
#         use_batch_norm=use_batch_norm,
#         dropout=dropout,
#         dropout_type=dropout_type,
#         filters=filters,
#         kernel_size=kernel,
#         activation=activation,
#         strides=(1,1),
#         kernel_initializer="he_normal",padding="same",
#     )
#     #b0.shape
#
#     _ = conv2d_block(
#         inputs=b0,
#         use_batch_norm=use_batch_norm,
#         dropout=dropout,
#         dropout_type=dropout_type,
#         filters=filters,
#         kernel_size=kernel,
#         activation=activation,
#         strides=(1,1),
#         kernel_initializer="he_normal",padding="same",
#     )
#     #_.shape
#
#     ## upsample
#     _ = upsamp_concat_block(_, e4, strides=strides)
#     #_ = res_block_orig(_, filters, kernel_size = kernel_size)
#     _ = res_block(_, filters, kernel_size = kernel,activation=activation,
#                     padding="same", strides=strides, use_batch_norm=use_batch_norm, dropout=dropout,
#                     dropout_change_per_layer=dropout_change_per_layer,
#                     dropout_type=dropout_type, use_dropout_on_upsampling=use_dropout_on_upsampling)
#     filters = int(filters/2)
#
#     _ = upsamp_concat_block(_, e3, strides=strides)
#     #_ = res_block_orig(_, filters, kernel_size = kernel_size)
#     _ = res_block(_, filters, kernel_size = kernel,activation=activation,
#                     padding="same", strides=strides, use_batch_norm=use_batch_norm, dropout=dropout,
#                     dropout_change_per_layer=dropout_change_per_layer,
#                     dropout_type=dropout_type, use_dropout_on_upsampling=use_dropout_on_upsampling)
#     filters = int(filters/2)
#
#     _ = upsamp_concat_block(_, e2, strides=strides)
#     #_ = res_block_orig(_, filters, kernel_size = kernel_size)
#     _ = res_block(_, filters, kernel_size = kernel,activation=activation,
#                     padding="same", strides=strides, use_batch_norm=use_batch_norm, dropout=dropout,
#                     dropout_change_per_layer=dropout_change_per_layer,
#                     dropout_type=dropout_type, use_dropout_on_upsampling=use_dropout_on_upsampling)
#     filters = int(filters/2)
#
#     _ = upsamp_concat_block(_, e1, strides=strides)
#     #_ = res_block_orig(_, f, kernel_size = kernel_size)
#
#     # ## classify
#     if num_classes==1:
#         outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid")(_)
#     else:
#         outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), padding="same", activation="softmax")(_)
#
#     #model creation
#     model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
#     return model
