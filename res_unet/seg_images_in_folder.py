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

import os, time

USE_GPU = True
DO_CRF_REFINE = True

if USE_GPU == True:
   ##use the first available GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#utils
#keras functions for early stopping and model weights saving
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf #numerical operations on gpu
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import as_strided as ast
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.restoration import inpaint
from scipy.ndimage import maximum_filter
from skimage.transform import resize
from tqdm import tqdm
from skimage.filters import threshold_otsu

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
def iou(obs, est, nclasses):
    IOU=0
    for n in range(1,nclasses+1):
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
def seg_file2tensor_3band(f, resize):
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


#-----------------------------------
def seg_file2tensor_4band(f, fir, resize):
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

    bigimage = tf.concat([bigimage, nir],-1)[:,:,:4]
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


#====================================================

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/model_training/weights",title = "Select file",filetypes = (("weights file","*.h5"),("all files","*.*")))
weights = root.filename
print(weights)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/samples",title = "Select directory of images to segment")
sample_direc = root.filename
print(sample_direc)
root.withdraw()


configfile = weights.replace('.h5','.json').replace('weights', 'config')

with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')


#=======================================================
model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), BATCH_SIZE, NCLASSES)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])

model.load_weights(weights)

# class_label_colormap = ['#0b19d9','#ffffff','#8f6727','#6b2241']


### predict
print('.....................................')
print('Using model for prediction on images ...')

sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))
if len(sample_filenames)==0:
    sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.png'))

print('Number of samples: %i' % (len(sample_filenames)))

for counter,f in enumerate(sample_filenames):

    start = time.time()
    
    if N_DATA_BANDS<=3:
        image, w, h, bigimage = seg_file2tensor_3band(f, resize=True)
        image = image/255
        bigimage = bigimage/255
        w = w.numpy(); h = h.numpy()
    else:
        image, w, h, bigimage = seg_file2tensor_4band(f, f.replace('aug_images', 'aug_nir'), resize=True )
        image = image/255
        bigimage = bigimage/255
        w = w.numpy(); h = h.numpy()
        
    print("Working on %i x %i image" % (w,h))

    if NCLASSES==1:
        E = []; W = []
        E.append(model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze())
        W.append(1)
        E.append(np.fliplr(model.predict(tf.expand_dims(np.fliplr(image), 0) , batch_size=1).squeeze()))
        W.append(.75)        
        E.append(np.flipud(model.predict(tf.expand_dims(np.flipud(image), 0) , batch_size=1).squeeze()))
        W.append(.75)

        for k in np.linspace(100,int(TARGET_SIZE[0]/5),10):
            #E.append(np.roll(model.predict(tf.expand_dims(np.roll(image, int(k)), 0) , batch_size=1).squeeze(), -int(k)))
            E.append(model.predict(tf.expand_dims(np.roll(image, int(k)), 0) , batch_size=1).squeeze())
            W.append(2*(1/np.sqrt(k)))

        for k in np.linspace(100,int(TARGET_SIZE[0]/5),10):
            #E.append(np.roll(model.predict(tf.expand_dims(np.roll(image, -int(k)), 0) , batch_size=1).squeeze(), int(k)))
            E.append(model.predict(tf.expand_dims(np.roll(image, -int(k)), 0) , batch_size=1).squeeze())
            W.append(2*(1/np.sqrt(k)))

        K.clear_session()
        
        # for c,e in enumerate(E):
            # plt.imshow(image); plt.imshow(e, alpha=0.4, cmap='gray')
            # plt.axis('off'); plt.title('W = '+str(W[c])[:5])
            # plt.savefig(str(c)+'png', dpi=200)
            # plt.close()
        
        E = [maximum_filter(resize(e,(w,h)), int(w/200)) for e in E]

        # for c,e in enumerate(E):
            # plt.imshow(bigimage); plt.imshow(e, alpha=0.4, cmap='gray')
            # plt.axis('off'); plt.savefig('f'+str(c)+'png', dpi=200)
            # plt.close()

        #est_label = np.median(np.dstack(E), axis=-1)
        est_label = np.average(np.dstack(E), axis=-1, weights=np.array(W))
        
        # plt.imshow(bigimage); plt.imshow(est_label, alpha=0.4, cmap='bwr'); plt.colorbar(); 
        # plt.axis('off'); plt.savefig('im-mask.png', dpi=200); plt.close()
        
        var = np.std(np.dstack(E), axis=-1)

        # plt.imshow(bigimage); plt.imshow(var, alpha=0.4, cmap='bwr'); plt.colorbar(); 
        # plt.axis('off'); plt.savefig('im-maskvar.png', dpi=200); plt.close()

        if np.max(est_label)-np.min(est_label) > .5:
            thres = threshold_otsu(est_label)
            print("Threshold: %f" % (thres))
        else:
            thres = .75
            print("Default threshold: %f" % (thres))

        if NCLASSES==1:
            conf = 1-est_label
            conf[est_label<thres] = est_label[est_label<thres]
            conf = 1-conf
        else:
            conf = np.max(est_label, -1)

        conf[np.isnan(conf)] = 0
        conf[np.isinf(conf)] = 0

        model_conf = np.sum(conf)/np.prod(conf.shape)
        print('Overall model confidence = %f'%(model_conf))

        # plt.imshow(bigimage); plt.imshow(conf, alpha=0.4, cmap='bwr'); plt.colorbar(); 
        # plt.axis('off'); plt.savefig('im-conf.png', dpi=200); plt.close()

    else:
        est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()

        K.clear_session()

        est_label = resize(est_label,(w,h))
        est_label = np.argmax(est_label,-1)

    est_label = np.squeeze(est_label[:w,:h])

    if NCLASSES==1:
        est_label[est_label<thres] = 0
        est_label[est_label>thres] = 1
        est_label = remove_small_holes(est_label.astype('uint8')*2, 2*w)
        est_label = remove_small_objects(est_label.astype('uint8')*2, 2*w)
        est_label[est_label<thres] = 0
        est_label[est_label>thres] = 1

    if NCLASSES>1:
        class_label_colormap = ['#00FFFF','#0000FF','#808080','#008000','#FFA500'][:NCLASSES]
        color_label = label_to_colors(est_label, bigimage.numpy()[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
        
    # plt.imshow(bigimage); plt.imshow(est_label, alpha=0.4, cmap='bwr'); plt.colorbar(); 
    # plt.axis('off'); plt.savefig('im-maskthreshold.png', dpi=200); plt.close()
    
    #print(est_label.shape)
    
    elapsed = (time.time() - start)/60
    print("Image masking took "+ str(elapsed) + " minutes")
    start = time.time()

    if NCLASSES==1:
        if 'jpg' in f:
            imsave(f.replace('.jpg', '_predseg.png'), (est_label*255).astype(np.uint8), check_contrast=False)
            np.savez(f.replace('.jpg', '_conf.npz'), conf)
            np.savez(f.replace('.jpg', '_var.npz'), var)
            
            # imsave(f.replace('.jpg', '_predseg_col.png'), (color_label).astype(np.uint8), check_contrast=False)
            cmd = 'convert '+f+' \( '+f.replace('.jpg', '_predseg.png')+' -normalize +level 0,50% \) -compose screen -composite '+f.replace('.jpg', '_segoverlay.png')
            if os.name=='posix':
                os.system(cmd)
            else:
                imsave(f.replace('.jpg', '_segoverlay.png'), np.dstack((255*bigimage.numpy(), (est_label*255))), check_contrast=False)
        elif 'png' in f:
            imsave(f.replace('.png', '_predseg.png'), (est_label*255).astype(np.uint8), check_contrast=False)
            np.savez(f.replace('.png', '_conf.npz'), conf)
            np.savez(f.replace('.png', '_var.npz'), var)
            
            # imsave(f.replace('.png', '_predseg_col.png'), (color_label).astype(np.uint8), check_contrast=False)
            cmd = 'convert '+f+' \( '+f.replace('.png', '_predseg.png')+' -normalize +level 0,50% \) -compose screen -composite '+f.replace('.png', '_segoverlay.png')
            if os.name=='posix':
                os.system(cmd)
            else:
                imsave(f.replace('.png', '_segoverlay.png'), np.dstack((255*bigimage.numpy(), (est_label*255))), check_contrast=False)
    else:
        if 'jpg' in f:
            imsave(f.replace('.jpg', '_predseg.png'), (est_label).astype(np.uint8), check_contrast=False)
            imsave(f.replace('.jpg', '_predseg_col.png'), (color_label).astype(np.uint8), check_contrast=False)
        elif 'png' in f:
            imsave(f.replace('.png', '_predseg.png'), (est_label).astype(np.uint8), check_contrast=False)
            imsave(f.replace('.png', '_predseg_col.png'), (color_label).astype(np.uint8), check_contrast=False)

    elapsed = (time.time() - start)/60
    print("File writing took "+ str(elapsed) + " minutes")
    print("%s done" % (f))




        # if N_DATA_BANDS<=3:
        #     image = seg_file2tensor_3band(f, resize=False)/255
        # else:
        #     image = seg_file2tensor_4band(f, f.replace('aug_images', 'aug_nir'), resize=False )/255

        #width = image.shape[0]
        #height = image.shape[1]
        # E = []
        # E.append(model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze())
        # E.append(np.fliplr(model.predict(tf.expand_dims(np.fliplr(image), 0) , batch_size=1).squeeze()))
        # E.append(np.flipud(model.predict(tf.expand_dims(np.flipud(image), 0) , batch_size=1).squeeze()))
        #
        # for k in np.linspace(100,TARGET_SIZE[0],10):
        #     E.append(np.roll(model.predict(tf.expand_dims(np.roll(image, int(k)), 0) , batch_size=1).squeeze(), -int(k)))
        #
        # for k in np.linspace(100,TARGET_SIZE[0],10):
        #     E.append(np.roll(model.predict(tf.expand_dims(np.roll(image, -int(k)), 0) , batch_size=1).squeeze(), int(k)))
        #
        # E = [maximum_filter(resize(e,(width,height)), int(width/100)) for e in E]
        #
        
        
    # est_label += 1
    # est_label[conf<np.mean(conf)/3] = 0

    # conf = conf[:width,:height]

    # try:
    #     est_label2 = inpaint.inpaint_biharmonic(resize(est_label, (int(width/4), (height/4))), resize(est_label, (int(width/4), (height/4)))==0, multichannel=False)
    #     est_label = resize(est_label2, (width, height))-1
    #     est_label[est_label<0]=0
    # except:
    #     pass


    # padwidth = width + (TARGET_SIZE[0] - width % TARGET_SIZE[0])
    # padheight = height + (TARGET_SIZE[1] - height % TARGET_SIZE[1])
    # I = np.zeros((padwidth, padheight, N_DATA_BANDS))
    # I[:width,:height,:] = image.numpy()
    #
    # gridx, gridy = np.meshgrid(np.arange(padheight), np.arange(padwidth))
    #
    # E = []
    # for n in tqdm([2,4,6,8]):
    #     Zx,_ = sliding_window(gridx, (TARGET_SIZE[0],TARGET_SIZE[1]), (int(TARGET_SIZE[0]/n), int(TARGET_SIZE[1]/n)))
    #     Zy,_ = sliding_window(gridy, (TARGET_SIZE[0],TARGET_SIZE[1]), (int(TARGET_SIZE[0]/n), int(TARGET_SIZE[1]/n)))
    #     #print(len(Zx))
    #
    #     Z,ind = sliding_window(I, (TARGET_SIZE[0],TARGET_SIZE[1],N_DATA_BANDS), (int(TARGET_SIZE[0]/n), int(TARGET_SIZE[1]/n), N_DATA_BANDS))
    #     #del I
    #     est_label = np.zeros((padwidth, padheight))
    #     N = np.zeros((padwidth, padheight))
    #
    #     for z,x,y in zip(Z,Zx,Zy):
    #         est_label[y,x] = model.predict(tf.expand_dims(z, 0) , batch_size=1).squeeze()
    #         N[y,x] += 1
    #     del Z, Zx, Zy
    #
    #     #est_label = median(est_label, disk(int(width/100)))/255.
    #     est_label = maximum_filter(est_label, int(width/100))
    #
    #     est_label /= N
    #     del N
    #     E.append(est_label)
    #
    # #est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
    # K.clear_session()
    #
    # #est_label = np.median(np.dstack(E), axis=-1)
    # est_label =  np.max(np.dstack(E), axis=-1)
    # est_label[np.isnan(est_label)] = 0
    # est_label[np.isinf(est_label)] = 0
    #
    # est_label = maximum_filter(est_label, int(width/100))
    # est_label = median(est_label, disk(int(width/100)))/255.

#
# # =========================================================
# def norm_shape(shap):
#    '''
#    Normalize numpy array shapes so they're always expressed as a tuple,
#    even for one-dimensional shapes.
#    '''
#    try:
#       i = int(shap)
#       return (i,)
#    except TypeError:
#       # shape was not a number
#       pass
#
#    try:
#       t = tuple(shap)
#       return t
#    except TypeError:
#       # shape was not iterable
#       pass
#
#    raise TypeError('shape must be an int, or a tuple of ints')
#
#
# # =========================================================
# # Return a sliding window over a in any number of dimensions
# # version with no memory mapping
# def sliding_window(a,ws,ss = None,flatten = True):
#     '''
#     Return a sliding window over a in any number of dimensions
#     '''
#     if None is ss:
#         # ss was not provided. the windows will not overlap in any direction.
#         ss = ws
#     ws = norm_shape(ws)
#     ss = norm_shape(ss)
#     # convert ws, ss, and a.shape to numpy arrays
#     ws = np.array(ws)
#     ss = np.array(ss)
#     shap = np.array(a.shape)
#     # ensure that ws, ss, and a.shape all have the same number of dimensions
#     ls = [len(shap),len(ws),len(ss)]
#     if 1 != len(set(ls)):
#         raise ValueError(\
#         'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
#
#     # ensure that ws is smaller than a in every dimension
#     if np.any(ws > shap):
#         raise ValueError(\
#         'ws cannot be larger than a in any dimension.\
#  a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
#     # how many slices will there be in each dimension?
#     newshape = norm_shape(((shap - ws) // ss) + 1)
#     # the shape of the strided array will be the number of slices in each dimension
#     # plus the shape of the window (tuple addition)
#     newshape += norm_shape(ws)
#     # the strides tuple will be the array's strides multiplied by step size, plus
#     # the array's strides (tuple addition)
#     newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
#     a = ast(a,shape = newshape,strides = newstrides)
#     if not flatten:
#         return a
#     # Collapse strided so that it has one more dimension than the window.  I.e.,
#     # the new array is a flat list of slices.
#     meat = len(ws) if ws.shape else 0
#     firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
#     dim = firstdim + (newshape[-meat:])
#     # remove any dimensions with size 1
#     #dim = filter(lambda i : i != 1,dim)
#
#     return a.reshape(dim), newshape
