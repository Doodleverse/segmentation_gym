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

USE_GPU = True #False

if USE_GPU == True:
   ##use the first available GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from skimage.filters.rank import median
from skimage.morphology import disk
from tkinter import filedialog
from tkinter import *
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

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels


# #-----------------------------------
def seg_file2tensor_3band(f):#, resize):
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


# =========================================================
def do_seg(f, model):

	# model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), BATCH_SIZE, NCLASSES)

	# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])

	# model.load_weights(weights)

	if NCLASSES==1:
		if 'jpg' in f:
			segfile = f.replace('.jpg', '_seg.tif')
		elif 'png' in f:
			segfile = f.replace('.png', '_seg.tif')

		segfile = os.path.normpath(segfile)
		# segfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks'))

		if os.path.exists(segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'certainty'))):
			print('%s exists ... skipping' % (segfile))
			pass
		else:
			print('%s does not exist ... creating' % (segfile))

		start = time.time()

		if N_DATA_BANDS<=3:
			image, w, h, bigimage = seg_file2tensor_3band(f)#, resize=True)
			if image is None:
				image = bigimage#/255
				#bigimage = bigimage#/255
				w = w.numpy(); h = h.numpy()
		else:
			image, w, h, bigimage = seg_file2tensor_4band(f, f.replace('aug_images', 'aug_nir'), resize=True )
			if image is None:
				image = bigimage#/255
				w = w.numpy(); h = h.numpy()

		print("Working on %i x %i image" % (w,h))

		image = standardize(image.numpy()).squeeze()

		E = []; W = []
		est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
		est_label = np.argmax(est_label, -1)
		E.append(est_label)
		W.append(1)
		est_label = np.fliplr(model.predict(tf.expand_dims(np.fliplr(image), 0) , batch_size=1).squeeze())
		est_label = np.argmax(est_label, -1)
		E.append(est_label)
		W.append(.5)
		est_label = np.flipud(model.predict(tf.expand_dims(np.flipud(image), 0) , batch_size=1).squeeze())
		est_label = np.argmax(est_label, -1)
		E.append(est_label)
		W.append(.5)

		# for k in np.linspace(100,int(TARGET_SIZE[0]),10):
		#     #E.append(np.roll(model.predict(tf.expand_dims(np.roll(image, int(k)), 0) , batch_size=1).squeeze(), -int(k)))
		#     E.append(model.predict(tf.expand_dims(np.roll(image, int(k)), 0) , batch_size=1).squeeze())
		#     W.append(2*(1/np.sqrt(k)))
		#
		# for k in np.linspace(100,int(TARGET_SIZE[0]),10):
		#     #E.append(np.roll(model.predict(tf.expand_dims(np.roll(image, -int(k)), 0) , batch_size=1).squeeze(), int(k)))
		#     E.append(model.predict(tf.expand_dims(np.roll(image, -int(k)), 0) , batch_size=1).squeeze())
		#     W.append(2*(1/np.sqrt(k)))

		K.clear_session()

		#E = [maximum_filter(resize(e,(w,h)), int(w/200)) for e in E]
		E = [resize(e,(w,h), preserve_range=True, clip=True) for e in E]

		#est_label = np.median(np.dstack(E), axis=-1)
		est_label = np.average(np.dstack(E), axis=-1, weights=np.array(W))

		est_label /= est_label.max()

		var = np.std(np.dstack(E), axis=-1)

		if np.max(est_label)-np.min(est_label) > .5:
			thres = threshold_otsu(est_label)
			print("Probability of land threshold: %f" % (thres))
		else:
			thres = .9
			print("Default threshold: %f" % (thres))

		conf = 1-est_label
		conf[est_label<thres] = est_label[est_label<thres]
		conf = 1-conf

		conf[np.isnan(conf)] = 0
		conf[np.isinf(conf)] = 0

		model_conf = np.sum(conf)/np.prod(conf.shape)
		print('Overall model confidence = %f'%(model_conf))

		out_stack = np.dstack((est_label,conf,var))
		outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'prob_stack'))

		try:
			os.mkdir(os.path.normpath(sample_direc+os.sep+'prob_stack'))
		except:
			pass

		imsave(outfile.replace('.tif','.png'),(100*out_stack).astype('uint8'),compression=9)

		#yellow = high prob land , high confidence, low variability
		#green = low prob of land, high confidence, low variability
		#purple = high prob land, low confidence, high variability
		#blue = low prob land, low confidence, high variability
		#red = high probability of land, low confidence, low variability

		thres_conf = threshold_otsu(conf)
		thres_var = threshold_otsu(var)
		print("Confidence threshold: %f" % (thres_conf))
		print("Variance threshold: %f" % (thres_var))

		land = (est_label>thres) & (conf>thres_conf) & (var<thres_conf)
		water = (est_label<thres)
		certainty = np.average(np.dstack((np.abs(est_label-thres) , conf , (1-var))), axis=2, weights=[2,1,1])
		outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'certainty'))
		try:
			os.mkdir(os.path.normpath(sample_direc+os.sep+'certainty'))
		except:
			pass

		imsave(outfile,(100*certainty).astype('uint8'),photometric='minisblack',compress=0)

		#land = remove_small_holes(land.astype('uint8'), 5*w)
		#land = remove_small_objects(land.astype('uint8'), 5*w)
		outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks'))

		try:
			os.mkdir(os.path.normpath(sample_direc+os.sep+'masks'))
		except:
			pass

		imsave(outfile.replace('.tif','.jpg'),255*land.astype('uint8'),quality=100)

		elapsed = (time.time() - start)/60
		print("Image masking took "+ str(elapsed) + " minutes")

	else: ###NCLASSES>1

		if 'jpg' in f:
			segfile = f.replace('.jpg', '_predseg.png')
		elif 'png' in f:
			segfile = f.replace('.png', '_predseg.png')

		segfile = os.path.normpath(segfile)
		segfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'out'))

		try:
			os.mkdir(os.path.normpath(sample_direc+os.sep+'out'))
		except:
			pass

		if os.path.exists(segfile):
			print('%s exists ... skipping' % (segfile))
			pass
		else:
			print('%s does not exist ... creating' % (segfile))

		# start = time.time()

		if N_DATA_BANDS<=3:
			image, w, h, bigimage = seg_file2tensor_3band(f)#, resize=True)
			image = image#/255
			bigimage = bigimage#/255
			w = w.numpy(); h = h.numpy()
		else:
			image, w, h, bigimage = seg_file2tensor_4band(f, f.replace('aug_images', 'aug_nir'), resize=True )
			image = image#/255
			bigimage = bigimage#/255
			w = w.numpy(); h = h.numpy()

		print("Working on %i x %i image" % (w,h))

		#image = tf.image.per_image_standardization(image)
		image = standardize(image.numpy())

		# est_label = model.predict(tf.expand_dims(image, 0 , batch_size=1).squeeze()
		est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
		K.clear_session()

		est_label = resize(est_label,(w,h))

		est_label = np.argmax(est_label, -1)

		# conf = np.max(est_label, -1)
		# conf[np.isnan(conf)] = 0
		# conf[np.isinf(conf)] = 0
		#est_label = np.argmax(est_label,-1)
        #print(est_label.shape)

		#est_label = np.squeeze(est_label[:w,:h])

		class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477','#66AA00','#B82E2E', '#316395']
		#add classes for more than 10 classes

		class_label_colormap = class_label_colormap[:NCLASSES]

		try:
			color_label = label_to_colors(est_label, bigimage.numpy()[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
		except:
			color_label = label_to_colors(est_label, bigimage[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)

		if 'jpg' in f:
			imsave(segfile, (color_label).astype(np.uint8), check_contrast=False)
		elif 'png' in f:
			imsave(segfile, (color_label).astype(np.uint8), check_contrast=False)

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


from imports import *

#=======================================================
# model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), BATCH_SIZE, NCLASSES)

print('.....................................')
print('Creating and compiling model ...')

if MODEL =='resunet':
    model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    FILTERS,
                    nclasses=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    kernel_size=(KERNEL,KERNEL),
                    strides=STRIDE,
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    upsample_mode=UPSAMPLE_MODE
                    )
elif MODEL=='unet':
    model =  custom_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    FILTERS,
                    nclasses=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    kernel_size=(KERNEL,KERNEL),
                    strides=STRIDE,
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    )

elif MODEL =='simple_resunet':
    # num_filters = 8 # initial filters
    # model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), num_filters, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))

    model = simple_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                kernel = (2, 2),
                num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                activation="relu",
                use_batch_norm=True,
                upsample_mode=UPSAMPLE_MODE,#"deconv",
                dropout=DROPOUT,#0.1,
                dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                dropout_type=DROPOUT_TYPE,#"standard",
                use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                filters=FILTERS,#8,
                num_layers=4,
                strides=(1,1))
#346,564
elif MODEL=='simple_unet':
    model = simple_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                kernel = (2, 2),
                num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                activation="relu",
                use_batch_norm=True,
                upsample_mode=UPSAMPLE_MODE,#"deconv",
                dropout=DROPOUT,#0.1,
                dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                dropout_type=DROPOUT_TYPE,#"standard",
                use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                filters=FILTERS,#8,
                num_layers=4,
                strides=(1,1))
#242,812

elif MODEL=='satunet':
    #model = sat_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), num_classes=NCLASSES)

    model = custom_satunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                kernel = (2, 2),
                num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                activation="relu",
                use_batch_norm=True,
                upsample_mode=UPSAMPLE_MODE,#"deconv",
                dropout=DROPOUT,#0.1,
                dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                dropout_type=DROPOUT_TYPE,#"standard",
                use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                filters=FILTERS,#8,
                num_layers=4,
                strides=(1,1))

else:
    print("Model must be one of 'unet', 'resunet', or 'satunet'")
    sys.exit(2)


# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])
model.compile(optimizer = 'adam', loss = dice_coef_loss, metrics = [mean_iou, dice_coef])

model.load_weights(weights)


### predict
print('.....................................')
print('Using model for prediction on images ...')

# sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))
sample_filenames = sorted(glob(sample_direc+os.sep+'*.jpg'))
if len(sample_filenames)==0:
    # sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.png'))
    sample_filenames = sorted(glob(sample_direc+os.sep+'*.png'))

print('Number of samples: %i' % (len(sample_filenames)))

for counter,f in enumerate(sample_filenames):
    do_seg(f, model)
    print('%i out of %i done'%(counter,len(sample_filenames)))


# w = Parallel(n_jobs=2, verbose=0, max_nbytes=None)(delayed(do_seg)(f) for f in tqdm(sample_filenames))



#
# # #-----------------------------------
# def seg_file2tensor_3band(f, resize):
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
#         bigimage = tf.image.decode_jpeg(bits)
#     elif 'png' in f:
#         bigimage = tf.image.decode_png(bits)
#
#     w = tf.shape(bigimage)[0]
#     h = tf.shape(bigimage)[1]
#
#     if resize:
#
#         tw = TARGET_SIZE[0]
#         th = TARGET_SIZE[1]
#         resize_crit = (w * th) / (h * tw)
#         image = tf.cond(resize_crit < 1,
#                       lambda: tf.image.resize(bigimage, [w*tw/w, h*tw/w]), # if true
#                       lambda: tf.image.resize(bigimage, [w*th/h, h*th/h])  # if false
#                      )
#
#         nw = tf.shape(image)[0]
#         nh = tf.shape(image)[1]
#         image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
#         # image = tf.cast(image, tf.uint8) #/ 255.0
#
#
#
#         return image, w, h, bigimage
#
#     else:
#         return None, w, h, bigimage
#
