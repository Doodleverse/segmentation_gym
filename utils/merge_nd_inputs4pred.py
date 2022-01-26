# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021-2022, Marda Science LLC
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


# utility to merge multiple coincident jpeg images into nd numpy arrays
import sys,os, time, json, shutil
sys.path.insert(1, '../src')

from skimage.io import imread, imsave
import numpy as np
from tkinter import filedialog, messagebox
from tkinter import *
from glob import glob
from skimage.transform import rescale ## this is actually for resizing
from skimage.morphology import remove_small_objects, remove_small_holes
from tqdm import tqdm
from joblib import Parallel, delayed

###===========================================

#-----------------------------------
# custom 2d resizing functions for 2d discrete labels
def scale(im, nR, nC):
  '''
  for reszing 2d integer arrays
  '''
  nR0 = len(im)     # source number of rows
  nC0 = len(im[0])  # source number of columns
  tmp = [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]
             for c in range(nC)] for r in range(nR)]
  return np.array(tmp).reshape((nR,nC))

#-----------------------------------
def scale_rgb(img, nR, nC, nD):
  '''
  for reszing 3d integer arrays
  '''
  imgout = np.zeros((nR, nC, nD))
  for k in range(3):
      im = img[:,:,k]
      nR0 = len(im)     # source number of rows
      nC0 = len(im[0])  # source number of columns
      tmp = [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                 for c in range(nC)] for r in range(nR)]
      imgout[:,:,k] = np.array(tmp).reshape((nR,nC))
  return imgout

#-----------------------------------
def do_pad_image(f):#, TARGET_SIZE):
    img = imread(f)

    # try:
    #     old_image_height, old_image_width, channels = img.shape
    # except:
    #     old_image_height, old_image_width = img.shape
    #     channels=0
    #
    # # create new image of desired size and color (black) for padding
    # new_image_width = TARGET_SIZE[0]
    # new_image_height = TARGET_SIZE[0]
    # if channels>0:
    #     color = (0,0,0)
    #     result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
    # else:
    #     color = (0)
    #     result = np.full((new_image_height,new_image_width), color, dtype=np.uint8)
    #
    # # compute center offset
    # x_center = (new_image_width - old_image_width) // 2
    # y_center = (new_image_height - old_image_height) // 2
    #
    # try:
    #     # copy img image into center of result image
    #     result[y_center:y_center+old_image_height,
    #            x_center:x_center+old_image_width] = img
    # except:
    #     ## AN ALTERNATIVE WAY - DO NOT REMOVE
    #     # sf = np.minimum(new_image_width/old_image_width,new_image_height/old_image_height)
    #     # if channels>0:
    #     #     img = rescale(img,(sf,sf,1),anti_aliasing=True, preserve_range=True, order=1)
    #     # else:
    #     #     img = rescale(img,(sf,sf),anti_aliasing=True, preserve_range=True, order=1)
    #     # if channels>0:
    #     #     old_image_height, old_image_width, channels = img.shape
    #     # else:
    #     #     old_image_height, old_image_width = img.shape
    #     #
    #     # x_center = (new_image_width - old_image_width) // 2
    #     # y_center = (new_image_height - old_image_height) // 2
    #     #
    #     # result[y_center:y_center+old_image_height,
    #     #        x_center:x_center+old_image_width] = img.astype('uint8')
    #     if channels>0:
    #         result = scale_rgb(img,TARGET_SIZE[0],TARGET_SIZE[1],3)
    #     else:
    #         result = scale(img,TARGET_SIZE[0],TARGET_SIZE[1])


    wend = f.split(os.sep)[-2]
    fdir = os.path.dirname(f)
    fdirout = fdir.replace(wend,'padded_'+wend)
    # save result
    # imsave(fdirout+os.sep+f.split(os.sep)[-1].replace('.jpg','.png'), result.astype('uint8'), check_contrast=False, compression=0)
    imsave(fdirout+os.sep+f.split(os.sep)[-1].replace('.jpg','.png'), img.astype('uint8'), check_contrast=False, compression=0)


#-----------------------------------
root = Tk()
root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select directory for storing OUTPUT files")
output_data_path = root.filename
print(output_data_path)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = output_data_path,title = "Select FIRST directory of IMAGE files")
data_path = root.filename
print(data_path)
root.withdraw()

W=[]
W.append(data_path)

result = 'yes'
while result == 'yes':
    result = messagebox.askquestion("More directories of images?", "More directories of images?", icon='warning')
    if result == 'yes':
        root = Tk()
        root.filename =  filedialog.askdirectory(initialdir =data_path,title = "Select directory of image files")
        data_path = root.filename
        print(data_path)
        root.withdraw()
        W.append(data_path)

##========================================================
## COLLATE FILES INTO LISTS
##========================================================

files = []
for data_path in W:
    f = sorted(glob(data_path+os.sep+'*.jpg'))
    if len(f)<1:
        f = sorted(glob(data_path+os.sep+'images'+os.sep+'*.jpg'))
    files.append(f)

# number of bands x number of samples
files = np.vstack(files).T

##========================================================
## MAKING PADDED (RESIZED) COPIES OF IMAGERY
##========================================================

# ## neeed resizing?
# szs = [imread(f).shape for f in files[:,0]]
# szs = np.vstack(szs)[:,0]
# if len(np.unique(szs))>1:
#     do_resize=True
# else:
#     do_resize=False
#
# from tkinter import simpledialog
# application_window = Tk()
# TARGET_X = simpledialog.askinteger("Imagery are different sizes and will be resized.",
#                                 "What is the TARGET_SIZE (X) of the intended model?",
#                                  parent=application_window,
#                                  minvalue=32, maxvalue=8192)
#
# TARGET_Y = simpledialog.askinteger("Imagery are different sizes and will be resized.",
#                                 "What is the TARGET_SIZE (Y) of the intended model?",
#                                  parent=application_window,
#                                  minvalue=32, maxvalue=8192)
#
# TARGET_SIZE = [TARGET_X,TARGET_Y]

## rersize / pad imagery so all a consistent size (TARGET_SIZE)
# if do_resize:

## make padded direcs
for w in W:
    wend = w.split(os.sep)[-1]
    print(wend)
    newdirec = w.replace(wend,'padded_'+wend)
    try:
        os.mkdir(newdirec)
    except:
        pass


if len(W)==1:
    for file in files:
        w = Parallel(n_jobs=-2, verbose=0, max_nbytes=None)(delayed(do_pad_image)(f) for f in file.squeeze())

else:
    ## cycle through, merge and padd/resize if need to
    for file in files:
        for f in file:
            do_pad_image(f)


## write padded labels to file
# if do_resize:

W2 = []
for w in W:
    wend = w.split(os.sep)[-1]
    w = w.replace(wend,'padded_'+wend)
    W2.append(w)
W = W2
del W2

files = []
for data_path in W:
    f = sorted(glob(data_path+os.sep+'*.png'))
    if len(f)<1:
        f = sorted(glob(data_path+os.sep+'images'+os.sep+'*.png'))
    files.append(f)

# number of bands x number of samples
files = np.vstack(files).T
print("{} sets of {} image files".format(len(W),len(files)))

# else:

# files = sorted(glob(data_path+os.sep+'*.jpg'))
# if len(f)<1:
#     files = sorted(glob(data_path+os.sep+'images'+os.sep+'*.jpg'))
#

###================================================

##========================================================
## NON-AUGMENTED FILES
##========================================================


# ROOT_STRING = 'forpred_'+str(TARGET_SIZE[0])+'_'+str(TARGET_SIZE[1])

## make non-aug subset first
# cycle through pairs of files and labels
for counter,f in enumerate(files):
    im=[] # read all images into a list
    for k in f:
        im.append(imread(k))
    datadict={}
    im=np.dstack(im)# create a dtack which takes care of different sized inputs

    datadict['arr_0'] = im.astype(np.uint8)
    datadict['num_bands'] = im.shape[-1]
    datadict['files'] = [fi.split(os.sep)[-1] for fi in f]
    ROOT_STRING = f[0].split(os.sep)[-1].split('.')[0]
    #print(ROOT_STRING)
    segfile = output_data_path+os.sep+ROOT_STRING+'_noaug_nd_data_000000'+str(counter)+'.npz'
    np.savez_compressed(segfile, **datadict)
    del datadict, im



###================================
from imports import *

#-----------------------------------
def load_npz(example):
    with np.load(example.numpy()) as data:
        image = data['arr_0'].astype('uint8')
        #image = standardize(image)
        file = [''.join(f) for f in data['files']]
    return image, file[0]

@tf.autograph.experimental.do_not_convert
#-----------------------------------
def read_seg_dataset_multiclass(example):
    """
    "read_seg_dataset_multiclass(example)"
    This function reads an example from a npz file into a single image and label
    INPUTS:
        * dataset example object (filename of npz)
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * class_label [tensor array]
    """
    image, file = tf.py_function(func=load_npz, inp=[example], Tout=[tf.float32, tf.string])

    return image, file

###================================

##========================================================
## READ, VERIFY and PLOT NON-AUGMENTED FILES
##========================================================
BATCH_SIZE = 8

filenames = tf.io.gfile.glob(output_data_path+os.sep+'*_noaug*.npz')
dataset = tf.data.Dataset.list_files(filenames, shuffle=False)

print('{} files made'.format(len(filenames)))

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
dataset = dataset.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
dataset = dataset.repeat()
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
dataset = dataset.prefetch(AUTO)

try:
    os.mkdir(output_data_path+os.sep+'noaug_sample')
except:
    pass

print('.....................................')
print('Printing examples to file ...')

counter=0
for imgs,files in dataset.take(10):

  for count,(im, file) in enumerate(zip(imgs, files)):

     im = rescale_array(im.numpy(), 0, 1)
     if im.shape[-1]:
         im = im[:,:,:3]

     plt.imshow(im)

     file = file.numpy()

     plt.axis('off')
     plt.title(file)
     plt.savefig(output_data_path+os.sep+'noaug_sample'+os.sep+ ROOT_STRING + 'noaug_ex'+str(counter)+'.png', dpi=200, bbox_inches='tight')
     #counter +=1
     plt.close('all')
     counter += 1
