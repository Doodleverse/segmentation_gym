# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# utility to merge multiple coincident jpeg images into nd numpy arrays
import sys,os, time, json, shutil)

from skimage.io import imread, imsave
import numpy as np
from tkinter import filedialog, messagebox
from tkinter import *
from glob import glob
from skimage.transform import rescale ## this is actually for resizing
from skimage.morphology import remove_small_objects, remove_small_holes
from tqdm import tqdm

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


root = Tk()
root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select directory for OUTPUT files")
output_data_path = root.filename
print(output_data_path)
root.withdraw()


root = Tk()
root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select directory of LABEL files")
label_data_path = root.filename
print(label_data_path)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select FIRST directory of IMAGE files")
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
        root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select directory of image files")
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

label_files = sorted(glob(label_data_path+os.sep+'*.jpg'))
if len(label_files)<1:
    label_files = sorted(glob(label_data_path+os.sep+'images'+os.sep+'*.jpg'))


##========================================================
## MAKING PADDED (RESIZED) COPIES OF IMAGERY
##========================================================

## neeed resizing?
szs = [imread(f).shape for f in files[:,0]]
szs = np.vstack(szs)[:,0]
if len(np.unique(szs))>1:
    do_resize=True
else:
    do_resize=False

## rersize / pad imagery so all a consistent size (TARGET_SIZE)
if do_resize:

    ## make padded direcs
    for w in W:
        wend = w.split(os.sep)[-1]
        print(wend)
        newdirec = w.replace(wend,'padded_'+wend)
        try:
            os.mkdir(newdirec)
        except:
            pass

    newdireclabels = label_data_path.replace('labels','padded_labels')
    try:
        os.mkdir(newdireclabels)
    except:
        pass

    ## cycle through, merge and padd/resize if need to
    for file,lfile in zip(files, label_files):

        for f in file:
            img = imread(f)

            try:
                old_image_height, old_image_width, channels = img.shape
            except:
                old_image_height, old_image_width = img.shape
                channels=0

            # create new image of desired size and color (black) for padding
            new_image_width = TARGET_SIZE[0]
            new_image_height = TARGET_SIZE[0]
            if channels>0:
                color = (0,0,0)
                result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
            else:
                color = (0)
                result = np.full((new_image_height,new_image_width), color, dtype=np.uint8)

            # compute center offset
            x_center = (new_image_width - old_image_width) // 2
            y_center = (new_image_height - old_image_height) // 2

            try:
                # copy img image into center of result image
                result[y_center:y_center+old_image_height,
                       x_center:x_center+old_image_width] = img
            except:
                ## AN ALTERNATIVE WAY - DO NOT REMOVE
                # sf = np.minimum(new_image_width/old_image_width,new_image_height/old_image_height)
                # if channels>0:
                #     img = rescale(img,(sf,sf,1),anti_aliasing=True, preserve_range=True, order=1)
                # else:
                #     img = rescale(img,(sf,sf),anti_aliasing=True, preserve_range=True, order=1)
                # if channels>0:
                #     old_image_height, old_image_width, channels = img.shape
                # else:
                #     old_image_height, old_image_width = img.shape
                #
                # x_center = (new_image_width - old_image_width) // 2
                # y_center = (new_image_height - old_image_height) // 2
                #
                # result[y_center:y_center+old_image_height,
                #        x_center:x_center+old_image_width] = img.astype('uint8')
                if channels>0:
                    result = scale_rgb(img,TARGET_SIZE[0],TARGET_SIZE[1],3)
                else:
                    result = scale(img,TARGET_SIZE[0],TARGET_SIZE[1])


            wend = f.split(os.sep)[-2]
            fdir = os.path.dirname(f)
            fdirout = fdir.replace(wend,'padded_'+wend)
            # save result
            imsave(fdirout+os.sep+f.split(os.sep)[-1].replace('.jpg','.png'), result.astype('uint8'), check_contrast=False, compression=0)


        ### labels ------------------------------------
        lab = imread(lfile)
        color = (0)
        result = np.full((new_image_height,new_image_width), color, dtype=np.uint8)

        try: #image is smaller
            # copy img image into center of result image
            result[y_center:y_center+old_image_height,
                   x_center:x_center+old_image_width] = lab+1
        except:
            result = scale(lab,TARGET_SIZE[0],TARGET_SIZE[1])+1

            ##lab2 =rescale(lab,(sf,sf),anti_aliasing=True, preserve_range=True, order=0)
            # result[y_center:y_center+old_image_height,
            #        x_center:x_center+old_image_width] = lab2+1
            # del lab2

        # save result
        imsave(lfile.replace('labels','padded_labels').replace('.jpg','.png'), result.astype('uint8'), check_contrast=False, compression=0)

## write padded labels to file
if do_resize:
    label_data_path = label_data_path.replace('labels','padded_labels')

    label_files = sorted(glob(label_data_path+os.sep+'*.png'))
    if len(label_files)<1:
        label_files = sorted(glob(label_data_path+os.sep+'images'+os.sep+'*.png'))
    print("{} label files".format(len(label_files)))

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
    print("{} sets of {} files".format(len(W),len(files)))

else:

    label_files = sorted(glob(label_data_path+os.sep+'*.jpg'))
    if len(label_files)<1:
        label_files = sorted(glob(label_data_path+os.sep+'images'+os.sep+'*.jpg'))
    print("{} label files".format(len(label_files)))

    files = sorted(glob(data_path+os.sep+'*.jpg'))
    if len(f)<1:
        files = sorted(glob(data_path+os.sep+'images'+os.sep+'*.jpg'))

###================================================

##========================================================
## NON-AUGMENTED FILES
##========================================================


print("Creating non-augmented subset")
## make non-aug subset first
# cycle through pairs of files and labels
for counter,(f,l) in enumerate(zip(files,label_files)):
    im=[] # read all images into a list
    for k in f:
        im.append(imread(k))
    datadict={}
    try:
        im=np.dstack(im)# create a dtack which takes care of different sized inputs
        datadict['arr_0'] = im.astype(np.uint8)

        lab = imread(l) # reac the label

        if 'REMAP_CLASSES' in locals():
            for k in REMAP_CLASSES.items():
                lab[lab==int(k[0])] = int(k[1])

        lab[lab>NCLASSES]=NCLASSES

        if len(np.unique(lab))==1:
            nx,ny = lab.shape
            if NCLASSES==1:
                lstack = np.zeros((nx,ny,NCLASSES+1))
            else:
                lstack = np.zeros((nx,ny,NCLASSES))

            lstack[:,:,np.unique(lab)[0]]=np.ones((nx,ny))
        else:
            nx,ny = lab.shape
            if NCLASSES==1:
                lstack = np.zeros((nx,ny,NCLASSES+1))
                lstack[:,:,:NCLASSES+1] = (np.arange(NCLASSES+1) == 1+lab[...,None]-1).astype(int) #one-hot encode
            else:
                lstack = np.zeros((nx,ny,NCLASSES))
                lstack[:,:,:NCLASSES] = (np.arange(NCLASSES) == 1+lab[...,None]-1).astype(int) #one-hot encode

        if FILTER_VALUE>1:

            for kk in range(lstack.shape[-1]):
                lab = remove_small_objects(lstack[:,:,kk].astype('uint8')>0, np.pi*(FILTER_VALUE**2))
                lab = remove_small_holes(lstack[:,:,kk].astype('uint8')>0, np.pi*(FILTER_VALUE**2))
                lstack[:,:,kk] = np.round(lab).astype(np.uint8)
                del lab

        datadict['arr_1'] = np.squeeze(lstack).astype(np.uint8)
        datadict['num_bands'] = im.shape[-1]
        datadict['files'] = [fi.split(os.sep)[-1] for fi in f]
        segfile = output_data_path+os.sep+ROOT_STRING+'_noaug_nd_data_000000'+str(counter)+'.npz'
        np.savez_compressed(segfile, **datadict)
        del datadict, im, lstack
    except:
        print("Inconsistent inputs associated with label file: ".format(l))




###================================

from imports import *

#-----------------------------------
def load_npz(example):
    with np.load(example.numpy()) as data:
        image = data['arr_0'].astype('uint8')
        image = standardize(image)
        label = data['arr_1'].astype('uint8')
        file = [''.join(f) for f in data['files']]
    return image, label, file[0]

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
    image, label, file = tf.py_function(func=load_npz, inp=[example], Tout=[tf.float32, tf.uint8, tf.string])

    if NCLASSES==1:
        label = tf.expand_dims(label,-1)

    return image, label, file

###================================

##========================================================
## READ, VERIFY and PLOT NON-AUGMENTED FILES
##========================================================


filenames = tf.io.gfile.glob(output_data_path+os.sep+ROOT_STRING+'_noaug*.npz')
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

#blue,red, yellow,green
class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477',
                        '#66AA00','#B82E2E', '#316395','#0d0887', '#46039f', '#7201a8',
                        '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']

if NCLASSES>1:
    class_label_colormap = class_label_colormap[:NCLASSES]
else:
    class_label_colormap = class_label_colormap[:NCLASSES+1]


print('.....................................')
print('Printing examples to file ...')

counter=0
for imgs,lbls,files in dataset.take(10):

  for count,(im,lab, file) in enumerate(zip(imgs, lbls, files)):

     im = rescale_array(im.numpy(), 0, 1)
     if im.shape[-1]:
         im = im[:,:,:3]

     plt.imshow(im)

     lab = np.argmax(lab.numpy().squeeze(),-1)

     color_label = label_to_colors(np.squeeze(lab), tf.cast(im[:,:,0]==0,tf.uint8),
                                    alpha=128, colormap=class_label_colormap,
                                     color_class_offset=0, do_alpha=False)

     if NCLASSES==1:
         plt.imshow(color_label, alpha=0.5)#, vmin=0, vmax=NCLASSES)
     else:
         #lab = np.argmax(lab,-1)
         plt.imshow(color_label,  alpha=0.5)#, vmin=0, vmax=NCLASSES)

     file = file.numpy()

     plt.axis('off')
     plt.title(file)
     plt.savefig(output_data_path+os.sep+'noaug_sample'+os.sep+ ROOT_STRING + 'noaug_ex'+str(counter)+'.png', dpi=200, bbox_inches='tight')
     #counter +=1
     plt.close('all')
     counter += 1
