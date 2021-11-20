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


import sys,os, time
sys.path.insert(1, 'src')

import json, shutil
from tkinter import filedialog
from tkinter import *
from random import shuffle
from skimage.morphology import remove_small_objects, remove_small_holes
from tqdm import tqdm

###############################################################
## VARIABLES
###############################################################

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/segmentation_zoo",title = "Select config file",filetypes = (("config files","*.json"),("all files","*.*")))
configfile = root.filename
print(configfile)
root.withdraw()


with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')

USE_GPU = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if USE_GPU == True:
    if 'SET_GPU' in locals():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(SET_GPU)
    else:
        #use the first available GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if N_DATA_BANDS<=3:
    root = Tk()
    root.filename =  filedialog.askdirectory(initialdir = "/segmentation_zoo",title = "Select directory of image files")
    imdir = root.filename
    print(imdir)
    root.withdraw()
elif N_DATA_BANDS==4:
    root = Tk()
    root.filename =  filedialog.askdirectory(initialdir = "/segmentation_zoo",title = "Select directory of RGB image files")
    imdir = root.filename
    print(imdir)
    root.withdraw()

    root = Tk()
    root.filename =  filedialog.askdirectory(initialdir = "/segmentation_zoo",title = "Select directory of additional (4th band) image files")
    nimdir = root.filename
    print(nimdir)
    root.withdraw()


root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/segmentation_zoo",title = "Select directory of label files")
lab_path = root.filename
print(lab_path)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/segmentation_zoo",title = "Select directory to write dataset files")
dataset_dir = root.filename
print(dataset_dir)
root.withdraw()


###############################################################
## FUNCTIONS
###############################################################

from imports import *

#-----------------------------------
def load_npz(example):
    if N_DATA_BANDS==4:
        with np.load(example.numpy()) as data:
            image = data['arr_0'].astype('uint8')
            image = standardize(image)
            nir = data['arr_1'].astype('uint8')
            label = data['arr_2'].astype('uint8')
            file = str(data['arr_2'])
        return image, nir,label, file
    else:
        with np.load(example.numpy()) as data:
            image = data['arr_0'].astype('uint8')
            image = standardize(image)
            label = data['arr_1'].astype('uint8')
            file = str(data['arr_2'])
        return image, label, file

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
    if N_DATA_BANDS==4:
        image, nir, label, file = tf.py_function(func=load_npz, inp=[example], Tout=[tf.uint8, tf.uint8, tf.uint8, tf.string])
        # image, nir, label = tf.py_function(func=load_npz, inp=[example], Tout=[tf.uint8, tf.uint8, tf.uint8])
        nir = tf.cast(nir, tf.float32)#/ 255.0
    else:
        image, label, file = tf.py_function(func=load_npz, inp=[example], Tout=[tf.float32, tf.uint8, tf.string])
        # image, label = tf.py_function(func=load_npz, inp=[example], Tout=[tf.float32, tf.uint8])

    if NCLASSES==1:
        label = tf.expand_dims(label,-1)

    if N_DATA_BANDS==4:
        image = tf.concat([image, tf.expand_dims(nir,-1)],-1)
        return image, label, file
    else:
        return image, label, file

##========================================================
## AUGMENTATION
##========================================================

IMS_PER_SHARD = 1
imdir2 = False

n_im = len(glob(imdir+os.sep+'*.png'))

try:
    os.mkdir(imdir+os.sep+'images')
    # os.mkdir(imdir+os.sep+'aug_images')
except:
    imdir2 = True

if n_im==0:
    if imdir2:
        n_im = len(glob(imdir+os.sep+'images'+os.sep+'*.jpg'))

    else:
        n_im = len(glob(imdir+os.sep+'*.jpg'))

        for file in glob(imdir+os.sep+'*.jpg'):
            shutil.move(file,imdir+os.sep+'images')

else:
    if imdir2:
        print(' ')
    else:
        for file in glob(imdir+os.sep+'*.png'):
            shutil.move(file,imdir+os.sep+'images')

#imdir += os.sep+'images'
print("%i images found" % (n_im))


if USEMASK:
    try:
        os.mkdir(lab_path+os.sep+'masks')
        # os.mkdir(lab_path+os.sep+'aug_masks')
    except:
        imdir2 = True

    if imdir2:
        print(' ')
    else:
        for file in glob(lab_path+os.sep+'*.png'):
            shutil.move(file,lab_path+os.sep+'masks')

        for file in glob(lab_path+os.sep+'*.jpg'):
            shutil.move(file,lab_path+os.sep+'masks')

else:
    try:
        os.mkdir(lab_path+os.sep+'labels')
        # os.mkdir(lab_path+os.sep+'aug_labels')
    except:
        imdir2 = True

    if imdir2:
        print(' ')
    else:
        for file in glob(lab_path+os.sep+'*.png'):
            shutil.move(file,lab_path+os.sep+'labels')

        for file in glob(lab_path+os.sep+'*.jpg'):
            shutil.move(file,lab_path+os.sep+'labels')

if N_DATA_BANDS==4:
    try:
        os.mkdir(nimdir+os.sep+'nir')
        # os.mkdir(imdir2+os.sep+'aug_nir')
    except:
        imdir2 = True

    if imdir2:
        print(' ')
    else:
        for file in glob(nimdir+os.sep+'*.png'):
            shutil.move(file,nimdir+os.sep+'nir')
        for file in glob(nimdir+os.sep+'*.jpg'):
            shutil.move(file,nimdir+os.sep+'nir')


if AUG_LOOPS<2:
    AUG_LOOPS=2

# if DO_AUG:
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=AUG_ROT,
                     width_shift_range=AUG_WIDTHSHIFT,
                     height_shift_range=AUG_HEIGHTSHIFT,
                     fill_mode='reflect', #'nearest',
                     zoom_range=AUG_ZOOM,
                     horizontal_flip=AUG_HFLIP,
                     vertical_flip=AUG_VFLIP)

null_data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=0,
                     width_shift_range=0,
                     height_shift_range=0,
                     fill_mode='reflect',
                     zoom_range=0,
                     horizontal_flip=False,
                     vertical_flip=False)

#get image dimensions
NX = TARGET_SIZE[0]
NY = TARGET_SIZE[1]

null_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**null_data_gen_args)
image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
null_mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**null_data_gen_args)

if N_DATA_BANDS==4:
    image_datagen2 = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

if N_DATA_BANDS==1:
    img_generator = image_datagen.flow_from_directory(
            imdir,
            target_size=(NX, NY),
            batch_size=int(n_im/AUG_LOOPS),
            class_mode=None, seed=SEED, shuffle=False, color_mode="grayscale")

    null_img_generator = null_image_datagen.flow_from_directory(
            imdir,
            target_size=(NX, NY),
            batch_size=int(n_im/AUG_LOOPS),
            class_mode=None, seed=SEED, shuffle=False, color_mode="grayscale")
else:
    img_generator = image_datagen.flow_from_directory(
            imdir,
            target_size=(NX, NY),
            batch_size=int(n_im/AUG_LOOPS),
            class_mode=None, seed=SEED, shuffle=False)

    null_img_generator = null_image_datagen.flow_from_directory(
            imdir,
            target_size=(NX, NY),
            batch_size=int(n_im/AUG_LOOPS),
            class_mode=None, seed=SEED, shuffle=False)

#the seed must be the same as for the training set to get the same images
mask_generator = mask_datagen.flow_from_directory(
        lab_path,
        target_size=(NX, NY),
        batch_size=int(n_im/AUG_LOOPS),
        class_mode=None, seed=SEED, shuffle=False, color_mode="grayscale", interpolation="nearest")

if N_DATA_BANDS==4:
    img_generator2 = image_datagen2.flow_from_directory(
            nimdir,
            target_size=(NX, NY),
            batch_size=int(n_im/AUG_LOOPS),
            class_mode=None, seed=SEED, shuffle=False)

null_mask_generator = null_mask_datagen.flow_from_directory(
        lab_path,
        target_size=(NX, NY),
        batch_size=int(n_im/AUG_LOOPS),
        class_mode=None, seed=SEED, shuffle=False, color_mode="grayscale", interpolation="nearest")



# F1=[]; F2=[]
# for file in img_generator.filepaths:
#     F1.append(file)
# for file2 in mask_generator.filepaths:
#     F2.append(file2)
#
# from skimage.io import imread
# for imfile,labfile in zip(F1,F2):
#     im=imread(imfile)
#     lab=imread(labfile)
#     plt.imshow(im)
#     plt.imshow(lab, alpha=0.5)
#     plt.show()

i = 0
for copy in tqdm(range(AUG_COPIES)):
    for k in range(AUG_LOOPS):

        if N_DATA_BANDS<=3:
            #The following merges the two generators (and their flows) together:
            train_generator = (pair for pair in zip(img_generator, mask_generator))
            #grab a batch of images and label images
            x, y = next(train_generator)

            null_train_generator = (pair for pair in zip(null_img_generator, null_mask_generator))
            #grab a batch of images and label images
            null_x, null_y = next(null_train_generator)

            idx = (img_generator.batch_index - 1) * img_generator.batch_size
            filenames = img_generator.filenames[idx : idx + img_generator.batch_size]

            idx = (null_img_generator.batch_index - 1) * null_img_generator.batch_size
            null_filenames = null_img_generator.filenames[idx : idx + null_img_generator.batch_size]

            # wrute them to file and increment the counter
            for im,nim,lab,nlab,file,nfile in zip(x,null_x,y,null_y, filenames, null_filenames):
            # for im,lab in zip(x,y):

                # plt.imshow(im/255.); plt.imshow(lab, alpha=0.15); plt.savefig('tmp.png', dpi=200); plt.close()
                #
                # plt.imshow(im/255.); plt.imshow(null_lab, alpha=0.15); plt.savefig('tmp2.png', dpi=200); plt.close()
                #

                ##============================================ label
                if NCLASSES==1:
                    lab=lab.squeeze()
                    lab[lab>0]=1

                if NCLASSES==1:
                    l = lab.astype(np.uint8)
                else:
                    l = np.round(lab[:,:,0]).astype(np.uint8)

                if 'REMAP_CLASSES' in locals():
                    for k in REMAP_CLASSES.items():
                        l[l==int(k[0])] = int(k[1])

                l[l>NCLASSES]=NCLASSES

                if len(np.unique(l))==1:
                    nx,ny = l.shape
                    if NCLASSES==1:
                        lstack = np.zeros((nx,ny,NCLASSES+1))
                    else:
                        lstack = np.zeros((nx,ny,NCLASSES))

                    lstack[:,:,np.unique(l)[0]]=np.ones((nx,ny))
                else:
                    nx,ny = l.shape
                    if NCLASSES==1:
                        lstack = np.zeros((nx,ny,NCLASSES+1))
                        lstack[:,:,:NCLASSES+1] = (np.arange(NCLASSES+1) == 1+l[...,None]-1).astype(int) #one-hot encode
                    else:
                        lstack = np.zeros((nx,ny,NCLASSES))
                        lstack[:,:,:NCLASSES] = (np.arange(NCLASSES) == 1+l[...,None]-1).astype(int) #one-hot encode

                if FILTER_VALUE>1:

                    for kk in range(lstack.shape[-1]):
                        #l = median(lstack[:,:,kk], disk(FILTER_VALUE))
                        l = remove_small_objects(lstack[:,:,kk].astype('uint8')>0, np.pi*(FILTER_VALUE**2))
                        l = remove_small_holes(lstack[:,:,kk].astype('uint8')>0, np.pi*(FILTER_VALUE**2))
                        lstack[:,:,kk] = np.round(l).astype(np.uint8)
                        del l

                np.savez_compressed(dataset_dir+os.sep+ROOT_STRING+'augimage_000000'+str(i), im.astype(np.uint8), np.squeeze(lstack).astype(np.uint8), file)

                del lstack, l, im
                ##============================================null label
                if NCLASSES==1:
                    nlab=nlab.squeeze()
                    nlab[nlab>0]=1

                if NCLASSES==1:
                    nl = nlab.astype(np.uint8)
                else:
                    nl = np.round(nlab[:,:,0]).astype(np.uint8)

                if 'REMAP_CLASSES' in locals():
                    for k in REMAP_CLASSES.items():
                        nl[nl==int(k[0])] = int(k[1])

                nl[nl>NCLASSES]=NCLASSES

                if len(np.unique(nl))==1:
                    nx,ny = nl.shape
                    if NCLASSES==1:
                        nlstack = np.zeros((nx,ny,NCLASSES+1))
                    else:
                        nlstack = np.zeros((nx,ny,NCLASSES))

                    nlstack[:,:,np.unique(nl)[0]]=np.ones((nx,ny))
                else:
                    nx,ny = nl.shape
                    if NCLASSES==1:
                        nlstack = np.zeros((nx,ny,NCLASSES+1))
                        nlstack[:,:,:NCLASSES+1] = (np.arange(NCLASSES+1) == 1+nl[...,None]-1).astype(int) #one-hot encode
                    else:
                        nlstack = np.zeros((nx,ny,NCLASSES))
                        nlstack[:,:,:NCLASSES] = (np.arange(NCLASSES) == 1+nl[...,None]-1).astype(int) #one-hot encode

                if FILTER_VALUE>1:

                    for kk in range(nlstack.shape[-1]):
                        nl = remove_small_objects(nlstack[:,:,kk].astype('uint8')>0, np.pi*(FILTER_VALUE**2))
                        nl = remove_small_holes(nlstack[:,:,kk].astype('uint8')>0, np.pi*(FILTER_VALUE**2))
                        nlstack[:,:,kk] = np.round(nl).astype(np.uint8)
                        del nl

                np.savez_compressed(dataset_dir+os.sep+ROOT_STRING+'noaugimage_000000'+str(i),
                                    nim.astype(np.uint8), np.squeeze(nlstack).astype(np.uint8), nfile)

                del nlstack, nl, nim

                i += 1


        elif N_DATA_BANDS==4:
            train_generator = (pair for pair in zip(img_generator, img_generator2, mask_generator))
            x, ii, y = next(train_generator)

            # wrute them to file and increment the counter
            for im,nir,lab in zip(x,ii,y):

                if NCLASSES==1:
                    lab=lab.squeeze()
                    lab[lab>0]=1

                if NCLASSES==1:
                    l = lab.astype(np.uint8)
                else:
                    l = np.round(lab[:,:,0]).astype(np.uint8)

                if 'REMAP_CLASSES' in locals():
                    for k in REMAP_CLASSES.items():
                        l[l==int(k[0])] = int(k[1])

                l[l>NCLASSES]=NCLASSES

                if len(np.unique(l))==1:
                    nx,ny = l.shape
                    if NCLASSES==1:
                        lstack = np.zeros((nx,ny,NCLASSES+1))
                    else:
                        lstack = np.zeros((nx,ny,NCLASSES))

                    lstack[:,:,np.unique(l)[0]]=np.ones((nx,ny))
                else:
                    nx,ny = l.shape
                    if NCLASSES==1:
                        lstack = np.zeros((nx,ny,NCLASSES+1))
                        lstack[:,:,:NCLASSES+1] = (np.arange(NCLASSES+1) == 1+l[...,None]-1).astype(int) #one-hot encode
                    else:
                        lstack = np.zeros((nx,ny,NCLASSES))
                        lstack[:,:,:NCLASSES] = (np.arange(NCLASSES) == 1+l[...,None]-1).astype(int) #one-hot encode
                # else:
                #     lstack = (np.arange(l.max()) == l[...,None]-1).astype(int) #one-hot encode

                if FILTER_VALUE>1:

                    for kk in range(lstack.shape[-1]):
                        #l = median(lstack[:,:,kk], disk(FILTER_VALUE))
                        l = remove_small_objects(lstack[:,:,kk].astype('uint8')>0, np.pi*(FILTER_VALUE**2))
                        l = remove_small_holes(lstack[:,:,kk].astype('uint8')>0, np.pi*(FILTER_VALUE**2))
                        lstack[:,:,kk] = np.round(l).astype(np.uint8)
                        del l

                np.savez_compressed(dataset_dir+os.sep+ROOT_STRING+'augimage_000000'+str(i), im.astype(np.uint8), nir[:,:,0].astype(np.uint8), lstack.astype(np.uint8))

                i += 1


##========================================================
## NPZ CREATION
##========================================================

filenames = tf.io.gfile.glob(dataset_dir+os.sep+ROOT_STRING+'aug*.npz')
# shuffle(filenames)
dataset = tf.data.Dataset.list_files(filenames, shuffle=False)

print('{} files made'.format(len(filenames)))

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
dataset = dataset.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
dataset = dataset.repeat()
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
dataset = dataset.prefetch(AUTO) #

try:
    os.mkdir(dataset_dir+os.sep+'sample')
except:
    pass


class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477',
                        '#66AA00','#B82E2E', '#316395','#0d0887', '#46039f', '#7201a8',
                        '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']

if NCLASSES>1:
    class_label_colormap = class_label_colormap[:NCLASSES]
else:
    class_label_colormap = class_label_colormap[:NCLASSES+1]


print('.....................................')
print('Printing examples to file ...')
if N_DATA_BANDS<=3:
    # plt.figure(figsize=(16,16))
    counter=0
    for imgs,lbls,files in dataset.take(10):
      #print(files)
      for count,(im,lab, file) in enumerate(zip(imgs, lbls, files)):

         im = rescale(im.numpy(), 0, 1)
         plt.imshow(im)

         print(lab.shape)
         lab = np.argmax(lab.numpy().squeeze(),-1)

         #print(np.unique(lab))
         # if len(np.unique(lab))==1:
         #     plt.imshow(im); plt.imshow(lab, alpha=0.5); plt.show()

         color_label = label_to_colors(np.squeeze(lab), tf.cast(im[:,:,0]==0,tf.uint8),
                                        alpha=128, colormap=class_label_colormap,
                                         color_class_offset=0, do_alpha=False)

         if NCLASSES==1:
             plt.imshow(color_label, alpha=0.5)#, vmin=0, vmax=NCLASSES)
         else:
             #lab = np.argmax(lab,-1)
             plt.imshow(color_label,  alpha=0.5)#, vmin=0, vmax=NCLASSES)
         #print(np.unique(lab))

         file = file.numpy()

         plt.axis('off')
         plt.title(file)
         plt.savefig(dataset_dir+os.sep+'sample'+os.sep+ ROOT_STRING + 'aug_ex'+str(counter)+'.png', dpi=200, bbox_inches='tight')
         #counter +=1
         plt.close('all')
         counter += 1


###================================

filenames = tf.io.gfile.glob(dataset_dir+os.sep+ROOT_STRING+'noaug*.npz')
# shuffle(filenames)
dataset = tf.data.Dataset.list_files(filenames, shuffle=False)

print('{} files made'.format(len(filenames)))

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
dataset = dataset.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
dataset = dataset.repeat()
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
dataset = dataset.prefetch(AUTO)

try:
    os.mkdir(dataset_dir+os.sep+'noaug_sample')
except:
    pass


print('.....................................')
print('Printing examples to file ...')
if N_DATA_BANDS<=3:
    # plt.figure(figsize=(16,16))
    counter=0
    for imgs,lbls,files in dataset.take(10):
      #print(files)
      for count,(im,lab, file) in enumerate(zip(imgs, lbls, files)):

         im = rescale(im.numpy(), 0, 1)
         plt.imshow(im)

         print(lab.shape)
         lab = np.argmax(lab.numpy().squeeze(),-1)

         #print(np.unique(lab))
         # if len(np.unique(lab))==1:
         #     plt.imshow(im); plt.imshow(lab, alpha=0.5); plt.show()

         color_label = label_to_colors(np.squeeze(lab), tf.cast(im[:,:,0]==0,tf.uint8),
                                        alpha=128, colormap=class_label_colormap,
                                         color_class_offset=0, do_alpha=False)

         if NCLASSES==1:
             plt.imshow(color_label, alpha=0.5)#, vmin=0, vmax=NCLASSES)
         else:
             #lab = np.argmax(lab,-1)
             plt.imshow(color_label,  alpha=0.5)#, vmin=0, vmax=NCLASSES)
         #print(np.unique(lab))

         file = file.numpy()

         plt.axis('off')
         plt.title(file)
         plt.savefig(dataset_dir+os.sep+'noaug_sample'+os.sep+ ROOT_STRING + 'noaug_ex'+str(counter)+'.png', dpi=200, bbox_inches='tight')
         #counter +=1
         plt.close('all')
         counter += 1




# elif N_DATA_BANDS==4:
#     plt.figure(figsize=(16,16))
#     for imgs,lbls in dataset.take(1):
#       #print(lbls)
#       for count,(im,lab) in enumerate(zip(imgs, lbls)):
#          plt.subplot(int(BATCH_SIZE/2),2,count+1)
#          plt.imshow(im)
#
#          color_label = label_to_colors(np.squeeze(lab), tf.cast(im[:,:,0]==0,tf.uint8),
#                                         alpha=128, colormap=class_label_colormap,
#                                          color_class_offset=0, do_alpha=False)
#
#          if NCLASSES==1:
#              #plt.imshow(lab, cmap='gray', alpha=0.5, vmin=0, vmax=NCLASSES)
#              plt.imshow(color_label, alpha=0.5)#, vmin=0, vmax=NCLASSES)
#
#          else:
#              # lab = np.argmax(lab,-1)
#              # plt.imshow(lab, cmap='bwr', alpha=0.5, vmin=0, vmax=NCLASSES)
#              plt.imshow(color_label, alpha=0.5)#, vmin=0, vmax=NCLASSES)
#
#          plt.axis('off')
#          #print(np.unique(lab))
#          plt.axis('off')
#          plt.savefig(dataset_dir+os.sep+'sample'+os.sep+ROOT_STRING+'ex'+str(count)+'.png', dpi=200, bbox_inches='tight')
#          plt.close('all')


                # l = np.round(lab[:,:,0]).astype(np.uint8)
                # # if FILTER_VALUE>1:
                # #     l = np.round(median(l, disk(FILTER_VALUE))).astype(np.uint8)
                #
                # if 'REMAP_CLASSES' not in locals():
                #     if np.min(l)==1:
                #         l -= 1
                #     if NCLASSES==1:
                #         l[l>0]=1 #null is water
                #
                # if 'REMAP_CLASSES' in locals():
                #     for k in REMAP_CLASSES.items():
                #         l[l==int(k[0])] = int(k[1])
                #
                #
                # l[l>NCLASSES]=NCLASSES
                #
                # if len(np.unique(l))==1:
                #     nx,ny = l.shape
                #     lstack = np.zeros((nx,ny,NCLASSES))
                #     lstack[:,:,np.unique(l)[0]]=np.ones((nx,ny))
                # else:
                #     nx,ny = l.shape
                #     lstack = np.zeros((nx,ny,NCLASSES))
                #
                #     lstack[:,:,:NCLASSES] = (np.arange(NCLASSES) == 1+l[...,None]-1).astype(int) #one-hot encode
                # # else:
                # #     lstack = (np.arange(l.max()) == l[...,None]-1).astype(int) #one-hot encode
                #
                # if FILTER_VALUE>1:
                #
                #     for kk in range(lstack.shape[-1]):
                #         #l = median(lstack[:,:,kk], disk(FILTER_VALUE))
                #         l = remove_small_objects(lstack[:,:,kk].astype('uint8')>0, np.pi*(FILTER_VALUE**2))
                #         l = remove_small_holes(lstack[:,:,kk].astype('uint8')>0, np.pi*(FILTER_VALUE**2))
                #         lstack[:,:,kk] = np.round(l).astype(np.uint8)
                #         del l
