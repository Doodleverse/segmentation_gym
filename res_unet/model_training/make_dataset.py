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

import os, json, shutil
from tkinter import filedialog
from tkinter import *
from random import shuffle

###############################################################
## VARIABLES
###############################################################

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/weights",title = "Select config file",filetypes = (("config files","*.json"),("all files","*.*")))
configfile = root.filename
print(configfile)
root.withdraw()


with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')

if N_DATA_BANDS<=3:
    root = Tk()
    root.filename =  filedialog.askdirectory(initialdir = "/data",title = "Select directory of image files")
    imdir = root.filename
    print(imdir)
    root.withdraw()
elif N_DATA_BANDS==4:
    root = Tk()
    root.filename =  filedialog.askdirectory(initialdir = "/data",title = "Select directory of RGB image files")
    imdir = root.filename
    print(imdir)
    root.withdraw()

    root = Tk()
    root.filename =  filedialog.askdirectory(initialdir = "/data",title = "Select directory of additional (4th band) image files")
    nimdir = root.filename
    print(nimdir)
    root.withdraw()


root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/data",title = "Select directory of label files")
lab_path = root.filename
print(lab_path)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/data",title = "Select directory to write dataset files")
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
            nir = data['arr_1'].astype('uint8')
            label = data['arr_2'].astype('uint8')
        return image, nir,label
    else:
        with np.load(example.numpy()) as data:
            image = data['arr_0'].astype('uint8')
            label = data['arr_1'].astype('uint8')
        return image, label

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
        image, nir, label = tf.py_function(func=load_npz, inp=[example], Tout=[tf.uint8, tf.uint8, tf.uint8])
        nir = tf.cast(nir, tf.float32)/ 255.0
    else:
        image, label = tf.py_function(func=load_npz, inp=[example], Tout=[tf.uint8, tf.uint8])

    image = tf.cast(image, tf.float32)/ 255.0
    label = tf.cast(label, tf.uint8)

    if NCLASSES==1:
        label = tf.expand_dims(label,-1)

    if N_DATA_BANDS==4:
        image = tf.concat([image, tf.expand_dims(nir,-1)],-1)
        return image, label
    else:
        return image, label

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

# print(imdir)
# print(lab_path)

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

#get image dimensions
# NX, NY, NZ = imread(glob(imdir+os.sep+'images'+os.sep+'*.jpg')[0]).shape

NX = TARGET_SIZE[0]
NY = TARGET_SIZE[1]

image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

if N_DATA_BANDS==4:
    image_datagen2 = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

if N_DATA_BANDS==1:
    img_generator = image_datagen.flow_from_directory(
            imdir,
            target_size=(NX, NY),
            batch_size=int(n_im/AUG_LOOPS),
            class_mode=None, seed=SEED, shuffle=True, color_mode="grayscale")
else:
    img_generator = image_datagen.flow_from_directory(
            imdir,
            target_size=(NX, NY),
            batch_size=int(n_im/AUG_LOOPS),
            class_mode=None, seed=SEED, shuffle=True)

#the seed must be the same as for the training set to get the same images
mask_generator = mask_datagen.flow_from_directory(
        lab_path,
        target_size=(NX, NY),
        batch_size=int(n_im/AUG_LOOPS),
        class_mode=None, seed=SEED, shuffle=True, color_mode="grayscale", interpolation="nearest")

if N_DATA_BANDS==4:
    img_generator2 = image_datagen2.flow_from_directory(
            nimdir,
            target_size=(NX, NY),
            batch_size=int(n_im/AUG_LOOPS),
            class_mode=None, seed=SEED, shuffle=True)

i = 0
for copy in range(AUG_COPIES):
    for k in range(AUG_LOOPS):

        if N_DATA_BANDS<=3:
            #The following merges the two generators (and their flows) together:
            train_generator = (pair for pair in zip(img_generator, mask_generator))
            #grab a batch of images and label images
            x, y = next(train_generator)

            # wrute them to file and increment the counter
            for im,lab in zip(x,y):
                l = np.round(lab[:,:,0]).astype(np.uint8)
                if MEDIAN_FILTER_VALUE>1:
                    l = np.round(median(l, disk(MEDIAN_FILTER_VALUE))).astype(np.uint8)
                #print(np.unique(l.flatten()))

                if 'REMAP_CLASSES' not in locals():
                    if NCLASSES==1:
                        l[l>0]=1

                if 'REMAP_CLASSES' in locals():
                    for k in REMAP_CLASSES.items():
                        l[l==int(k[0])] = int(k[1])

                l[l>NCLASSES]=NCLASSES

                lstack = (np.arange(l.max()) == l[...,None]-1).astype(int) #one-hot encode
                if lstack.shape[-1]!=NCLASSES:
                    lstack = np.dstack(( lstack, np.zeros(lstack.shape[:2]).astype(np.uint8) ))

                #print(lstack.shape)

                try:

                    if NCLASSES>1:

                        if DO_CRF_REFINE:
                            for kk in range(lstack.shape[-1]):
                                #print(k)
                                l,_ = crf_refine(lstack[:,:,kk], im, nclasses = NCLASSES, theta_col=40, theta_spat=1, compat=100)
                                if MEDIAN_FILTER_VALUE>1:
                                    lstack[:,:,kk] = np.round(median(l, disk(MEDIAN_FILTER_VALUE))).astype(np.uint8)
                                else:
                                    lstack[:,:,kk] = np.round(l).astype(np.uint8)

                        #for kk in range(lstack.shape[-1]):
                        if USEMASK:
                            np.savez(dataset_dir+os.sep+ROOT_STRING+'augimage_000000'+str(i), im.astype(np.uint8), lstack.astype(np.uint8))
                        else:
                            np.savez(dataset_dir+os.sep+ROOT_STRING+'augimage_000000'+str(i), im.astype(np.uint8), lstack.astype(np.uint8))
                    else:
                        if USEMASK:
                            np.savez(dataset_dir+os.sep+ROOT_STRING+'augimage_000000'+str(i), im.astype(np.uint8), np.squeeze(l).astype(np.uint8))
                        else:
                            np.savez(dataset_dir+os.sep+ROOT_STRING+'augimage_000000'+str(i), im.astype(np.uint8), np.squeeze(l).astype(np.uint8))

                except:
                    print('Error ')
                    pass

                i += 1


        elif N_DATA_BANDS==4:
            train_generator = (pair for pair in zip(img_generator, img_generator2, mask_generator))
            x, ii, y = next(train_generator)

            # wrute them to file and increment the counter
            for im,nir,lab in zip(x,ii,y):
                l = np.round(lab[:,:,0]).astype(np.uint8)
                if MEDIAN_FILTER_VALUE>1:
                    l = np.round(median(l, disk(MEDIAN_FILTER_VALUE))).astype(np.uint8)

                if 'REMAP_CLASSES' not in locals():
                    if NCLASSES==1:
                        l[l>0]=1 #null is water

                if 'REMAP_CLASSES' in locals():
                    for k in REMAP_CLASSES.items():
                        l[l==int(k[0])] = int(k[1])

                l[l>NCLASSES]=NCLASSES
                #print(np.unique(l.flatten()))
                lstack = (np.arange(l.max()) == l[...,None]-1).astype(int) #one-hot encode
                if lstack.shape[-1]!=NCLASSES:
                    lstack = np.dstack(( lstack, np.zeros(lstack.shape[:2]).astype(np.uint8) ))

                try:

                    if NCLASSES>1:

                        if DO_CRF_REFINE:
                            for kk in range(lstack.shape[-1]):
                                #print(k)
                                l,_ = crf_refine(lstack[:,:,kk], im, nclasses = NCLASSES, theta_col=40, theta_spat=1, compat=100)
                                if MEDIAN_FILTER_VALUE>1:
                                    lstack[:,:,kk] = np.round(median(l, disk(MEDIAN_FILTER_VALUE))).astype(np.uint8)
                                else:
                                    lstack[:,:,kk] = np.round(l).astype(np.uint8)

                        if USEMASK:
                            np.savez(dataset_dir+os.sep+ROOT_STRING+'augimage_000000'+str(i), im.astype(np.uint8), nir[:,:,0].astype(np.uint8), lstack.astype(np.uint8))
                        else:
                            np.savez(dataset_dir+os.sep+ROOT_STRING+'augimage_000000'+str(i), im.astype(np.uint8), nir[:,:,0].astype(np.uint8), lstack.astype(np.uint8))
                    else:
                        if USEMASK:
                            np.savez(dataset_dir+os.sep+ROOT_STRING+'augimage_000000'+str(i), im.astype(np.uint8), nir[:,:,0].astype(np.uint8), np.squeeze(l).astype(np.uint8))
                        else:
                            np.savez(dataset_dir+os.sep+ROOT_STRING+'augimage_000000'+str(i), im.astype(np.uint8), nir[:,:,0].astype(np.uint8), np.squeeze(l).astype(np.uint8))

                except:
                    print('Error ')
                    pass

                i += 1

        #save memory
        del x, y, im, lab
        #get a new batch



##========================================================
## NPZ CREATION
##========================================================

filenames = tf.io.gfile.glob(dataset_dir+os.sep+ROOT_STRING+'*.npz')
shuffle(filenames)
dataset = tf.data.Dataset.list_files(filenames, shuffle=False)

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
dataset = dataset.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
dataset = dataset.repeat()
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
dataset = dataset.prefetch(AUTO) #

# if N_DATA_BANDS==4:
for imgs,lbls in dataset.take(1):
    print(imgs.shape)
    print(lbls.shape)
# else:
#     for imgs,nirs,lbls in dataset.take(1):
#         print(imgs.shape)
#         print(nirs.shape)
#         print(lbls.shape)

print('.....................................')
print('Printing examples to file ...')
if N_DATA_BANDS<=3:
    plt.figure(figsize=(16,16))
    for imgs,lbls in dataset.take(1):
      #print(lbls)
      for count,(im,lab) in enumerate(zip(imgs, lbls)):
         plt.subplot(int(BATCH_SIZE/2),2,count+1)
         plt.imshow(im)
         if NCLASSES==1:
             plt.imshow(lab, cmap='gray', alpha=0.5, vmin=0, vmax=NCLASSES)
         else:
             lab = np.argmax(lab,-1)
             plt.imshow(lab, cmap='bwr', alpha=0.5, vmin=0, vmax=NCLASSES)

         plt.axis('off')
         print(np.unique(lab))
         plt.axis('off')
         plt.savefig(ROOT_STRING+'ex'+str(count)+'.png', dpi=200, bbox_inches='tight')
         #counter +=1
         plt.close('all')
elif N_DATA_BANDS==4:
    plt.figure(figsize=(16,16))
    for imgs,lbls in dataset.take(1):
      #print(lbls)
      for count,(im,lab) in enumerate(zip(imgs, lbls)):
         plt.subplot(int(BATCH_SIZE/2),2,count+1)
         plt.imshow(im)
         if NCLASSES==1:
             plt.imshow(lab, cmap='gray', alpha=0.5, vmin=0, vmax=NCLASSES)
         else:
             lab = np.argmax(lab,-1)
             plt.imshow(lab, cmap='bwr', alpha=0.5, vmin=0, vmax=NCLASSES)

         plt.axis('off')
         print(np.unique(lab))
         plt.axis('off')
         plt.savefig(ROOT_STRING+'ex'+str(count)+'.png', dpi=200, bbox_inches='tight')
         plt.close('all')





#
# @tf.autograph.experimental.do_not_convert
# #-----------------------------------
# def read_seg_tfrecord_multiclass(example):
#     """
#     "read_seg_tfrecord_multiclass(example)"
#     This function reads an example from a npz file into a single image and label
#     INPUTS:
#         * TFRecord example object (filename of npz)
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: TARGET_SIZE
#     OUTPUTS:
#         * image [tensor array]
#         * class_label [tensor array]
#     """
#     image, label = tf.py_function(func=load_npz, inp=[example], Tout=[tf.uint8, tf.uint8])
#
#     image = tf.cast(image, tf.float32)/ 255.0
#     label = tf.cast(label, tf.uint8)
#     return tf.squeeze(image), tf.squeeze(label)


                            #imsave(lab_path+os.sep+os.sep+'aug_masks'+os.sep+'augimage_000000'+str(i)+'_'+str(k)+'.jpg', lstack[:,:,k].astype(np.uint8), quality=100, check_contrast=False)
                            #imsave(lab_path+os.sep+os.sep+'aug_masks'+os.sep+'augimage_000000'+str(i)+'_'+str(k)+'.png', lstack[:,:,k].astype(np.uint8), compression=0, check_contrast=False)

                            #imsave(lab_path+os.sep+os.sep+'aug_labels'+os.sep+'augimage_000000'+str(i)+'_'+str(k)+'.jpg', lstack[:,:,k].astype(np.uint8), quality=100, check_contrast=False)
                            #imsave(lab_path+os.sep+os.sep+'aug_labels'+os.sep+'augimage_000000'+str(i)+'_'+str(k)+'.png', lstack[:,:,k].astype(np.uint8), compression=0, check_contrast=False)

                            #imsave(lab_path+os.sep+os.sep+'aug_masks'+os.sep+'augimage_000000'+str(i)+'.jpg', np.squeeze(l).astype(np.uint8), quality=100, check_contrast=False)
                            #imsave(lab_path+os.sep+os.sep+'aug_masks'+os.sep+'augimage_000000'+str(i)+'.png', np.squeeze(l).astype(np.uint8), compression=0, check_contrast=False)
                            #imsave(lab_path+os.sep+os.sep+'aug_labels'+os.sep+'augimage_000000'+str(i)+'.jpg', np.squeeze(l).astype(np.uint8), quality=100, check_contrast=False)
                            #imsave(lab_path+os.sep+os.sep+'aug_labels'+os.sep+'augimage_000000'+str(i)+'.png', np.squeeze(l).astype(np.uint8), compression=0, check_contrast=False)
                    #imsave(imdir+os.sep+'aug_images'+os.sep+'augimage_000000'+str(i)+'.jpg', im.astype(np.uint8), quality=100, check_contrast=False)
                    #imsave(imdir+os.sep+'aug_images'+os.sep+'augimage_000000'+str(i)+'.png', im.astype(np.uint8), compression=0, check_contrast=False)
                    #np.savez(imdir+os.sep+'aug_images'+os.sep+'augimage_000000'+str(i), im.astype(np.uint8))
                # if NCLASSES>1:
                #     l[l==0]=1
                    #if NCLASSES>1:
                        # for k in range(lstack.shape[-1]):
                            #imsave(lab_path+os.sep+os.sep+'aug_masks'+os.sep+'augimage_000000'+str(i)+'_'+str(k)+'.jpg', lstack[:,:,k].astype(np.uint8), quality=100, check_contrast=False)
                            #imsave(lab_path+os.sep+os.sep+'aug_masks'+os.sep+'augimage_000000'+str(i)+'_'+str(k)+'.png', lstack[:,:,k].astype(np.uint8), compression=0, check_contrast=False)
                            #imsave(lab_path+os.sep+os.sep+'aug_labels'+os.sep+'augimage_000000'+str(i)+'_'+str(k)+'.jpg', lstack[:,:,k].astype(np.uint8), quality=100, check_contrast=False)
                            #imsave(lab_path+os.sep+os.sep+'aug_labels'+os.sep+'augimage_000000'+str(i)+'_'+str(k)+'.png', lstack[:,:,k].astype(np.uint8), compression=0, check_contrast=False)
                            #imsave(lab_path+os.sep+os.sep+'aug_masks'+os.sep+'augimage_000000'+str(i)+'.jpg', np.squeeze(l).astype(np.uint8), quality=100, check_contrast=False)
                            #imsave(lab_path+os.sep+os.sep+'aug_masks'+os.sep+'augimage_000000'+str(i)+'.png', np.squeeze(l).astype(np.uint8), compression=0, check_contrast=False)
                            #imsave(lab_path+os.sep+os.sep+'aug_labels'+os.sep+'augimage_000000'+str(i)+'.jpg', np.squeeze(l).astype(np.uint8), quality=100, check_contrast=False)
                            #imsave(lab_path+os.sep+os.sep+'aug_labels'+os.sep+'augimage_000000'+str(i)+'.png', np.squeeze(l).astype(np.uint8), compression=0, check_contrast=False)

                    #imsave(imdir+os.sep+'aug_images'+os.sep+'augimage_000000'+str(i)+'.jpg', im.astype(np.uint8), quality=100, check_contrast=False)
                    #imsave(imdir+os.sep+'aug_images'+os.sep+'augimage_000000'+str(i)+'.png', im.astype(np.uint8), compression=0, check_contrast=False)
                    #np.savez(imdir+os.sep+'aug_images'+os.sep+'augimage_000000'+str(i), im.astype(np.uint8))

                    #imsave(imdir2+os.sep+'aug_nir'+os.sep+'augimage_000000'+str(i)+'.jpg', nir.astype(np.uint8), quality=100, check_contrast=False)
                    #imsave(imdir2+os.sep+'aug_nir'+os.sep+'augimage_000000'+str(i)+'.png', nir.astype(np.uint8), compression=0, check_contrast=False)
                    #np.savez(imdir2+os.sep+'aug_nir'+os.sep+'augimage_000000'+str(i), nir.astype(np.uint8))

#================================================================
# images = sorted(tf.io.gfile.glob(imdir+os.sep+'aug_images'+os.sep+'*.png'))
# if USEMASK:
# images = sorted(tf.io.gfile.glob(tfrecord_dir+os.sep+'*.npz'))
# else:
#     images = sorted(tf.io.gfile.glob(imdir.replace('images', 'labels')+os.sep+'aug_labels'+os.sep+'*.npz'))
# #
# else:
#
#     images = sorted(tf.io.gfile.glob(imdir+os.sep+'images'+os.sep+'*.jpg'))
#


# nb_images=len(images)
# print(nb_images)

# SHARDS = int(nb_images / IMS_PER_SHARD) + (1 if nb_images % IMS_PER_SHARD != 0 else 0)
#
# shared_size = int(np.ceil(1.0 * nb_images / SHARDS))
#
# if DO_AUG:
#     dataset = get_seg_dataset_for_tfrecords(imdir+os.sep+'aug_images',  shared_size) #[], lab_path+os.sep+'aug_labels',
# else:
#     dataset = get_seg_dataset_for_tfrecords(imdir+os.sep+'images',  shared_size) #[], lab_path+os.sep+'aug_labels',

# # write tfrecords
# if N_DATA_BANDS<=3:
#     write_seg_records(dataset, tfrecord_dir, ROOT_STRING)
# elif N_DATA_BANDS==4:
#     write_seg_records_4bands(dataset, tfrecord_dir, ROOT_STRING)


# counter = 0
# for imgs,lbls in dataset:
#     try:
#         if N_DATA_BANDS<=3:
#         ##print(imgs.shape)
#             np.savez(tfrecord_dir+os.sep+ROOT_STRING+str(counter), imgs, lbls)
#         elif N_DATA_BANDS==4:
#             np.savez(tfrecord_dir+os.sep+ROOT_STRING+str(counter), imgs, nirs, lbls)
#         counter+=1
#     except:
#         print('Error %i' % counter)

# path = tfrecord_dir+os.sep+ROOT_STRING+str(counter)+'.npz'
# with np.load(path) as data:
#     train_examples = data['arr_0'].astype('uint8')
#     train_labels = data['arr_1'].astype('uint8')
#plt.imshow(train_examples[0]); plt.imshow(np.argmax(train_labels[0],-1), alpha=0.4); plt.show()


#
# #visualize some examples
# counter = 0
# if N_DATA_BANDS<=3:
#
#     for imgs,lbls in dataset.take(1):
#
#       for count,(im,lab) in enumerate(zip(imgs,lbls)):
#          print(im.shape)
#          print(lab.shape)
#          if N_DATA_BANDS==4:
#              plt.imshow(im[:,:,:3]) #tf.image.decode_png(im))#jpeg(im, channels=3))
#          else:
#              plt.imshow(im) #tf.image.decode_png(im)) #jpeg(im, channels=N_DATA_BANDS))
#
#          if NCLASSES==1:
#              #lab = tf.image.decode_png(lab, channels=0)
#              plt.imshow(lab, alpha=0.3, cmap='bwr',vmin=0, vmax=NCLASSES)
#          else:
#              #lab = tf.argmax(lab, -1) #tf.image.decode_png(lab, channels=0)
#              plt.imshow(lab, alpha=0.3, cmap='bwr',vmin=0, vmax=NCLASSES)
#          plt.axis('off')
#          plt.savefig('ex'+str(counter)+'.png', dpi=200, bbox_inches='tight')
#          counter +=1
#          plt.close('all')
#
# elif N_DATA_BANDS==4:
#
#     for imgs,nirs,lbls in dataset.take(1):
#
#       for count,(im,ii,lab) in enumerate(zip(imgs,nirs,lbls)):
#
#          #im = tf.image.decode_png(im) #jpeg(im, channels=3)
#          plt.imshow(im)
#
#          if NCLASSES==1:
#              plt.imshow(lab, alpha=0.3, cmap='bwr',vmin=0, vmax=NCLASSES)
#          else:
#              #lab = tf.argmax(lab, -1) #tf.image.decode_png(lab, channels=0)
#              plt.imshow(lab, alpha=0.3, cmap='bwr',vmin=0, vmax=NCLASSES)
#
#          plt.colorbar()
#          plt.axis('off')
#          plt.savefig('ex'+str(counter)+'.png', dpi=200, bbox_inches='tight')
#          plt.close('all')
#
#          plt.imshow(tf.image.decode_png(ii), cmap='gray') #jpeg(ii, channels=1), cmap='gray')
#          plt.imshow(lab, alpha=0.3, cmap='bwr',vmin=0, vmax=NCLASSES)
#          plt.colorbar()
#          plt.axis('off')
#          plt.savefig('ex'+str(counter)+'_band4.png', dpi=200, bbox_inches='tight')
#          plt.close('all')
#          counter +=1
#


# #-----------------------------------
# def read_seg_image_and_label(img_path):
#     """
#     "read_seg_image_and_label_obx(img_path)"
#     This function reads an image and label and decodes both jpegs
#     into bytestring arrays.
#     This works by parsing out the label image filename from its image pair
#     Thre are different rules for non-augmented versus augmented imagery
#     INPUTS:
#         * img_path [tensor string]
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS:
#         * image [bytestring]
#         * label [bytestring]
#     """
#     bits = tf.io.read_file(img_path)
#     image = tf.image.decode_png(bits) #jpeg(bits)
#
#     # have to use this tf.strings.regex_replace utility because img_path is a Tensor object
#     if USEMASK:
#         lab_path = tf.strings.regex_replace(img_path, "images", "masks")
#
#     else:
#         lab_path = tf.strings.regex_replace(img_path, "images", "labels")
#
#
#     if NCLASSES==1:
#         bits = tf.io.read_file(lab_path)
#         label = tf.image.decode_png(bits) #jpeg(bits)
#     else:
#         L = []
#         for k in range(NCLASSES):
#             bits = tf.io.read_file(tf.strings.regex_replace(lab_path, ".png", "_"+str(k)+".png"))
#             L.append(tf.image.decode_png(bits)) #jpeg(bits))
#
#         label = tf.squeeze(tf.cast(L, tf.uint8))
#         label = tf.transpose(label, perm=[1,2,0]) # X, Y, N_DATA_BANDS
#
#     return image, label
#
# #-----------------------------------
# def read_seg_image_and_label_4bands(img_path): #, nir_path):
#     """
#     This function reads an image and label and decodes both jpegs
#     into bytestring arrays.
#     This works by parsing out the label image filename from its image pair
#     Thre are different rules for non-augmented versus augmented imagery
#     INPUTS:
#         * img_path [tensor string]
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS:
#         * image [bytestring]
#         * label [bytestring]
#     """
#     bits = tf.io.read_file(img_path)
#     image = tf.image.decode_png(bits) #jpeg(bits)
#
#     # have to use this tf.strings.regex_replace utility because img_path is a Tensor object
#     if USEMASK:
#         lab_path = tf.strings.regex_replace(img_path, "images", "masks")
#
#     else:
#         lab_path = tf.strings.regex_replace(img_path, "images", "labels")
#
#
#     if NCLASSES==1:
#         bits = tf.io.read_file(lab_path)
#         label = tf.image.decode_png(bits)#jpeg(bits)
#     else:
#         L = []
#         for k in range(NCLASSES):
#             bits = tf.io.read_file(tf.strings.regex_replace(lab_path, ".png", "_"+str(k)+".png"))
#             L.append(tf.image.decode_png(bits))#jpeg(bits))
#
#         label = tf.squeeze(tf.cast(L, tf.uint8))
#         label = tf.transpose(label, perm=[1,2,0])
#
#     nir_path = tf.strings.regex_replace(img_path, "images", "nir")
#
#     bits = tf.io.read_file(nir_path)
#     nir = tf.image.decode_png(bits) #jpeg(bits)
#
#     return image, nir, label
#
# #-----------------------------------
# def resize_and_crop_seg_image(image, label):
#     """
#     "resize_and_crop_seg_image_obx"
#     This function crops to square and resizes an image and label
#     INPUTS:
#         * image [tensor array]
#         * label [tensor array]
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: TARGET_SIZE
#     OUTPUTS:
#         * image [tensor array]
#         * label [tensor array]
#     """
#     w = tf.shape(image)[0]
#     h = tf.shape(image)[1]
#     tw = TARGET_SIZE[0]
#     th = TARGET_SIZE[1]
#     resize_crit = (w * th) / (h * tw)
#     image = tf.cond(resize_crit < 1,
#                   lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
#                   lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
#                  )
#     nw = tf.shape(image)[0]
#     nh = tf.shape(image)[1]
#     image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
#
#     label = tf.cond(resize_crit < 1,
#                   lambda: tf.image.resize(label, [w*tw/w, h*tw/w]), # if true
#                   lambda: tf.image.resize(label, [w*th/h, h*th/h])  # if false
#                  )
#     label = tf.image.crop_to_bounding_box(label, (nw - tw) // 2, (nh - th) // 2, tw, th)
#
#     return image, label
#
#
# #-----------------------------------
# def resize_and_crop_seg_image_4bands(image, nir, label):
#     """
#     This function crops to square and resizes an image and label
#     INPUTS:
#         * image [tensor array]
#         * label [tensor array]
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: TARGET_SIZE
#     OUTPUTS:
#         * image [tensor array]
#         * label [tensor array]
#     """
#     w = tf.shape(image)[0]
#     h = tf.shape(image)[1]
#     tw = TARGET_SIZE[0]
#     th = TARGET_SIZE[1]
#     resize_crit = (w * th) / (h * tw)
#     image = tf.cond(resize_crit < 1,
#                   lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
#                   lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
#                  )
#     nw = tf.shape(image)[0]
#     nh = tf.shape(image)[1]
#     image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
#
#     label = tf.cond(resize_crit < 1,
#                   lambda: tf.image.resize(label, [w*tw/w, h*tw/w]), # if true
#                   lambda: tf.image.resize(label, [w*th/h, h*th/h])  # if false
#                  )
#     label = tf.image.crop_to_bounding_box(label, (nw - tw) // 2, (nh - th) // 2, tw, th)
#
#
#     nir = tf.cond(resize_crit < 1,
#                   lambda: tf.image.resize(nir, [w*tw/w, h*tw/w]), # if true
#                   lambda: tf.image.resize(nir, [w*th/h, h*th/h])  # if false
#                  )
#     nir = tf.image.crop_to_bounding_box(nir, (nw - tw) // 2, (nh - th) // 2, tw, th)
#
#     return image, nir, label
#
# #-----------------------------------
# def get_seg_dataset_for_tfrecords(imdir, shared_size):
#     """
#     "get_seg_dataset_for_tfrecords"
#     This function reads an image and label and decodes both jpegs
#     into bytestring arrays.
#     INPUTS:
#         * image [tensor array]
#         * label [tensor array]
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: TARGET_SIZE
#     OUTPUTS:
#         * image [tensor array]
#         * label [tensor array]
#     """
#     dataset = tf.data.Dataset.list_files(imdir+os.sep+'*.png', seed=10000) # This also shuffles the images
#     if N_DATA_BANDS<=3:
#         dataset = dataset.map(read_seg_image_and_label)
#         dataset = dataset.map(resize_and_crop_seg_image, num_parallel_calls=AUTO)
#         # dataset = dataset.map(recompress_seg_image, num_parallel_calls=AUTO)
#     elif N_DATA_BANDS==4:
#         dataset = dataset.map(read_seg_image_and_label_4bands)
#         dataset = dataset.map(resize_and_crop_seg_image_4bands, num_parallel_calls=AUTO)
#         # dataset = dataset.map(recompress_seg_image_4bands, num_parallel_calls=AUTO)
#
#     dataset = dataset.batch(shared_size)
#     return dataset








# #-----------------------------------
# def recompress_seg_image_4bands(image, nir_image, label):
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
#     nir_image = tf.cast(nir_image, tf.uint8)
#     nir_image = tf.image.encode_png(nir_image, compression=0) #jpeg(nir_image, optimize_size=False, chroma_downsampling=False, quality=100, x_density=1000, y_density=1000)
#
#     label = tf.cast(label, tf.uint8)
#
#     label = tf.image.encode_png(label, compression=0) #, optimize_size=False, chroma_downsampling=False, quality=100, x_density=1000, y_density=1000)
#
#     return image, nir_image, label




#
         # if NCLASSES>1:
         #     if DO_CRF_REFINE:
         #         l = np.argmax(lab, -1)
         #         lstack = (np.arange(l.max()) == l[...,None]-1).astype(int) #one-hot encode
         #         for k in range(lstack.shape[-1]):
         #            #print(k)
         #             l,_ = crf_refine(lstack[:,:,k], im.numpy(), nclasses = NCLASSES, theta_col=40, theta_spat=1, compat=100)
         #             if MEDIAN_FILTER_VALUE>1:
         #                 lstack[:,:,k] = np.round(median(l, disk(MEDIAN_FILTER_VALUE))).astype(np.uint8)
         #             else:
         #                 lstack[:,:,k] = np.round(l).astype(np.uint8)
         #             lab = np.argmax(lstack, -1)
         #     else:
         #             lab = np.argmax(lab, -1)
