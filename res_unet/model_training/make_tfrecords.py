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

# import os
# USE_GPU = False #True
#
# if USE_GPU == True:
#    ##use the first available GPU
#    os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
# else:
#    ## to use the CPU (not recommended):
#    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import os, json, shutil
from tkinter import filedialog
from tkinter import *


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
    imdir2 = root.filename
    print(imdir2)
    root.withdraw()


root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/data",title = "Select directory of label files")
lab_path = root.filename
print(lab_path)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/data",title = "Select directory to write tfrecord files")
tfrecord_dir = root.filename
print(tfrecord_dir)
root.withdraw()


from imports import *

#-----------------------------------
def read_seg_image_and_label(img_path):
    """
    "read_seg_image_and_label_obx(img_path)"
    This function reads an image and label and decodes both jpegs
    into bytestring arrays.
    This works by parsing out the label image filename from its image pair
    Thre are different rules for non-augmented versus augmented imagery
    INPUTS:
        * img_path [tensor string]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [bytestring]
        * label [bytestring]
    """
    bits = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(bits)

    # have to use this tf.strings.regex_replace utility because img_path is a Tensor object
    if USEMASK:
        lab_path = tf.strings.regex_replace(img_path, "images", "masks")

    else:
        lab_path = tf.strings.regex_replace(img_path, "images", "labels")

    bits = tf.io.read_file(lab_path)
    label = tf.image.decode_jpeg(bits)

    return image, label

#-----------------------------------
def read_seg_image_and_label_4bands(img_path): #, nir_path):
    """
    This function reads an image and label and decodes both jpegs
    into bytestring arrays.
    This works by parsing out the label image filename from its image pair
    Thre are different rules for non-augmented versus augmented imagery
    INPUTS:
        * img_path [tensor string]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [bytestring]
        * label [bytestring]
    """
    bits = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(bits)

    # have to use this tf.strings.regex_replace utility because img_path is a Tensor object
    if USEMASK:
        lab_path = tf.strings.regex_replace(img_path, "images", "masks")

    else:
        lab_path = tf.strings.regex_replace(img_path, "images", "labels")

    bits = tf.io.read_file(lab_path)
    label = tf.image.decode_jpeg(bits)

    nir_path = tf.strings.regex_replace(img_path, "images", "nir")

    bits = tf.io.read_file(nir_path)
    nir = tf.image.decode_jpeg(bits)

    return image, nir, label

#-----------------------------------
def resize_and_crop_seg_image(image, label):
    """
    "resize_and_crop_seg_image_obx"
    This function crops to square and resizes an image and label
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
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

    label = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(label, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(label, [w*th/h, h*th/h])  # if false
                 )
    label = tf.image.crop_to_bounding_box(label, (nw - tw) // 2, (nh - th) // 2, tw, th)

    return image, label


#-----------------------------------
def resize_and_crop_seg_image_4bands(image, nir, label):
    """
    This function crops to square and resizes an image and label
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
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

    label = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(label, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(label, [w*th/h, h*th/h])  # if false
                 )
    label = tf.image.crop_to_bounding_box(label, (nw - tw) // 2, (nh - th) // 2, tw, th)


    nir = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(nir, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(nir, [w*th/h, h*th/h])  # if false
                 )
    nir = tf.image.crop_to_bounding_box(nir, (nw - tw) // 2, (nh - th) // 2, tw, th)

    return image, nir, label


#-----------------------------------
def recompress_seg_image_4bands(image, nir_image, label):
    """
    "recompress_seg_image"
    This function takes an image and label encoded as a byte string
    and recodes as an 8-bit jpeg
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False, quality=100)

    nir_image = tf.cast(nir_image, tf.uint8)
    nir_image = tf.image.encode_jpeg(nir_image, optimize_size=True, chroma_downsampling=False, quality=100)

    label = tf.cast(label, tf.uint8)
    label = tf.image.encode_jpeg(label, optimize_size=True, chroma_downsampling=False, quality=100)
    return image, nir_image, label

#-----------------------------------
def get_seg_dataset_for_tfrecords(imdir, shared_size):
    """
    "get_seg_dataset_for_tfrecords"
    This function reads an image and label and decodes both jpegs
    into bytestring arrays.
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
    dataset = tf.data.Dataset.list_files(imdir+os.sep+'*.jpg', seed=10000) # This also shuffles the images
    if N_DATA_BANDS<=3:
        dataset = dataset.map(read_seg_image_and_label)
        dataset = dataset.map(resize_and_crop_seg_image, num_parallel_calls=AUTO)
        dataset = dataset.map(recompress_seg_image, num_parallel_calls=AUTO)
    elif N_DATA_BANDS==4:
        dataset = dataset.map(read_seg_image_and_label_4bands)
        dataset = dataset.map(resize_and_crop_seg_image_4bands, num_parallel_calls=AUTO)
        dataset = dataset.map(recompress_seg_image_4bands, num_parallel_calls=AUTO)

    dataset = dataset.batch(shared_size)
    return dataset


##========================================================
## AUGMENTATION
##========================================================

n_im = len(glob(imdir+os.sep+'*.jpg'))
print(n_im)

try:
    os.mkdir(imdir+os.sep+'images')
    os.mkdir(imdir+os.sep+'aug_images')
except:
    pass


for file in glob(imdir+os.sep+'*.jpg'):
    shutil.move(file,imdir+os.sep+'images')

#imdir += os.sep+'images'

if USEMASK:
    try:
        os.mkdir(lab_path+os.sep+'masks')
        os.mkdir(lab_path+os.sep+'aug_masks')
    except:
        pass
    for file in glob(lab_path+os.sep+'*.jpg'):
        shutil.move(file,lab_path+os.sep+'masks')
    #lab_path += os.sep+'masks'

else:
    try:
        os.mkdir(lab_path+os.sep+'labels')
        os.mkdir(lab_path+os.sep+'aug_labels')
    except:
        pass
    for file in glob(lab_path+os.sep+'*.jpg'):
        shutil.move(file,lab_path+os.sep+'labels')
    #lab_path += os.sep+'labels'

# print(imdir)
# print(lab_path)

if N_DATA_BANDS==4:
    try:
        os.mkdir(imdir2+os.sep+'nir')
        os.mkdir(imdir2+os.sep+'aug_nir')
    except:
        pass


    for file in glob(imdir2+os.sep+'*.jpg'):
        shutil.move(file,imdir2+os.sep+'nir')


# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=AUG_ROT,
                     width_shift_range=AUG_WIDTHSHIFT,
                     height_shift_range=AUG_HEIGHTSHIFT,
                     fill_mode='nearest',
                     zoom_range=AUG_ZOOM,
                     horizontal_flip=AUG_HFLIP,
                     vertical_flip=AUG_VFLIP)

# AUG_LOOPS = 3

i = 0
for copy in range(AUG_COPIES):
    for k in range(AUG_LOOPS):

        image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

        if N_DATA_BANDS==4:
            image_datagen2 = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

        img_generator = image_datagen.flow_from_directory(
                imdir,
                target_size=(TARGET_SIZE[0], TARGET_SIZE[1]),
                batch_size=int(n_im/AUG_LOOPS),
                class_mode=None, seed=SEED, shuffle=True)

        #the seed must be the same as for the training set to get the same images
        mask_generator = mask_datagen.flow_from_directory(
                lab_path,
                target_size=(TARGET_SIZE[0], TARGET_SIZE[1]),
                batch_size=int(n_im/AUG_LOOPS),
                class_mode=None, seed=SEED, shuffle=True)

        if N_DATA_BANDS==4:
            img_generator2 = image_datagen2.flow_from_directory(
                    imdir2,
                    target_size=(TARGET_SIZE[0], TARGET_SIZE[1]),
                    batch_size=int(n_im/AUG_LOOPS),
                    class_mode=None, seed=SEED, shuffle=True)

        if N_DATA_BANDS<=3:
            #The following merges the two generators (and their flows) together:
            train_generator = (pair for pair in zip(img_generator, mask_generator))
            #grab a batch of images and label images
            x, y = next(train_generator)

            # wrute them to file and increment the counter
            for im,lab in zip(x,y):
                l = np.round(lab[:,:,0]).astype(np.uint8)
                if NCLASSES==1:
                    l[l>0]=1 #null is water
                else:
                    l[l==0]=1
                    l[l>NCLASSES]=NCLASSES

                #print(np.unique(l.flatten()))

                imsave(imdir+os.sep+'aug_images'+os.sep+'augimage_000000'+str(i)+'.jpg', im.astype(np.uint8))
                if USEMASK:
                    imsave(lab_path+os.sep+os.sep+'aug_masks'+os.sep+'augimage_000000'+str(i)+'.jpg', l)
                else:
                    imsave(lab_path+os.sep+os.sep+'aug_labels'+os.sep+'augimage_000000'+str(i)+'.jpg', l)

                i += 1


        elif N_DATA_BANDS==4:
            train_generator = (pair for pair in zip(img_generator, img_generator2, mask_generator))
            x, ii, y = next(train_generator)

            # wrute them to file and increment the counter
            for im,nir,lab in zip(x,ii,y):
                l = np.round(lab[:,:,0]).astype(np.uint8)
                if NCLASSES==1:
                    l[l>0]=1 #null is water
                else:
                    l[l==0]=1
                    l[l>NCLASSES-1]=NCLASSES-1

                imsave(imdir+os.sep+'aug_images'+os.sep+'augimage_000000'+str(i)+'.jpg', im.astype(np.uint8))
                imsave(imdir2+os.sep+'aug_nir'+os.sep+'augimage_000000'+str(i)+'.jpg', nir.astype(np.uint8))

                if USEMASK:
                    imsave(lab_path+os.sep+os.sep+'aug_masks'+os.sep+'augimage_000000'+str(i)+'.jpg', l)
                else:
                    imsave(lab_path+os.sep+os.sep+'aug_labels'+os.sep+'augimage_000000'+str(i)+'.jpg', l)

                i += 1


        #save memory
        del x, y, im, lab
        #get a new batch


##========================================================
## TFRECORD CREATION
##========================================================

images = sorted(tf.io.gfile.glob(imdir+os.sep+'aug_images'+os.sep+'*.jpg'))

nb_images=len(images)
print(nb_images)

SHARDS = int(nb_images / IMS_PER_SHARD) + (1 if nb_images % IMS_PER_SHARD != 0 else 0)

shared_size = int(np.ceil(1.0 * nb_images / SHARDS))

if USEMASK:
    if N_DATA_BANDS<=3:
        dataset = get_seg_dataset_for_tfrecords(imdir+os.sep+'aug_images', shared_size) # [], lab_path+os.sep+'aug_masks'
    elif N_DATA_BANDS==4:
        dataset = get_seg_dataset_for_tfrecords(imdir+os.sep+'aug_images',shared_size) # imdir2+os.sep+'aug_images', lab_path+os.sep+'aug_masks',
else:
    if N_DATA_BANDS<=3:
        dataset = get_seg_dataset_for_tfrecords(imdir+os.sep+'aug_images',  shared_size) #[], lab_path+os.sep+'aug_labels',
    elif N_DATA_BANDS==4:
        dataset = get_seg_dataset_for_tfrecords(imdir.replace('images', 'nir')+os.sep+'aug_nir',  shared_size) #imdir2+os.sep+'aug_images', lab_path+os.sep+'aug_labels',


#visualize some examples
counter = 0
# view a batch of 4

if N_DATA_BANDS<=3:

    for imgs,lbls in dataset.take(1):
      imgs = imgs[:4]
      lbls = lbls[:4]
      for count,(im,lab) in enumerate(zip(imgs,lbls)):
         plt.imshow(tf.image.decode_jpeg(im, channels=3))
         plt.imshow(tf.image.decode_jpeg(lab, channels=1), alpha=0.3, cmap='bwr',vmin=0, vmax=NCLASSES)
         plt.axis('off')
         plt.savefig('ex'+str(counter)+'.png', dpi=200, bbox_inches='tight')
         counter +=1
         plt.close('all')

elif N_DATA_BANDS==4:

    for imgs,nirs,lbls in dataset.take(1):
      imgs = imgs[:4]
      nirs = nirs[:4]
      lbls = lbls[:4]
      for count,(im,ii,lab) in enumerate(zip(imgs,nirs,lbls)):
         plt.imshow(tf.image.decode_jpeg(im, channels=3))
         plt.imshow(tf.image.decode_jpeg(lab, channels=1), alpha=0.3, cmap='bwr',vmin=0, vmax=NCLASSES)
         plt.axis('off')
         plt.savefig('ex'+str(counter)+'.png', dpi=200, bbox_inches='tight')
         plt.close('all')

         plt.imshow(tf.image.decode_jpeg(ii, channels=1), cmap='gray')
         plt.imshow(tf.image.decode_jpeg(lab, channels=1), alpha=0.3, cmap='bwr',vmin=0, vmax=NCLASSES)
         plt.axis('off')
         plt.savefig('ex'+str(counter)+'_band4.png', dpi=200, bbox_inches='tight')
         plt.close('all')
         counter +=1

# write tfrecords
if N_DATA_BANDS<=3:
    write_seg_records(dataset, tfrecord_dir, ROOT_STRING)
elif N_DATA_BANDS==4:
    write_seg_records_4bands(dataset, tfrecord_dir, ROOT_STRING)

#
