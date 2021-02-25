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

import os, json, shutil
from tkinter import filedialog
from tkinter import *

#-----------------------------------
def write_seg_records(dataset, tfrecord_dir, root_string):
    """
    "write_seg_records(dataset, tfrecord_dir)"
    This function writes a tf.data.Dataset object to TFRecord shards
    INPUTS:
        * dataset [tf.data.Dataset]
        * tfrecord_dir [string] : path to directory where files will be written
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (files written to disk)
    """
    for shard, (image, label) in enumerate(dataset):
      shard_size = image.numpy().shape[0]
      filename = tfrecord_dir+os.sep+root_string + "{:02d}-{}.tfrec".format(shard, shard_size)

      with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
          example = to_seg_tfrecord(image.numpy()[i],label.numpy()[i])
          out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, shard_size))


#-----------------------------------
def _bytestring_feature(list_of_bytestrings):
    """
    "_bytestring_feature"
    cast inputs into tf dataset 'feature' classes
    INPUTS:
        * list_of_bytestrings
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS: tf.train.Feature example
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

#-----------------------------------
def to_seg_tfrecord(img_bytes, label_bytes):
    """
    "to_seg_tfrecord"
    This function creates a TFRecord example from an image byte string and a label feature
    INPUTS:
        * img_bytes
        * label_bytes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: tf.train.Feature example
    """
    feature = {
      "image": _bytestring_feature([img_bytes]), # one image in the list
      "label": _bytestring_feature([label_bytes]), # one label image in the list
              }
    return tf.train.Example(features=tf.train.Features(feature=feature))


###############################################################
## VARIABLES
###############################################################

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/weights",title = "Select config file",filetypes = (("config files","*.json"),("all files","*.*")))
configfile = root.filename
print(configfile)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/data",title = "Select directory of image files")
imdir = root.filename
print(imdir)
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


with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')

from imports import *

#-----------------------------------
def get_seg_dataset_for_tfrecords(imdir, lab_path, shared_size):
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
    dataset = dataset.map(read_seg_image_and_label)
    dataset = dataset.map(resize_and_crop_seg_image, num_parallel_calls=AUTO)
    dataset = dataset.map(recompress_seg_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(shared_size)
    return dataset


#-----------------------------------
def recompress_seg_image(image, label):
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
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)

    label = tf.cast(label, tf.uint8)
    label = tf.image.encode_jpeg(label, optimize_size=True, chroma_downsampling=False)
    return image, label


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

##========================================================
## AUGMENTATION
##========================================================

n_im = len(glob(imdir+os.sep+'*.jpg'))
print(n_im)

os.mkdir('images/images')
os.mkdir('images/aug_images')

for file in glob(os.getcwd()+os.sep+'images/*.jpg'):
    shutil.move(file,'images/images')

imdir += os.sep+'aug_images'

if USEMASK:
    os.mkdir('masks/masks')
    os.mkdir('masks/aug_masks')
    for file in glob(os.getcwd()+os.sep+'masks/*.jpg'):
        shutil.move(file,'masks/masks')
    lab_path += os.sep+'aug_masks'

else:
    os.mkdir('labels/labels')
    os.mkdir('labels/aug_labels')
    for file in glob(os.getcwd()+os.sep+'labels/*.jpg'):
        shutil.move(file,'labels/labels')
    lab_path += os.sep+'labels'


# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=5,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     fill_mode='nearest',
                     zoom_range=0.05)
image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

img_generator = image_datagen.flow_from_directory(
        imdir,
        target_size=(TARGET_SIZE[0], TARGET_SIZE[1]),
        batch_size=n_im,
        class_mode=None, seed=SEED, shuffle=True)

#the seed must be the same as for the training set to get the same images
mask_generator = mask_datagen.flow_from_directory(
        lab_path,
        target_size=(TARGET_SIZE[0], TARGET_SIZE[1]),
        batch_size=n_im,
        class_mode=None, seed=SEED, shuffle=True)

i = 0
for k in range(3):

    #The following merges the two generators (and their flows) together:
    train_generator = (pair for pair in zip(img_generator, mask_generator))

    #grab a batch of images and label images
    x, y = next(train_generator)

    # wrute them to file and increment the counter
    for im,lab in zip(x,y):
        if NCLASSES==1:
            l = np.round(lab[:,:,0]).astype(np.uint8)
            l[l>0]=1 #null is water
            print(np.unique(l.flatten()))

        imsave(imdir+os.sep+'aug_images/augimage_000000'+str(i)+'.jpg', im.astype(np.uint8))
        imsave(lab_path+os.sep+'aug_labels/augimage_000000'+str(i)+'.jpg', l)
        i += 1

    #save memory
    del x, y, im, lab
    #get a new batch


images = sorted(tf.io.gfile.glob(imdir+os.sep+'*.jpg'))

nb_images=len(images)
print(nb_images)

SHARDS = int(nb_images / IMS_PER_SHARD) + (1 if nb_images % IMS_PER_SHARD != 0 else 0)

shared_size = int(np.ceil(1.0 * nb_images / SHARDS))

dataset = get_seg_dataset_for_tfrecords(imdir, lab_path, shared_size)

counter = 0
# view a batch
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

write_seg_records(dataset, tfrecord_dir, ROOT_STRING)

#
