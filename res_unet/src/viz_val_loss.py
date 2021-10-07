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

import json, os
from tkinter import filedialog
from tkinter import *
from random import shuffle

USE_GPU = False #True

if USE_GPU == True:
   ##use the first available GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#utils
#keras functions for early stopping and model weights saving
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf #numerical operations on gpu


root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/data",title = "Select config file",filetypes = (("config files","*.json"),("all files","*.*")))
configfile = root.filename
print(configfile)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/samples",title = "Select directory of TFrecord data files")
data_path = root.filename
print(data_path)
root.withdraw()

weights = configfile.replace('.json','.h5').replace('config', 'weights')


with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')

from imports import *


###############################################################
### DATA FUNCTIONS
###############################################################

@tf.autograph.experimental.do_not_convert
#-----------------------------------
def read_seg_tfrecord_multiclass(example):
    """
    "read_seg_tfrecord_multiclass(example)"
    This function reads an example from a TFrecord file into a single image and label
    This is the "multiclass" version for imagery, where the classes are mapped as follows:
    INPUTS:
        * TFRecord example object
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * class_label [tensor array]
    """

    if N_DATA_BANDS<=3:
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
            "label": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
        }
    else:
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
            "nir": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
            "label": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
        }

    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    if N_DATA_BANDS==4:
        image = tf.image.decode_jpeg(example['image'], channels=3)
    else:
        image = tf.image.decode_jpeg(example['image'], channels=3)

    image = tf.cast(image, tf.float32)/ 255.0
    #image = tf.reshape(image, [TARGET_SIZE[0],TARGET_SIZE[1], 3])
    #print(image.shape)
    #image = tf.reshape(tf.image.rgb_to_grayscale(image), [TARGET_SIZE,TARGET_SIZE, 1])

    if N_DATA_BANDS==4:
        nir = tf.image.decode_jpeg(example['nir'], channels=3)
        nir = tf.cast(nir, tf.float32)/ 255.0
        #nir = tf.reshape(nir, [TARGET_SIZE[0],TARGET_SIZE[1], 3])
        #print(nir.shape)

        image = tf.concat([image, nir],-1)[:,:,:4]
        #print(image.shape)

    #label = tf.image.decode_jpeg(example['label'], channels=1)
    label = tf.image.decode_png(example['label'], channels=0)
    label = tf.cast(label, tf.uint8)#/ 255.0
    #label = tf.reshape(label, [TARGET_SIZE[0],TARGET_SIZE[1], 1])

    # cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*0)
    # label = tf.where(cond,  tf.ones(tf.shape(label),dtype=tf.uint8)*4, label)
    #print(label.shape)

    # if NCLASSES>1:
    #     label = tf.one_hot(tf.cast(label, tf.uint8), NCLASSES+1) # 5 classes (water, surf, wet, dry) + null (0)
    #     label = tf.squeeze(label)

    #image = tf.reshape(image, (image.shape[0], image.shape[1], image.shape[2]))

    #image = tf.image.per_image_standardization(image)
    return image, label

#-----------------------------------
def get_batched_dataset(filenames):
    """
    "get_batched_dataset(filenames)"
    This function defines a workflow for the model to read data from
    tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
    and also formats the imagery properly for model training
    INPUTS:
        * filenames [list]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: BATCH_SIZE, AUTO
    OUTPUTS: tf.data.Dataset object
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = True

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    # if NCLASSES==1:
    #     dataset = dataset.map(read_seg_tfrecord_binary, num_parallel_calls=AUTO)
    # else:
    dataset = dataset.map(read_seg_tfrecord_multiclass, num_parallel_calls=AUTO)

    #dataset = dataset.cache() # if dataset fits in RAM
    dataset = dataset.repeat()
    #dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) #

    return dataset

#-----------------------------------
def get_training_dataset(training_filenames):
    """
    This function will return a batched dataset for model training
    INPUTS: None
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: training_filenames
    OUTPUTS: batched data set object
    """
    return get_batched_dataset(training_filenames)

def get_validation_dataset(validation_filenames):
    """
    This function will return a batched dataset for model training
    INPUTS: None
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: validation_filenames
    OUTPUTS: batched data set object
    """
    return get_batched_dataset(validation_filenames)

#---------------------------------------------------
# learning rate function
def lrfn(epoch):
    """
    lrfn(epoch)
    This function creates a custom piecewise linear-exponential learning rate function for a custom learning rate scheduler. It is linear to a max, then exponentially decays

    * INPUTS: current `epoch` number
    * OPTIONAL INPUTS: None
    * GLOBAL INPUTS:`START_LR`, `MIN_LR`, `MAX_LR`, `RAMPUP_EPOCHS`, `SUSTAIN_EPOCHS`, `EXP_DECAY`
    * OUTPUTS:  the function lr with all arguments passed

    """
    def lr(epoch, START_LR, MIN_LR, MAX_LR, RAMPUP_EPOCHS, SUSTAIN_EPOCHS, EXP_DECAY):
        if epoch < RAMPUP_EPOCHS:
            lr = (MAX_LR - START_LR)/RAMPUP_EPOCHS * epoch + START_LR
        elif epoch < RAMPUP_EPOCHS + SUSTAIN_EPOCHS:
            lr = MAX_LR
        else:
            lr = (MAX_LR - MIN_LR) * EXP_DECAY**(epoch-RAMPUP_EPOCHS-SUSTAIN_EPOCHS) + MIN_LR
        return lr
    return lr(epoch, START_LR, MIN_LR, MAX_LR, RAMPUP_EPOCHS, SUSTAIN_EPOCHS, EXP_DECAY)



#-------------------------------------------------
filenames = tf.io.gfile.glob(data_path+os.sep+'*.tfrec')
# print(filenames[:10])
shuffle(filenames)
# print(filenames[:10])

print('.....................................')
print('Reading files and making datasets ...')

nb_images = IMS_PER_SHARD * len(filenames)
print(nb_images)

split = int(len(filenames) * VALIDATION_SPLIT)

training_filenames = filenames[split:]
validation_filenames = filenames[:split]

n = len(validation_filenames)*IMS_PER_SHARD
print("validation files: %i" % (n))

val_ds = get_validation_dataset(validation_filenames)
validation_steps = int(nb_images // len(filenames) * len(validation_filenames)) // BATCH_SIZE
print(validation_steps)


print('.....................................')
print('Creating and compiling model ...')

if NCLASSES==1:
    model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), BATCH_SIZE, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))
else:
    model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), BATCH_SIZE, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))

model.compile(optimizer = 'adam', loss =dice_coef_loss, metrics = [mean_iou, dice_coef])

model.load_weights(weights)

# scores = model.evaluate(val_ds, steps=validation_steps)
test_samples_fig =  weights.replace('.h5','_val.png').replace('weights', 'examples')

IOUc = []; Dc = []

counter = 0
for i,l in val_ds.take(-1): #10):

    for img,lbl in zip(i,l):
        #print(img.shape)
        est_label = model.predict(tf.expand_dims(img, 0) , batch_size=1).squeeze()
        if NCLASSES==1:
            est_label[est_label<.5] = 0
            est_label[est_label>.5] = 1
        else:
            est_label = np.argmax(est_label, -1)
            # L = []
            # if DO_CRF_REFINE:
            #     for k in range(NCLASSES):
            #         l = est_label[:,:,k]>.5
            #         l = l.astype(np.uint8)
            #         l,_ = crf_refine(l, img.numpy().astype(np.uint8), nclasses=2, theta_col=10, theta_spat=3, compat=10)
            #         L.append(l)
            # est_label = np.argmax(np.dstack(L), -1)

        if MEDIAN_FILTER_VALUE>1:
            est_label = np.round(median(est_label, disk(MEDIAN_FILTER_VALUE))).astype(np.uint8)
            if NCLASSES==1:
                est_label[est_label<0.5] = 0
                est_label[est_label>0.5] = 1

        if NCLASSES==1:
            lbl = lbl.numpy().squeeze()
        else:
            lbl = np.argmax(lbl.numpy(), -1)

        if len(np.unique(lbl))==1:
            if len(np.unique(est_label))==1:
                iouscore = dice = 1.0
            else:
                iouscore = iou(lbl, est_label, NCLASSES+1)
                dice = dice_coef(lbl, tf.cast(est_label, tf.uint8)).numpy()
        else:
            iouscore = iou(lbl, est_label, NCLASSES+1)
            dice = dice_coef(lbl, tf.cast(est_label, tf.uint8)).numpy()

        if dice<=1.0:
            IOUc.append(iouscore)
            Dc.append(dice)

            if dice<.5:
                print('Dice={dice:0.5f}'.format(dice=dice))
                plt.subplot(221)
                plt.imshow(img)
                if NCLASSES==1:
                    plt.imshow(lbl, alpha=0.3, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES)
                else:
                    plt.imshow(lbl, alpha=0.3, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES-1)

                plt.axis('off')

                plt.subplot(222)
                plt.imshow(img)
                if NCLASSES==1:
                    plt.imshow(est_label, alpha=0.3, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES)
                else:
                    plt.imshow(est_label, alpha=0.3, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES-1)

                plt.axis('off')
                plt.title('dice = '+str(dice)[:5], fontsize=6)
                IOUc.append(iouscore)

                plt.savefig(test_samples_fig.replace('_val.png', '_val_'+str(counter)+'.png'),
                        dpi=200, bbox_inches='tight')
                plt.close('all')
        counter+=1

Dc=np.asarray(Dc)
IOUc=np.asarray(IOUc)
print('Dice={dice:0.5f}'.format(dice=np.mean(Dc[Dc>.1])))
print('MeanIOU={iou:0.5f}'.format(iou=np.mean(IOUc[IOUc>.1])))
print('N = %i'%(counter))
