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
import matplotlib.pyplot as plt

###############################################################
## VARIABLES
###############################################################

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

weights = configfile.replace('.json','.h5')


with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')

from imports import *


trainsamples_fig = weights.replace('.h5','_train_sample_batch.png').replace('weights', 'examples')
valsamples_fig = weights.replace('.h5','_val_sample_batch.png').replace('weights', 'examples')

hist_fig = weights.replace('.h5','_trainhist_'+str(BATCH_SIZE)+'.png').replace('weights', 'examples')

test_samples_fig =  weights.replace('.h5','_val.png').replace('weights', 'examples')


###############################################################
### DATA FUNCTIONS
###############################################################

@tf.autograph.experimental.do_not_convert
#-----------------------------------
def read_seg_tfrecord_binary(example):
    """
    "read_seg_tfrecord_binary(example)"
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

    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "label": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=N_DATA_BANDS)
    image = tf.cast(image, tf.float32)/ 255.0
    image = tf.reshape(image, [TARGET_SIZE[0],TARGET_SIZE[1], N_DATA_BANDS])
    #print(image.shape)

    label = tf.image.decode_jpeg(example['label'], channels=1)[:,:,0]>0
    label = tf.cast(label, tf.uint8)#/ 255.0
    label = tf.reshape(label, [TARGET_SIZE[0],TARGET_SIZE[1], 1])

    #print(label.shape)
    image = tf.reshape(image, (image.shape[0], image.shape[1], image.shape[2]))

    return image, label


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

    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "label": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=N_DATA_BANDS)
    image = tf.cast(image, tf.float32)/ 255.0
    image = tf.reshape(image, [TARGET_SIZE[0],TARGET_SIZE[1], N_DATA_BANDS])
    # print(image.shape)

    label = tf.image.decode_jpeg(example['label'], channels=1)
    label = tf.cast(label, tf.uint8)#/ 255.0
    label = tf.reshape(label, [TARGET_SIZE[0],TARGET_SIZE[1], 1])
    # print(label.shape)

    label = tf.one_hot(tf.cast(label, tf.uint8), NCLASSES)

    label = tf.squeeze(label)

    image = tf.reshape(image, (image.shape[0], image.shape[1], image.shape[2]))

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
    if NCLASSES==1:
        dataset = dataset.map(read_seg_tfrecord_binary, num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(read_seg_tfrecord_multiclass, num_parallel_calls=AUTO)

    #dataset = dataset.cache() # This dataset fits in RAM
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


#-------------------------------------------------
filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))

print('.....................................')
print('Reading files and making datasets ...')

nb_images = IMS_PER_SHARD * len(filenames)
print(nb_images)

split = int(len(filenames) * VALIDATION_SPLIT)

training_filenames = filenames[split:]
validation_filenames = filenames[:split]

validation_steps = int(nb_images // len(filenames) * len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(nb_images // len(filenames) * len(training_filenames)) // BATCH_SIZE

print(steps_per_epoch)
print(validation_steps)

n = len(training_filenames)*IMS_PER_SHARD
print("training files: %i" % (n))
n = len(validation_filenames)*IMS_PER_SHARD
print("validation files: %i" % (n))

train_ds = get_training_dataset(training_filenames)
val_ds = get_validation_dataset(validation_filenames)

if DO_TRAIN:
    for imgs,lbls in train_ds.take(1):
        print(imgs.shape)
        print(lbls.shape)

    print('.....................................')
    print('Printing examples to file ...')

    plt.figure(figsize=(16,16))
    for imgs,lbls in train_ds.take(1):
      #print(lbls)
      for count,(im,lab) in enumerate(zip(imgs, lbls)):
         plt.subplot(int(BATCH_SIZE/2),2,count+1)
         plt.imshow(im)
         plt.imshow(lab, cmap='gray', alpha=0.5, vmin=0, vmax=1)
         plt.axis('off')
         print(np.unique(lab))
    # plt.show()
    plt.savefig(trainsamples_fig, dpi=200, bbox_inches='tight')
    plt.close('all')

    del imgs, lbls

    plt.figure(figsize=(16,16))
    for imgs,lbls in val_ds.take(1):

      #print(lbls)
      for count,(im,lab) in enumerate(zip(imgs, lbls)):
         plt.subplot(int(BATCH_SIZE/2),2,count+1) #int(BATCH_SIZE/2)
         plt.imshow(im)
         plt.imshow(lab, cmap='gray', alpha=0.5, vmin=0, vmax=1)
         plt.axis('off')
         print(np.unique(lab))
    # plt.show()
    plt.savefig(valsamples_fig, dpi=200, bbox_inches='tight')
    plt.close('all')
    del imgs, lbls


print('.....................................')
print('Creating and compiling model ...')

model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), BATCH_SIZE, NCLASSES)
model.compile(optimizer = 'adam', loss =dice_coef_loss, metrics = [mean_iou, dice_coef])


earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=PATIENCE)

# set checkpoint file
model_checkpoint = ModelCheckpoint(weights, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)


# models are sensitive to specification of learning rate. How do you decide? Answer: you don't. Use a learning rate scheduler

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

callbacks = [model_checkpoint, earlystop, lr_callback]

if DO_TRAIN:
    print('.....................................')
    print('Training model ...')
    history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                          validation_data=val_ds, validation_steps=validation_steps,
                          callbacks=callbacks)

    # Plot training history
    plot_seg_history_iou(history, hist_fig)

    plt.close('all')
    K.clear_session()

else:
    model.load_weights(weights)


# # ##########################################################
# ### evaluate
print('.....................................')
print('Evaluating model ...')
# # testing
scores = model.evaluate(val_ds, steps=validation_steps)

print('loss={loss:0.4f}, Mean IOU={mean_iou:0.4f}, Mean Dice={mean_dice:0.4f}'.format(loss=scores[0], mean_iou=scores[1], mean_dice=scores[2]))

# # # ##########################################################

IOUc = []

counter = 0
for i,l in val_ds.take(10):

    for img,lbl in zip(i,l):
        est_label = model.predict(tf.expand_dims(img, 0) , batch_size=1).squeeze() #tf.expand_dims(i, 0) , batch_size=1).squeeze()
        est_label[est_label<1] = 0

        if DO_CRF_REFINE:
            est_label,_ = crf_refine(est_label.astype(np.uint8), img.numpy().astype(np.uint8), theta_col=100, theta_spat=3)

        if MEDIAN_FILTER_VALUE>1:
            est_label = np.round(median(est_label, disk(MEDIAN_FILTER_VALUE))).astype(np.uint8)
            est_label[est_label<1] = 0
            est_label[est_label>1] = 1

        lbl = lbl.numpy().squeeze()

        iouscore = iou(lbl, est_label, 2)

        if DOPLOT:
            plt.subplot(221)
            plt.imshow(img)
            plt.imshow(lbl, alpha=0.3, cmap=plt.cm.bwr, vmin=0, vmax=1)
            plt.axis('off')

            plt.subplot(222)
            plt.imshow(img)
            plt.imshow(est_label, alpha=0.3, cmap=plt.cm.bwr, vmin=0, vmax=1)
            plt.axis('off')
            plt.title('iou = '+str(iouscore)[:5], fontsize=6)
            IOUc.append(iouscore)

            plt.savefig(test_samples_fig.replace('_val.png', '_val_'+str(counter)+'.png'),
                    dpi=200, bbox_inches='tight')
            plt.close('all')
        counter += 1

print('Mean IoU={mean_iou:0.3f}'.format(mean_iou=np.mean(IOUc)))
