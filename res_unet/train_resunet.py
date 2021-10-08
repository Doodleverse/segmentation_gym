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
USE_GPU = True

if USE_GPU == True:
   ##use the first available GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
from tkinter import filedialog
from tkinter import *
from random import shuffle

###############################################################
## VARIABLES
###############################################################

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/data",title = "Select config file",filetypes = (("config files","*.json"),("all files","*.*")))
configfile = root.filename
print(configfile)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/samples",title = "Select directory of data files")
data_path = root.filename
print(data_path)
root.withdraw()

weights = configfile.replace('.json','.h5').replace('config', 'weights')

try:
    os.mkdir(os.path.dirname(weights))
except:
    pass

#---------------------------------------------------
with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')

from imports import *
#---------------------------------------------------

trainsamples_fig = weights.replace('.h5','_train_sample_batch.png').replace('weights', 'data')
valsamples_fig = weights.replace('.h5','_val_sample_batch.png').replace('weights', 'data')

hist_fig = weights.replace('.h5','_trainhist_'+str(BATCH_SIZE)+'.png').replace('weights', 'data')

try:
    direc = os.path.dirname(hist_fig)
    print("Making new directory for example model outputs: %s"% (direc))
    os.mkdir(direc)
except:
    pass

test_samples_fig =  weights.replace('.h5','_val.png').replace('weights', 'data')

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


#-----------------------------------
def load_npz(example):
    if N_DATA_BANDS==4:
        with np.load(example.numpy()) as data:
            image = data['arr_0'].astype('uint8')
            image = standardize(image)
            nir = data['arr_1'].astype('uint8')
            nir = standardize(nir)
            label = data['arr_2'].astype('uint8')
            image = tf.stack([image, nir], axis=-1)
        # if USE_LOCATION:
        #     gx,gy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        #     loc = np.sqrt(gx**2 + gy**2)
        #     loc /= loc.max()
        #     loc = (255*loc).astype('uint8')
        #     image = np.dstack((image, loc))
        #
        #     mx = np.max(image)
        #     m = np.min(image)
        #     tmp = rescale(loc, m, mx)
        #     image = tf.stack([image[:,:,0], image[:,:,1], image[:,:,2], nir, tmp], axis=-1)
        #     image = tf.cast(image, 'float32')

        return image, nir,label
    else:
        with np.load(example.numpy()) as data:
            image = data['arr_0'].astype('uint8')
            image = standardize(image)
            label = data['arr_1'].astype('uint8')
        # if USE_LOCATION:
        #     gx,gy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        #     loc = np.sqrt(gx**2 + gy**2)
        #     loc /= loc.max()
        #     loc = (255*loc).astype('uint8')
        #     image = np.dstack((image, loc))
        #     image = standardize(image)
        #
        #     mx = np.max(image)
        #     m = np.min(image)
        #     tmp = rescale(loc, m, mx)
        #     image = tf.stack([image[:,:,0], image[:,:,1], image[:,:,2], tmp], axis=-1)
        #     image = tf.cast(image, 'float32')

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
        nir = tf.cast(nir, tf.float32)#/ 255.0
    else:
        image, label = tf.py_function(func=load_npz, inp=[example], Tout=[tf.float32, tf.uint8])

    # image = tf.cast(image, tf.float32)#/ 255.0
    # label = tf.cast(label, tf.uint8)

    if N_DATA_BANDS==4:
        image = tf.concat([image, tf.expand_dims(nir,-1)],-1)

    if NCLASSES==1:
        label = tf.expand_dims(label,-1)

    #image = tf.image.per_image_standardization(image)

    if NCLASSES>1:
        if N_DATA_BANDS>1:
            return tf.squeeze(image), tf.squeeze(label)
        else:
            return image, label
    else:
        return image, label

###############################################################
### main
###############################################################
if USE_GPU == True:
    print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#-------------------------------------------------

filenames = tf.io.gfile.glob(data_path+os.sep+ROOT_STRING+'*.npz')
shuffle(filenames)

list_ds = tf.data.Dataset.list_files(filenames, shuffle=False)

val_size = int(len(filenames) * VALIDATION_SPLIT)

validation_steps = val_size // BATCH_SIZE
steps_per_epoch =  int(len(filenames) * 1-VALIDATION_SPLIT) // BATCH_SIZE

print(steps_per_epoch)
print(validation_steps)

train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
train_ds = train_ds.repeat()
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
train_ds = train_ds.prefetch(AUTO) #

val_ds = val_ds.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
val_ds = val_ds.repeat()
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
val_ds = val_ds.prefetch(AUTO) #


# if DO_TRAIN:
#     # if N_DATA_BANDS<=3:
#     for imgs,lbls in train_ds.take(10):
#         print(imgs.shape)
#         print(lbls.shape)

# plt.figure(figsize=(16,16))
# for imgs,lbls in train_ds.take(100):
#   #print(lbls)
#   for count,(im,lab) in enumerate(zip(imgs, lbls)):
#      plt.subplot(int(BATCH_SIZE+1/2),2,count+1)
#      plt.imshow(im)
#      if NCLASSES==1:
#          plt.imshow(lab, cmap='gray', alpha=0.5, vmin=0, vmax=NCLASSES)
#      else:
#          lab = np.argmax(lab,-1)
#          plt.imshow(lab, cmap='bwr', alpha=0.5, vmin=0, vmax=NCLASSES)
#
#      plt.axis('off')
#      print(np.unique(lab))
#      plt.axis('off')
#      plt.close('all')


print('.....................................')
print('Creating and compiling model ...')

if NCLASSES==1:
    # if USE_LOCATION:
    #     model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS+1), BATCH_SIZE, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))
    # else:
    model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), BATCH_SIZE, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))
else:
    # if USE_LOCATION:
    #     model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS+1), BATCH_SIZE, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))
    # else:
    model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), BATCH_SIZE, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))

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
print('Evaluating model on entire validation set ...')
# # testing
scores = model.evaluate(val_ds, steps=validation_steps)

print('loss={loss:0.4f}, Mean IOU={mean_iou:0.4f}, Mean Dice={mean_dice:0.4f}'.format(loss=scores[0], mean_iou=scores[1], mean_dice=scores[2]))

# # # ##########################################################

IOUc = []

counter = 0
for i,l in val_ds.take(10):

    for img,lbl in zip(i,l):
        # print(img.shape)

        # img = tf.image.per_image_standardization(img)
        # if USE_LOCATION:
        #     img = standardize(img)
        #     mx = np.max(img)
        #     m = np.min(img)
        #     tmp = rescale(loc, m, mx)
        #     img = tf.stack([img[:,:,0], img[:,:,1], img[:,:,2], tmp], axis=-1)
        # else:
        #     img = standardize(img)
        img2 = standardize(img)

        est_label = model.predict(tf.expand_dims(img2, 0) , batch_size=1).squeeze()

        if NCLASSES==1:
            est_label[est_label<.5] = 0
            est_label[est_label>.5] = 1
        else:
            est_label = np.argmax(est_label, -1)

        if NCLASSES==1:
            lbl = lbl.numpy().squeeze()
        else:
            lbl = np.argmax(lbl.numpy(), -1)

        iouscore = iou(lbl, est_label, NCLASSES+1)

        img = rescale(img.numpy(), 0, 1)

        if DOPLOT:
            plt.subplot(221)
            if np.ndim(img)>=3:
                plt.imshow(img[:,:,0], cmap='gray')
            else:
                plt.imshow(img)#, cmap='gray')
            if NCLASSES==1:
                plt.imshow(lbl, alpha=0.1, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES)
            else:
                plt.imshow(lbl, alpha=0.1, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES-1)

            plt.axis('off')

            plt.subplot(222)
            if np.ndim(img)>=3:
                plt.imshow(img[:,:,0], cmap='gray')
            else:
                plt.imshow(img)#, cmap='gray')
            if NCLASSES==1:
                plt.imshow(est_label, alpha=0.1, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES)
            else:
                plt.imshow(est_label, alpha=0.1, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES-1)

            plt.axis('off')
            plt.title('iou = '+str(iouscore)[:5], fontsize=6)
            IOUc.append(iouscore)

            plt.savefig(test_samples_fig.replace('_val.png', '_val_'+str(counter)+'.png'),
                    dpi=200, bbox_inches='tight')
            plt.close('all')
        counter += 1

print('Mean IoU (validation subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(IOUc)))


##### training subset
IOUc = []

counter = 0
for i,l in train_ds.take(10):

    for img,lbl in zip(i,l):

        img2 = standardize(img)

        est_label = model.predict(tf.expand_dims(img2, 0) , batch_size=1).squeeze()

        if NCLASSES==1:
            est_label[est_label<.5] = 0
            est_label[est_label>.5] = 1
        else:
            est_label = np.argmax(est_label, -1)

        if NCLASSES==1:
            lbl = lbl.numpy().squeeze()
        else:
            lbl = np.argmax(lbl.numpy(), -1)

        iouscore = iou(lbl, est_label, NCLASSES+1)

        img = rescale(img.numpy(), 0, 1)

        if DOPLOT:
            plt.subplot(221)
            if np.ndim(img)>=3:
                plt.imshow(img[:,:,0], cmap='gray')
            else:
                plt.imshow(img)#, cmap='gray')
            if NCLASSES==1:
                plt.imshow(lbl, alpha=0.1, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES)
            else:
                plt.imshow(lbl, alpha=0.1, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES-1)

            plt.axis('off')

            plt.subplot(222)
            if np.ndim(img)>=3:
                plt.imshow(img[:,:,0], cmap='gray')
            else:
                plt.imshow(img)#, cmap='gray')
            if NCLASSES==1:
                plt.imshow(est_label, alpha=0.1, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES)
            else:
                plt.imshow(est_label, alpha=0.1, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES-1)

            plt.axis('off')
            plt.title('iou = '+str(iouscore)[:5], fontsize=6)
            IOUc.append(iouscore)

            plt.savefig(test_samples_fig.replace('_val.png', '_train_'+str(counter)+'.png'),
                    dpi=200, bbox_inches='tight')
            plt.close('all')
        counter += 1

print('Mean IoU (train subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(IOUc)))
