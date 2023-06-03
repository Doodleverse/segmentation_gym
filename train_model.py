# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021-23, Marda Science LLC
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

import sys,os
import json, gc
from tkinter import filedialog
from tkinter import *
from random import shuffle
import pandas as pd
from skimage.transform import resize

###############################################################
## VARIABLES
###############################################################

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "./",title = "Select directory of TRAIN data files")
data_path = root.filename
print(data_path)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = data_path,title = "Select directory of VALIDATION data files")
val_data_path = root.filename
print(val_data_path)
root.withdraw()

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = data_path,title = "Select config file",filetypes = (("config files","*.json"),("all files","*.*")))
configfile = root.filename
print(configfile)
root.withdraw()

configfile = os.path.normpath(configfile)
data_path = os.path.normpath(data_path)

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

if NCLASSES>1:
    pass
else:
    print("NCLASSES must be > 1. Use NCLASSES==2 for binary problems")
    sys.exit(2)

##########################################
##### set up hardware
#######################################
if 'SET_PCI_BUS_ID' not in locals():
    SET_PCI_BUS_ID = False

SET_GPU = str(SET_GPU)

if SET_GPU != '-1':
    USE_GPU = True
    print('Using GPU')
else:
    USE_GPU = False
    print('Warning: using CPU - model training will be slow')

if len(SET_GPU.split(','))>1:
    USE_MULTI_GPU = True
    print('Using multiple GPUs')
else:
    USE_MULTI_GPU = False
    if USE_GPU:
        print('Using single GPU device')
    else:
        print('Using single CPU device')

if USE_GPU == True:

    ## this could be a bad idea - at least on windows, it reorders the gpus in a way you dont want
    if SET_PCI_BUS_ID:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ['CUDA_VISIBLE_DEVICES'] = SET_GPU

    from doodleverse_utils.imports import *
    # from tensorflow.python.client import device_lib
    from tensorflow.keras.callbacks import Callback
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    # tf.config.LogicalDeviceConfiguration(memory_limit=12288)

    if physical_devices:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(physical_devices, 'GPU')
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from doodleverse_utils.imports import *
    # from tensorflow.python.client import device_lib
    from tensorflow.keras.callbacks import Callback
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)

if MODEL!='segformer':
    ### mixed precision
    from tensorflow.keras import mixed_precision
    try:
        mixed_precision.set_global_policy('mixed_float16')
    except:
        mixed_precision.experimental.set_policy('mixed_float16')


for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)
print(tf.config.get_visible_devices())

if USE_MULTI_GPU:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy([p.name.split('/physical_device:')[-1] for p in physical_devices], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print("Number of distributed devices: {}".format(strategy.num_replicas_in_sync))


###############################################################
### main
###############################################################

##########################################
##### set up output variables
#######################################

trainsamples_fig = weights.replace('.h5','_train_sample_batch.png').replace('weights', 'modelOut')
valsamples_fig = weights.replace('.h5','_val_sample_batch.png').replace('weights', 'modelOut')

hist_fig = weights.replace('.h5','_trainhist_'+str(BATCH_SIZE)+'.png').replace('weights', 'modelOut')

try:
    direc = os.path.dirname(hist_fig)
    print("Making new directory for example model outputs: %s"% (direc))
    os.mkdir(direc)
except:
    pass

test_samples_fig =  weights.replace('.h5','_val.png').replace('weights', 'modelOut')


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
    with np.load(example.numpy()) as data:
        image = data['arr_0'].astype('uint8')
        image = standardize(image)
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
    image, label = tf.py_function(func=load_npz, inp=[example], Tout=[tf.float32, tf.uint8])

    if NCLASSES==2:
        label = tf.expand_dims(label,-1)

    return image, label


@tf.autograph.experimental.do_not_convert
#-----------------------------------
def read_seg_dataset_multiclass_segformer(example):
    """
    "read_seg_dataset_multiclass_segformer(example)"
    This function reads an example from a npz file into a single image and label
    INPUTS:
        * dataset example object (filename of npz)
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: N_DATA_BANDS
    OUTPUTS:
        * image [tensor array]
        * class_label [tensor array]
    """
    image, label = tf.py_function(func=load_npz, inp=[example], Tout=[tf.float32, tf.uint8])

    imdim = image.shape[0]
    
    if N_DATA_BANDS==1:
        image = tf.concat([image, image, image], axis=2)

    image = tf.transpose(image, (2, 0, 1))

    if N_DATA_BANDS==1:
        image.set_shape([3, imdim, imdim])
    else:
        image.set_shape([N_DATA_BANDS, imdim, imdim])
    
    label.set_shape([imdim, imdim])

    label = tf.squeeze(tf.argmax(tf.squeeze(label),-1))

    return {"pixel_values": image, "labels": label}


#-----------------------------------
def plotcomp_n_metrics(ds,model,NCLASSES, DOPLOT, test_samples_fig, subset,MODEL,num_batches=20):

    class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477',
                            '#66AA00','#B82E2E', '#316395','#0d0887', '#46039f', '#7201a8',
                            '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']

    class_label_colormap = class_label_colormap[:NCLASSES]

    IOUc = []; Dc=[]; Kc = []
    OA = []; MIOU = []; FWIOU = []
    F1 = []; P =[]; R = []; MCC=[]
    counter = 0

    if MODEL=='segformer':

        for samples in ds.take(num_batches):

            for img,lbl in zip(samples["pixel_values"],samples["labels"]):

                img = tf.transpose(img, (1, 2, 0))

                img = img.numpy() 
                img = standardize(img)

                img = tf.transpose(img, (2, 0, 1))
                img = np.expand_dims(img,axis=0)

                #We use the model to make a prediction on this image
                est_label = model.predict(img).logits

                nR, nC = lbl.shape
                # est_label = scale(est_label, nR, nC)
                est_label = resize(est_label, (1, NCLASSES, nR,nC), preserve_range=True, clip=True)

                imgPredict = tf.math.argmax(est_label, axis=1)[0]

                out = AllMetrics(NCLASSES, imgPredict, lbl)


                OA.append(out['OverallAccuracy'])
                FWIOU.append(out['Frequency_Weighted_Intersection_over_Union'])
                MIOU.append(out['MeanIntersectionOverUnion'])
                F1.append(out['F1Score'])
                R.append(out['Recall'])
                P.append(out['Precision'])
                MCC.append(out['MatthewsCorrelationCoefficient'])

                # #one-hot encode
                lstack = np.zeros((nR,nC,NCLASSES))
                lstack[:,:,:NCLASSES+1] = (np.arange(NCLASSES) == 1+imgPredict.numpy()[...,None]-1).astype(int) 

                lstack_gt = np.zeros((nR,nC,NCLASSES))
                lstack_gt[:,:,:NCLASSES+1] = (np.arange(NCLASSES) == 1+lbl.numpy()[...,None]-1).astype(int) 

                iouscore = mean_iou_np(tf.expand_dims(tf.squeeze(lstack_gt), 0), tf.expand_dims(tf.squeeze(lstack), 0), NCLASSES)

                dicescore = mean_dice_np(tf.expand_dims(tf.squeeze(lstack_gt), 0), tf.expand_dims(tf.squeeze(lstack), 0), NCLASSES)

                kl = tf.keras.losses.KLDivergence() #initiate object
                #compute on one-hot encoded integer tensors
                kld = kl(tf.squeeze(lstack_gt), lstack).numpy()

                IOUc.append(iouscore)
                Dc.append(dicescore)
                Kc.append(kld)

                img = rescale_array(np.array(img), 0, 1) ##.numpy()
                img = np.squeeze(img)
                img = tf.transpose(img, (1, 2, 0))

                color_estlabel = label_to_colors(imgPredict, tf.cast(img[:,:,0]==0,tf.uint8),
                                                alpha=128, colormap=class_label_colormap,
                                                color_class_offset=0, do_alpha=False)

                color_label = label_to_colors(lbl.numpy(), tf.cast(img[:,:,0]==0,tf.uint8),
                                                alpha=128, colormap=class_label_colormap,
                                                color_class_offset=0, do_alpha=False)

                if DOPLOT:
                    plt.subplot(221)
                    if np.ndim(img)>=3:
                        plt.imshow(img[:,:,0], cmap='gray')
                    else:
                        plt.imshow(img)

                    plt.imshow(color_label, alpha=0.5)

                    plt.axis('off')

                    plt.subplot(222)
                    if np.ndim(img)>=3:
                        plt.imshow(img[:,:,0], cmap='gray')
                    else:
                        plt.imshow(img)

                    plt.imshow(color_estlabel, alpha=0.5)

                    plt.axis('off')
                    plt.title('dice = '+str(dicescore)[:5]+', kl = '+str(kld)[:5], fontsize=6)

                    if subset=='val':
                        plt.savefig(test_samples_fig.replace('_val.png', '_val_'+str(counter)+'.png'),
                                dpi=200, bbox_inches='tight')
                    else:
                        plt.savefig(test_samples_fig.replace('_val.png', '_train_'+str(counter)+'.png'),
                                dpi=200, bbox_inches='tight')

                    plt.close('all')
                counter += 1
                K.clear_session()
                del iouscore, dicescore, kld

    else: # models other than segformer

        for i,l in ds.take(num_batches):

            for img,lbl in zip(i,l):

                img = standardize(img)

                try:
                    est_label = model.predict(tf.expand_dims(img, 0) , batch_size=1)
                except:
                    est_label = model.predict(tf.expand_dims(img[:,:,0], 0) , batch_size=1)

                imgPredict = np.argmax(est_label.squeeze(),axis=-1)
                label = np.argmax(tf.squeeze(lbl),axis=-1)

                out = AllMetrics(NCLASSES, imgPredict, label)

                OA.append(out['OverallAccuracy'])
                FWIOU.append(out['Frequency_Weighted_Intersection_over_Union'])
                MIOU.append(out['MeanIntersectionOverUnion'])
                F1.append(out['F1Score'])
                R.append(out['Recall'])
                P.append(out['Precision'])
                MCC.append(out['MatthewsCorrelationCoefficient'])

                iouscore = mean_iou_np(tf.expand_dims(tf.squeeze(lbl), 0), est_label, NCLASSES)

                dicescore = mean_dice_np(tf.expand_dims(tf.squeeze(lbl), 0), est_label, NCLASSES)

                kl = tf.keras.losses.KLDivergence() 

                est_label = np.argmax(est_label.squeeze(),axis=-1) 

                #one-hot encode
                nx,ny = est_label.shape
                lstack = np.zeros((nx,ny,NCLASSES))
                lstack[:,:,:NCLASSES+1] = (np.arange(NCLASSES) == 1+est_label[...,None]-1).astype(int) 

                #compute on one-hot encoded integer tensors
                kld = kl(tf.squeeze(lbl), lstack).numpy()

                img = rescale_array(np.array(img), 0, 1) 

                color_estlabel = label_to_colors(est_label, tf.cast(img[:,:,0]==0,tf.uint8),
                                                alpha=128, colormap=class_label_colormap,
                                                color_class_offset=0, do_alpha=False)

                color_label = label_to_colors(np.argmax(tf.squeeze(lbl).numpy(),axis=-1), tf.cast(img[:,:,0]==0,tf.uint8),
                                                alpha=128, colormap=class_label_colormap,
                                                color_class_offset=0, do_alpha=False)

                if DOPLOT:
                    plt.subplot(221)
                    if np.ndim(img)>=3:
                        plt.imshow(img[:,:,0], cmap='gray')
                    else:
                        plt.imshow(img)

                    plt.imshow(color_label, alpha=0.5)

                    plt.axis('off')

                    plt.subplot(222)
                    if np.ndim(img)>=3:
                        plt.imshow(img[:,:,0], cmap='gray')
                    else:
                        plt.imshow(img)

                    plt.imshow(color_estlabel, alpha=0.5)

                    plt.axis('off')
                    plt.title('dice = '+str(dicescore)[:5]+', kl = '+str(kld)[:5], fontsize=6)
                    IOUc.append(iouscore)
                    Dc.append(dicescore)
                    Kc.append(kld)

                    del iouscore, dicescore, kld

                    if subset=='val':
                        plt.savefig(test_samples_fig.replace('_val.png', '_val_'+str(counter)+'.png'),
                                dpi=200, bbox_inches='tight')
                    else:
                        plt.savefig(test_samples_fig.replace('_val.png', '_train_'+str(counter)+'.png'),
                                dpi=200, bbox_inches='tight')

                    plt.close('all')
                counter += 1
                K.clear_session()


    metrics_table = np.vstack((OA,MIOU,FWIOU)).T
    metrics_per_class = np.hstack((F1,R,P))

    metrics_table = {}
    metrics_table['OverallAccuracy'] = np.array(OA)
    metrics_table['Frequency_Weighted_Intersection_over_Union'] = np.array(FWIOU)
    metrics_table['MeanIntersectionOverUnion'] = np.array(MIOU)
    metrics_table['MatthewsCorrelationCoefficient'] = np.array(MCC)

    df_out1 = pd.DataFrame.from_dict(metrics_table)
    df_out1.to_csv(test_samples_fig.replace('_val.png', '_model_metrics_per_sample_'+subset+'.csv'))

    metrics_per_class = {}
    for k in range(NCLASSES):
        metrics_per_class['F1Score_class{}'.format(k)] = np.array(F1)[:,k]
        metrics_per_class['Recall_class{}'.format(k)] = np.array(R)[:,k]
        metrics_per_class['Precision_class{}'.format(k)] = np.array(P)[:,k]

    df_out2 = pd.DataFrame.from_dict(metrics_per_class)
    df_out2.to_csv(test_samples_fig.replace('_val.png', '_model_metrics_per_sample_per_class_'+subset+'.csv'))

    return IOUc, Dc, Kc, OA, MIOU, FWIOU, MCC

#-----------------------------------
def get_model(for_model_save=False):
    if MODEL =='resunet':
        model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=NCLASSES, 
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                        dropout_type=DROPOUT_TYPE,
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING
                        )
    elif MODEL=='unet':
        model =  custom_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=NCLASSES, 
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                        dropout_type=DROPOUT_TYPE,
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                        )

    elif MODEL =='simple_resunet':

        model = simple_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=NCLASSES,
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1,1))

    elif MODEL=='simple_unet':
        model = simple_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=NCLASSES,
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1,1))

    elif MODEL=='satunet':
        model = custom_satunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=NCLASSES, 
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1,1))

    elif MODEL=='segformer':
        id2label = {}
        for k in range(NCLASSES):
            id2label[k]=str(k)
        model = segformer(id2label,num_classes=NCLASSES)
        if not for_model_save: # if model_save is False (default), model is compiled
            model.compile(optimizer='adam')
    else:
        print("Model must be one of 'unet', 'resunet', 'segformer', or 'satunet'")
        sys.exit(2)

    return model
        
###==========================================================
#-------------------------------------------------

##########################################
##### set up dataset
#######################################

train_filenames = tf.io.gfile.glob(data_path+os.sep+ROOT_STRING+'*.npz')

val_filenames = tf.io.gfile.glob(val_data_path+os.sep+ROOT_STRING+'*.npz')

#----------------------------------------------------------

shuffle(train_filenames) ##shuffle here
shuffle(val_filenames) ##shuffle here

list_ds = tf.data.Dataset.list_files(train_filenames, shuffle=False) ##dont shuffle here

val_list_ds = tf.data.Dataset.list_files(val_filenames, shuffle=False) ##dont shuffle here


validation_steps = len(val_filenames) // BATCH_SIZE
steps_per_epoch =  len(train_filenames) // BATCH_SIZE

train_files = []
for i in list_ds:
    train_files.append(i.numpy().decode().split(os.sep)[-1])

val_files = []
for i in val_list_ds:
    val_files.append(i.numpy().decode().split(os.sep)[-1])

try:
    np.savetxt(weights.replace('.h5','_train_files.txt'), train_files, fmt='%s')
except:
    dir_path = os.path.dirname(os.path.realpath(weights))
    os.mkdir(dir_path)
    np.savetxt(weights.replace('.h5','_train_files.txt'), train_files, fmt='%s')


np.savetxt(weights.replace('.h5','_val_files.txt'), val_files, fmt='%s')

######################
#### set up data throughput pipeline
#####################
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

if 'LOAD_DATA_WITH_CPU' not in locals():
    LOAD_DATA_WITH_CPU = False #default
    print('LOAD_DATA_WITH_CPU not specified in config file. Setting to "False"')

if LOAD_DATA_WITH_CPU:
    with tf.device("CPU"):
        if MODEL=='segformer':
            train_ds = list_ds.map(read_seg_dataset_multiclass_segformer, num_parallel_calls=AUTO)

            val_ds = val_list_ds.map(read_seg_dataset_multiclass_segformer, num_parallel_calls=AUTO)

        else:
            train_ds = list_ds.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
            val_ds = val_list_ds.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
else:
    if MODEL=='segformer':
        train_ds = list_ds.map(read_seg_dataset_multiclass_segformer, num_parallel_calls=AUTO)

        val_ds = val_list_ds.map(read_seg_dataset_multiclass_segformer, num_parallel_calls=AUTO)

    else:
        train_ds = list_ds.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
        val_ds = val_list_ds.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)


train_ds = train_ds.repeat()
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU (and possible with distributed gpus)
train_ds = train_ds.prefetch(AUTO) #

val_ds = val_ds.repeat()
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
val_ds = val_ds.prefetch(AUTO) #

### the following code is for troubleshooting, when do_viz=True
do_viz = False
# do_viz=True

if do_viz == True:
    ## to view examples
    class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477',
                            '#66AA00','#B82E2E', '#316395','#0d0887', '#46039f', '#7201a8',
                            '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']

    class_label_colormap = class_label_colormap[:NCLASSES]

    if MODEL=='segformer':

        for counter, samples in enumerate(train_ds.take(10)):
            sample_image, sample_mask = samples["pixel_values"][0], samples["labels"][0]
            sample_image = tf.transpose(sample_image, (1, 2, 0))
            sample_mask = tf.expand_dims(sample_mask, -1)

            im = sample_image.numpy() #tf.keras.utils.array_to_img(sample_image)
            lab = sample_mask.numpy() #tf.keras.utils.array_to_img(sample_mask)

            if im.shape[-1]>3:
                plt.imshow(im[:,:,:3])
            else:
                plt.imshow(im)
            color_label = label_to_colors(np.squeeze(lab), tf.cast(im[:,:,0]==0,tf.uint8),
                                            alpha=128, colormap=class_label_colormap,
                                            color_class_offset=0, do_alpha=False)

            plt.imshow(color_label,  alpha=0.75, vmin=0, vmax=NCLASSES)

            plt.axis('off')
            plt.savefig('example-{}.png'.format(counter), dpi=300)
            plt.close()

    else:

        for counter, (imgs,lbls) in enumerate(train_ds.take(10)):
            for count,(im,lab) in enumerate(zip(imgs, lbls)):
                print(im.shape)
                if im.shape[-1]>3:
                    plt.imshow(im[:,:,:3])
                else:
                    plt.imshow(im)

                print(lab.shape)
                lab = np.argmax(lab.numpy().squeeze(),-1)
                print(np.unique(lab))

                color_label = label_to_colors(np.squeeze(lab), tf.cast(im[:,:,0]==0,tf.uint8),
                                                alpha=128, colormap=class_label_colormap,
                                                color_class_offset=0, do_alpha=False)

                plt.imshow(color_label,  alpha=0.75, vmin=0, vmax=NCLASSES)

                plt.axis('off')
                #plt.show()
                plt.savefig('example-{}-{}.png'.format(counter,count), dpi=200)
                plt.close()


##===============================================

##########################################
##### set up model
#######################################

print('.....................................')
print('Creating and compiling model ...')
if USE_MULTI_GPU:
    with strategy.scope():
        model = get_model(for_model_save=False)

else: ## single GPU
    model = get_model(for_model_save=False)

if MODEL!='segformer':
    if LOSS=='dice':
        if 'LOSS_WEIGHTS' in locals():
            if LOSS_WEIGHTS is True:

                    print("Computing loss weights per class ...")
                    N = []
                    # compute class-frequency distribution for 30 batches of training images 
                    for _,lbls in train_ds.take(30):
                        for _,lab in zip(_, lbls):
                            lab = np.argmax(lab.numpy().squeeze(),-1).flatten()
                            # make sure bincount is same length as number of classes
                            N.append(np.bincount(lab,minlength=NCLASSES)) #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0]))
                    # mean of class-frequencies
                    class_weights = np.mean(np.vstack(N),axis=0)
                    # inverse weighting
                    class_weights = 1-(class_weights/np.sum(class_weights))

            elif type(LOSS_WEIGHTS) is list:
                    class_weights = np.array(LOSS_WEIGHTS)
                    # inverse weighting
                    class_weights = 1-(class_weights/np.sum(class_weights))
                    print("Model compiled with class weights {}".format(class_weights)) 
            else:
                class_weights = np.ones(NCLASSES)

                model.compile(optimizer = 'adam', loss =weighted_dice_coef_loss(NCLASSES,class_weights), metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)], run_eagerly=True)
        else:
                model.compile(optimizer = 'adam', loss =dice_coef_loss(NCLASSES), metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)], run_eagerly=True)

    else:
            if LOSS=='hinge':
                model.compile(optimizer = 'adam', loss =tf.keras.losses.CategoricalHinge(), metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)], run_eagerly=True) 
            elif LOSS.startswith('cat'):
                model.compile(optimizer = 'adam', loss =tf.keras.losses.CategoricalCrossentropy(), metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)], run_eagerly=True)
            elif LOSS.startswith('k'):
                model.compile(optimizer = 'adam', loss =tf.keras.losses.KLDivergence(), metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)], run_eagerly=True)

#----------------------------------------------------------

##########################################
##### set up callbacks
#######################################

# Open the file
with open(MODEL+'_report.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

if MODEL =='resunet':
    try:
        tf.keras.utils.plot_model(model,to_file="residual_unet_test.png",dpi=200)
    except:
        pass
elif MODEL=='unet':
    try:
        tf.keras.utils.plot_model(model,to_file="unet_test.png",dpi=200)
    except:
        pass

earlystop = EarlyStopping(monitor="val_loss",
                            mode="min", patience=PATIENCE)

# set checkpoint file
model_checkpoint = ModelCheckpoint(weights, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)


# models are sensitive to specification of learning rate. How do you decide? Answer: you don't. Use a learning rate scheduler
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

if 'CLEAR_MEMORY' not in locals():
    CLEAR_MEMORY = False

if CLEAR_MEMORY:

    class ClearMemory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            K.clear_session()

    print("Garbage collection will be perfomed")
    callbacks = [model_checkpoint, earlystop, lr_callback, ClearMemory()]
else:
    print("Garbage collection will NOT be perfomed. To change this behaviour, set CLEAR_MEMORY=True in the config file")
    callbacks = [model_checkpoint, earlystop, lr_callback]


#----------------------------------------------------------
##########################################
##### train!
#######################################

if DO_TRAIN:

    if 'HOT_START' in locals():
        if 'INITIAL_EPOCH' not in locals():
            print("if HOT_START is specified, INITIAL_EPOCH must also be specified in the config file. Exiting ...")
            sys.exit(2)
        model.load_weights(HOT_START)
        print('transfering model weights for hot start ...')
    else:
        if 'INITIAL_EPOCH' not in locals():
            INITIAL_EPOCH=0
            print("INITIAL_EPOCH not specified in the config file. Setting to default of 0 ...")

    print('.....................................')
    print('Training model ...')
    history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                        validation_data=val_ds, validation_steps=validation_steps, initial_epoch=INITIAL_EPOCH,
                        callbacks=callbacks)

    # Plot training history
    plot_seg_history_iou(history, hist_fig, MODEL)

    plt.close('all')
    K.clear_session()

    try:    
        np.savez_compressed(weights.replace('.h5','_model_history.npz'),**history.history)
    except: 
        print("model training history could not be saved")

    # if MODEL=='segformer':
    try:
        model.save_weights(weights.replace('.h5','_fullmodel.h5'))
    except:
        print("fullmodel weights could not be saved")

    # try:
    #     model.save(weights.replace('.h5','_model.keras'), save_format="keras_v3")
    # except:
    #     print("keras format could not be saved")

else:
    # if MODEL!='segformer':
    #     try:
    #         model = tf.keras.models.load_model(weights.replace('.h5','_model.keras'))
    #     except:
    #         model.load_weights(weights)
    # else:
    #     try:
    #         model = tf.keras.models.load_model(weights.replace('.h5','_model.keras'))
    #     except:
    #         model.load_weights(weights)
    # if os.path.exists(weights.replace('.h5','_model.keras')):
    #     model = tf.keras.models.load_model(weights.replace('.h5','_model.keras'))
    #     model.compile('adam',None)

    # if 'h5' in weights:
    model = get_model()
    try:
        model.load_weights(weights.replace('.h5','_fullmodel.h5'))
    except:
        model.load_weights(weights)


# # ##########################################################
##########################################
##### evaluate
#######################################

print('.....................................')
print('Evaluating model on entire validation set ...')
# # testing
scores = model.evaluate(val_ds, steps=validation_steps)

if MODEL!='segformer':
    print('loss={loss:0.4f}, Mean IOU={mean_iou:0.4f}, Mean Dice={mean_dice:0.4f}'.format(loss=scores[0], mean_iou=scores[1], mean_dice=scores[2]))
else:
    print('loss={loss:0.4f}'.format(loss=scores))

# # # ##########################################################
IOUc, Dc, Kc, OA, MIOU, FWIOU, MCC = plotcomp_n_metrics(
                                val_ds,model,NCLASSES,DOPLOT,test_samples_fig,'val', MODEL)
print('Mean of mean IoUs (validation subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(IOUc)))
print('Mean of mean IoUs, confusion matrix (validation subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(MIOU)))
print('Mean of mean frequency weighted IoUs, confusion matrix (validation subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(FWIOU)))
print('Mean of Matthews Correlation Coefficients (validation subset)={mean_dice:0.3f}'.format(mean_dice=np.mean(MCC)))
print('Mean of mean Dice scores (validation subset)={mean_dice:0.3f}'.format(mean_dice=np.mean(Dc)))
print('Mean of mean KLD scores (validation subset)={mean_kld:0.3f}'.format(mean_kld=np.mean(Kc)))


IOUc, Dc, Kc, OA, MIOU, FWIOU, MCC = plotcomp_n_metrics(
                                train_ds,model,NCLASSES,DOPLOT,test_samples_fig,'train', MODEL)
print('Mean of mean IoUs (train subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(IOUc)))
print('Mean of mean IoUs, confusion matrix (train subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(MIOU)))
print('Mean of mean frequency weighted IoUs, confusion matrix (train subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(FWIOU)))
print('Mean of Matthews Correlation Coefficients (train subset)={mean_dice:0.3f}'.format(mean_dice=np.mean(MCC)))
print('Mean of mean Dice scores (train subset)={mean_dice:0.3f}'.format(mean_dice=np.mean(Dc)))
print('Mean of mean KLD scores (train subset)={mean_kld:0.3f}'.format(mean_kld=np.mean(Kc)))

