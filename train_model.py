# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021-22, Marda Science LLC
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
import json
from tkinter import filedialog
from tkinter import *
from random import shuffle
import pandas as pd

###############################################################
## VARIABLES
###############################################################

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "./",title = "Select directory of data files")
data_path = root.filename
print(data_path)
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

## NCLASSES>=2
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
    from tensorflow.python.client import device_lib
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)

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
    from tensorflow.python.client import device_lib
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)

### mixed precision
from tensorflow.keras import mixed_precision
try:
    mixed_precision.set_global_policy('mixed_float16')
except:
    mixed_precision.experimental.set_policy('mixed_float16')

# tf.debugging.set_log_device_placement(True)

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


###############################################################
### main
###############################################################

##########################################
##### set up defs
#######################################

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

    return image, label#, file


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
    #image, label, file = tf.py_function(func=load_npz, inp=[example], Tout=[tf.float32, tf.uint8, tf.string])
    image, label = tf.py_function(func=load_npz, inp=[example], Tout=[tf.float32, tf.uint8])

    if NCLASSES==2:
        label = tf.expand_dims(label,-1)

    return image, label


#-----------------------------------
def plotcomp_n_metrics(ds,model,NCLASSES, DOPLOT, test_samples_fig, subset,num_batches=20):

    class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477',
                            '#66AA00','#B82E2E', '#316395','#0d0887', '#46039f', '#7201a8',
                            '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']

    class_label_colormap = class_label_colormap[:NCLASSES]

    IOUc = []; Dc=[]; Kc = []
    OA = []; MIOU = []; FWIOU = []
    F1 = []; P =[]; R = []; MCC=[]
    counter = 0
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

            kl = tf.keras.losses.KLDivergence() #initiate object

            est_label = np.argmax(est_label.squeeze(),axis=-1) #argmax to flatten()

            #one-hot encode
            nx,ny = est_label.shape
            lstack = np.zeros((nx,ny,NCLASSES))
            lstack[:,:,:NCLASSES+1] = (np.arange(NCLASSES) == 1+est_label[...,None]-1).astype(int)

            #compute on one-hot encoded integer tensors
            kld = kl(tf.squeeze(lbl), lstack).numpy()

            img = rescale_array(np.array(img), 0, 1) ##.numpy()

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
                    plt.imshow(img)#, cmap='gray')

                plt.imshow(color_label, alpha=0.5)#, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES-1)

                plt.axis('off')

                plt.subplot(222)
                if np.ndim(img)>=3:
                    plt.imshow(img[:,:,0], cmap='gray')
                else:
                    plt.imshow(img)#, cmap='gray')

                plt.imshow(color_estlabel, alpha=0.5)#, cmap=plt.cm.bwr, vmin=0, vmax=NCLASSES-1)

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

###==========================================================
#-------------------------------------------------

##########################################
##### set up dataset
#######################################

if 'MODE' not in locals():
    MODE = 'all'
    print('MODE not specified in config file. Setting to "all" files')

if MODE=='all':
    print('MODE "all": using all augmented and non-augmented files')
    #  use all files instead
    filenames = tf.io.gfile.glob(data_path+os.sep+ROOT_STRING+'*.npz')

elif MODE=='noaug':
    print('MODE "noaug": using non-augmented files')
    # use non-augmented files instead
    filenames = tf.io.gfile.glob(data_path+os.sep+ROOT_STRING+'*noaug*.npz')
    if len(filenames)==0:
        filenames = tf.io.gfile.glob(data_path+os.sep+ROOT_STRING+'*_noaug*.npz')

else:
    print('MODE "aug": using augmented files')
    filenames = tf.io.gfile.glob(data_path+os.sep+ROOT_STRING+'*aug*.npz')
    if len(filenames)==0:
        filenames = tf.io.gfile.glob(data_path+os.sep+ROOT_STRING+'*_aug*.npz')

try:
    dir_path = os.path.dirname(os.getcwd())
    os.mkdir(dir_path+os.sep+'weights')
except:
    pass # weights direc already exists

#----------------------------------------------------------

shuffle(filenames) ##shuffle here

list_ds = tf.data.Dataset.list_files(filenames, shuffle=False) ##dont shuffle here

val_size = int(len(filenames) * VALIDATION_SPLIT)

validation_steps = val_size // BATCH_SIZE
steps_per_epoch =  int(len(filenames) * 1-VALIDATION_SPLIT) // BATCH_SIZE

print(steps_per_epoch)
print(validation_steps)

train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

train_files = []
for i in train_ds:
    train_files.append(i.numpy().decode().split(os.sep)[-1])

val_files = []
for i in val_ds:
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

train_ds = train_ds.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
train_ds = train_ds.repeat()
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU (and possible with distributed gpus)
train_ds = train_ds.prefetch(AUTO) #


val_ds = val_ds.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
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

        if MODEL =='resunet':
            model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                            FILTERS,
                            nclasses=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
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
                            nclasses=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
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
                        num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
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
                        num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
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
                        num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                        dropout_type=DROPOUT_TYPE,
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                        filters=FILTERS,
                        num_layers=4,
                        strides=(1,1))

        else:
            print("Model must be one of 'unet', 'resunet', or 'satunet'")
            sys.exit(2)

else:

    if MODEL =='resunet':
        model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,#0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                        dropout_type=DROPOUT_TYPE,#"standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING
                        )
    elif MODEL=='unet':
        model =  custom_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,#0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                        dropout_type=DROPOUT_TYPE,#"standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                        )

    elif MODEL =='simple_resunet':

        model = simple_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
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
                    num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1,1))

    elif MODEL=='satunet':
        model = custom_satunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1,1))

    else:
        print("Model must be one of 'unet', 'resunet', or 'satunet'")
        sys.exit(2)

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

        model.compile(optimizer = 'adam', loss =weighted_dice_coef_loss(NCLASSES,class_weights), metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)])

    else:

        model.compile(optimizer = 'adam', loss =dice_coef_loss(NCLASSES), metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)])

else:

    if LOSS=='hinge':
        model.compile(optimizer = 'adam', loss =tf.keras.losses.CategoricalHinge(), metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)]) #, steps_per_execution=2, jit_compile=True
    elif LOSS.startswith('cat'):
        model.compile(optimizer = 'adam', loss =tf.keras.losses.CategoricalCrossentropy(), metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)])
    elif LOSS.startswith('k'):
        model.compile(optimizer = 'adam', loss =tf.keras.losses.KLDivergence(), metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)])

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
    plot_seg_history_iou(history, hist_fig)

    plt.close('all')
    K.clear_session()

    model.save(weights.replace('.h5','_fullmodel.h5'))

    np.savez_compressed(weights.replace('.h5','_model_history.npz'),**history.history)

else:
    try:
        model = tf.keras.models.load_model(weights.replace('.h5','_fullmodel.h5'))
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

print('loss={loss:0.4f}, Mean IOU={mean_iou:0.4f}, Mean Dice={mean_dice:0.4f}'.format(loss=scores[0], mean_iou=scores[1], mean_dice=scores[2]))

# # # ##########################################################
IOUc, Dc, Kc, OA, MIOU, FWIOU, MCC = plotcomp_n_metrics(
                                val_ds,model,NCLASSES,DOPLOT,test_samples_fig,'val')
print('Mean of mean IoUs (validation subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(IOUc)))
print('Mean of mean IoUs, confusion matrix (validation subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(MIOU)))
print('Mean of mean frequency weighted IoUs, confusion matrix (validation subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(FWIOU)))
print('Mean of Matthews Correlation Coefficients (validation subset)={mean_dice:0.3f}'.format(mean_dice=np.mean(MCC)))
print('Mean of mean Dice scores (validation subset)={mean_dice:0.3f}'.format(mean_dice=np.mean(Dc)))
print('Mean of mean KLD scores (validation subset)={mean_kld:0.3f}'.format(mean_kld=np.mean(Kc)))


IOUc, Dc, Kc, OA, MIOU, FWIOU, MCC = plotcomp_n_metrics(
                                train_ds,model,NCLASSES,DOPLOT,test_samples_fig,'train')
print('Mean of mean IoUs (train subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(IOUc)))
print('Mean of mean IoUs, confusion matrix (train subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(MIOU)))
print('Mean of mean frequency weighted IoUs, confusion matrix (train subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(FWIOU)))
print('Mean of Matthews Correlation Coefficients (train subset)={mean_dice:0.3f}'.format(mean_dice=np.mean(MCC)))
print('Mean of mean Dice scores (train subset)={mean_dice:0.3f}'.format(mean_dice=np.mean(Dc)))
print('Mean of mean KLD scores (train subset)={mean_kld:0.3f}'.format(mean_kld=np.mean(Kc)))

## uncomment to see comparison plot of validation metrics

# plt.figure(figsize=(10,10))
# plt.subplots_adjust(hspace=0.3, wspace=0.3)
# plt.subplot(321)
# plt.plot(IOUc,MIOU,'.')
# plt.xlabel('mean IOU'); plt.ylabel('mean IOU (Confusion Matrix)')

# plt.subplot(322)
# plt.plot(IOUc,FWIOU,'.')
# plt.xlabel('mean IOU'); plt.ylabel('mean Frequency-weighted IOU (Confusion Matrix)')

# plt.subplot(323)
# plt.plot(IOUc,MCC,'.')
# plt.xlabel('mean IOU'); plt.ylabel('Matthews Correlation Coefficient (Confusion Matrix)')

# plt.subplot(324)
# plt.plot(IOUc,Dc,'.')
# plt.xlabel('mean IOU'); plt.ylabel('mean Dice')

# plt.subplot(325)
# plt.plot(IOUc,Kc,'.')
# plt.xlabel('mean IOU'); plt.ylabel('mean K-L divergence')

# plt.savefig('tmp2.png', dpi=200); plt.close()
