# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020-22, Marda Science LLC
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
# sys.path.insert(1, 'src')
from tqdm import tqdm

#####################################
#### editable variables
####################################
do_crf = True

USE_GPU = True
# USE_GPU = False

## edit to set GPU number
SET_GPU = 1

## to store interim model outputs and metadata, use True
WRITE_MODELMETADATA = False 

#####################################
#### hardware
####################################

SET_GPU = str(SET_GPU)

if SET_GPU != '-1':
    USE_GPU = True
    print('Using GPU')

if len(SET_GPU.split(','))>1:
    USE_MULTI_GPU = True 
    print('Using multiple GPUs')
else:
    USE_MULTI_GPU = False
    if USE_GPU:
        print('Using single GPU device')
    else:
        print('Using single CPU device')

#suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if USE_GPU == True:
    os.environ['CUDA_VISIBLE_DEVICES'] = SET_GPU

    from doodleverse_utils.prediction_imports import *
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

    from doodleverse_utils.prediction_imports import *
    from tensorflow.python.client import device_lib
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)

### mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
# tf.debugging.set_log_device_placement(True)

for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)
print(tf.config.get_visible_devices())

if USE_MULTI_GPU:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy([p.name.split('/physical_device:')[-1] for p in physical_devices], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print("Number of distributed devices: {}".format(strategy.num_replicas_in_sync))

#####################################
#### session variables
####################################


# from prediction_imports import *
#====================================================
from doodleverse_utils.prediction_imports import *
#---------------------------------------------------

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = './', title = "Select text file listing folders to process",filetypes = (("txt file","*.txt"),("all files","*.*")))
folderlist = root.filename
print(folderlist)
root.withdraw()

with open(folderlist) as f:
    folders = f.readlines()
folders = [f.strip() for f in folders]
print("{} folders to process".format(len(folders)))

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = folders[0], title = "Select FIRST weights file",filetypes = (("weights file","*.h5"),("all files","*.*")))
weights = root.filename
print(weights)
root.withdraw()


W=[]
W.append(weights)

result = 'yes'
while result == 'yes':
    result = messagebox.askquestion("More Weights files?", "More Weights files?", icon='warning')
    if result == 'yes':
        root = Tk()
        root.filename =  filedialog.askopenfilename(title = "Select weights file",filetypes = (("weights file","*.h5"),("all files","*.*")))
        weights = root.filename
        root.withdraw()
        W.append(weights)


M= []; C=[]; T = []
for counter,weights in enumerate(W):

    try:
        configfile = weights.replace('_fullmodel.h5','.json').replace('weights', 'config')
        with open(configfile) as f:
            config = json.load(f)
    except:
        configfile = weights.replace('.h5','.json').replace('weights', 'config')
        with open(configfile) as f:
            config = json.load(f)

    for k in config.keys():
        exec(k+'=config["'+k+'"]')


    #from imports import *
    from doodleverse_utils.imports import *
    #---------------------------------------------------

    #=======================================================

    print('.....................................')
    print('Creating and compiling model {}...'.format(counter))

    if MODEL =='resunet':
        model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,#0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                        dropout_type=DROPOUT_TYPE,#"standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                        )
    elif MODEL=='unet':
        model =  custom_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,#0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                        dropout_type=DROPOUT_TYPE,#"standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                        )

    elif MODEL =='simple_resunet':
        # num_filters = 8 # initial filters
        # model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), num_filters, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))

        model = simple_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    filters=FILTERS,#8,
                    num_layers=4,
                    strides=(1,1))
    #346,564
    elif MODEL=='simple_unet':
        model = simple_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    filters=FILTERS,#8,
                    num_layers=4,
                    strides=(1,1))
    #242,812

    elif MODEL=='satunet':
        #model = sat_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), num_classes=NCLASSES)

        model = custom_satunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    filters=FILTERS,#8,
                    num_layers=4,
                    strides=(1,1))

    else:
        print("Model must be one of 'unet', 'resunet', or 'satunet'")
        sys.exit(2)

    try:
        model = tf.keras.models.load_model(weights)

    except:
        # model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])
        model.compile(optimizer = 'adam', loss = dice_coef_loss, metrics = [mean_iou, dice_coef])

        model.load_weights(weights)

        M.append(model)
        C.append(configfile)
        T.append(MODEL)



### predict
print('.....................................')
print('Using model for prediction on images ...')

for sample_direc in folders:

    metadatadict = {}
    metadatadict['model_weights'] = W
    metadatadict['config_files'] = C
    metadatadict['model_types'] = T

    sample_filenames = sorted(glob(sample_direc+os.sep+'*.*'))
    if sample_filenames[0].split('.')[-1]=='npz':
        sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.npz'))
    else:
        sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))
        if len(sample_filenames)==0:
            # sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.png'))
            sample_filenames = sorted(glob(sample_direc+os.sep+'*.png'))

    print('Number of samples: %i' % (len(sample_filenames)))

    #look for TTA config
    if not 'TESTTIMEAUG' in locals():
        TESTTIMEAUG = False

    for f in tqdm(sample_filenames):
        try:
            do_seg(f, M, metadatadict, sample_direc,NCLASSES,N_DATA_BANDS,TARGET_SIZE,TESTTIMEAUG, WRITE_MODELMETADATA,do_crf)
        except:
            print("{} failed".format(f))