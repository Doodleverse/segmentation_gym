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

import sys,os, json
from tqdm import tqdm
from tkinter import filedialog, messagebox
from tkinter import *

#####################################
#### session variables
####################################

#====================================================
#---------------------------------------------------

# Request the folder containing the imagery/npz to segment 
# sample_direc: full path to the directory
root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/samples",title = "Select directory of images (or npzs) to segment")
sample_direc = root.filename
print(sample_direc)
root.withdraw()

# Request the folder containing the model weights
# weights: full path to the weights file location
root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = sample_direc, title = "Select FIRST weights file",filetypes = (("weights file","*.h5"),("all files","*.*")))
weights = root.filename
print(weights)
root.withdraw()

#####################################
#### concatenate models
####################################

# W : list containing all the weight files fill paths
W=[]
W.append(weights)

# Prompt user for more model weights and appends them to the list W that contains all the weights
result = 'yes'
while result == 'yes':
    result = messagebox.askquestion("More Weights files?", "More Weights files?", icon='warning')
    if result == 'yes':
        root = Tk()
        root.filename =  filedialog.askopenfilename(title = "Select weights file",filetypes = (("weights file","*.h5"),("all files","*.*")))
        weights = root.filename
        root.withdraw()
        W.append(weights)

# For each set of weights in W load them in
M= []; C=[]; T = []
for counter,weights in enumerate(W):

    try:
        # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
        # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
        configfile = weights.replace('_fullmodel.h5','.json').replace('weights', 'config')
        with open(configfile) as f:
            config = json.load(f)
    except:
        # Turn the .h5 file into a json so that the data can be loaded into dynamic variables        
        configfile = weights.replace('.h5','.json').replace('weights', 'config')
        with open(configfile) as f:
            config = json.load(f)
    # Dynamically creates all variables from config dict.
    # For example configs's {'TARGET_SIZE': [768, 768]} will be created as TARGET_SIZE=[768, 768]
    # This is how the program is able to use variables that have never been explicitly defined
    for k in config.keys():
        exec(k+'=config["'+k+'"]')


    if counter==0:
        #####################################
        #### hardware
        ####################################

        SET_GPU = str(SET_GPU)

        if SET_GPU != '-1':
            USE_GPU = True
            print('Using GPU')
        else:
            USE_GPU = False
            print('Using CPU')

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


    #from imports import *
    from doodleverse_utils.imports import *
    from doodleverse_utils.model_imports import *

    #---------------------------------------------------

    #=======================================================
    # Import the architectures for following models from doodleverse_utils
    # 1. custom_resunet
    # 2. custom_unet
    # 3. simple_resunet
    # 4. simple_unet
    # 5. satunet
    # 6. custom_resunet
    # 7. custom_satunet

    # Get the selected model based on the weights file's MODEL key provided
    # create the model with the data loaded in from the weights file
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
        # Load in the model from the weights which is the location of the weights file        
        model = tf.keras.models.load_model(weights)

    except:
        # Load the metrics mean_iou, dice_coef from doodleverse_utils
        # Load in the custom loss function from doodleverse_utils        
        model.compile(optimizer = 'adam', loss = dice_coef_loss(NCLASSES))#, metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)])

        model.load_weights(weights)

        M.append(model)
        C.append(configfile)
        T.append(MODEL)

# metadatadict contains the model name (T) the config file(C) and the model weights(W)
metadatadict = {}
metadatadict['model_weights'] = W
metadatadict['config_files'] = C
metadatadict['model_types'] = T

#####################################
#### read images
####################################

# The following lines prepare the data to be predicted
sample_filenames = sorted(glob(sample_direc+os.sep+'*.*'))
if sample_filenames[0].split('.')[-1]=='npz':
    sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.npz'))
else:
    sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))
    if len(sample_filenames)==0:
        sample_filenames = sorted(glob(sample_direc+os.sep+'*.png'))

print('Number of samples: %i' % (len(sample_filenames)))

#####################################
#### run model on each image in a for loop
####################################
### predict
print('.....................................')
print('Using model for prediction on images ...')

#look for TTA config
if not 'TESTTIMEAUG' in locals():
    TESTTIMEAUG = False
#look for do_crf in config
if not 'DO_CRF' in locals():
    DO_CRF = False
if not 'WRITE_MODELMETADATA' in locals():
    WRITE_MODELMETADATA = False
if not 'OTSU_THRESHOLD' in locals():
    OTSU_THRESHOLD = False

# Import do_seg() from doodleverse_utils to perform the segmentation on the images
for f in tqdm(sample_filenames):
    try:
        do_seg(f, M, metadatadict, sample_direc,NCLASSES,N_DATA_BANDS,TARGET_SIZE,TESTTIMEAUG, WRITE_MODELMETADATA,DO_CRF,OTSU_THRESHOLD)
    except:
        print("{} failed".format(f))


