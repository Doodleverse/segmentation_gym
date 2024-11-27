# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021-24, Marda Science LLC
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

profile = 'meta' # meta + predseg
# profile = 'minimal' # predseg
# profile = 'full' # meta + predseg + overlay + probs
##===================================================
#=======================================================
# Import the architectures for following models from doodleverse_utils
# 1. custom_resunet
# 2. custom_unet
# 3. simple_resunet
# 4. simple_unet
# 5. satunet
# 6. custom_resunet
# 7. custom_satunet
# 8. segformer (pre-trained)

def get_model():

    if MODEL =='resunet':
        model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                        dropout_type=DROPOUT_TYPE,
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
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

    elif MODEL=='segformer':
        id2label = {}
        for k in range(NCLASSES):
            id2label[k]=str(k)
        model = segformer(id2label,num_classes=NCLASSES)
        # model.compile(optimizer='adam')

    else:
        print("Model must be one of 'unet', 'resunet', 'segformer', or 'satunet'")
        sys.exit(2)

    return model


if __name__ == "__main__":

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
    # root.filename =  filedialog.askopenfilename(initialdir = sample_direc, title = "Select FIRST model (.keras) or weights (.h5) file",filetypes = (("keras model file","*.keras"),("h5 files","*.h5*")))
    root.filename =  filedialog.askopenfilename(initialdir = sample_direc, title = "Select FIRST model weights (.h5) file",filetypes = (("h5 weights file","*.h5"),("all files","*.**")))

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
            # root.filename =  filedialog.askopenfilename(title = "Select model (.keras) or weights (.h5) file",filetypes = (("keras model file","*.keras"),("h5 files","*.h5*")))
            # root.filename =  filedialog.askopenfilename(title = "Select model weights (.h5) file",filetypes = (("h5 files","*.h5*")))
            root.filename =  filedialog.askopenfilename(initialdir = os.path.dirname(weights), title = "Select NEXT model weights (.h5) file",filetypes = (("h5 weights file","*.h5"),("all files","*.**")))

            weights = root.filename
            root.withdraw()
            W.append(weights)
            print(weights)

    # For each set of weights in W load them in
    M= []; C=[]; T = []
    for counter,weights in enumerate(W):

        # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
        # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
        if 'h5' in weights:
            configfile = weights.replace('_fullmodel.h5','.json').replace('weights', 'config')
        else:
            configfile = weights.replace('.keras','_fullmodel.h5').replace('_fullmodel.h5','.json').replace('weights', 'config')

        if os.path.exists(configfile):

            with open(configfile) as f:
                config = json.load(f)
        else:
            # Turn the .h5 file into a json so that the data can be loaded into dynamic variables        
            if 'h5' in weights:
                configfile = weights.replace('.h5','.json').replace('weights', 'config')
            else:
                configfile = weights.replace('.keras','.h5').replace('.h5','.json').replace('weights', 'config')

            if os.path.exists(configfile):
                with open(configfile) as f:
                    config = json.load(f)
            else:
                configfile = weights.replace('_fullmodel_model.keras','.h5').replace('.h5','.json').replace('weights', 'config')
                if os.path.exists(configfile):
                    with open(configfile) as f:
                        config = json.load(f)

        # Dynamically creates all variables from config dict.
        # For example configs's {'TARGET_SIZE': [768, 768]} will be created as TARGET_SIZE=[768, 768]
        # This is how the program is able to use variables that have never been explicitly defined
        for k in config.keys():
            exec(k+'=config["'+k+'"]')

        if counter==0:

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


        from doodleverse_utils.imports import *
        from doodleverse_utils.model_imports import *
        from doodleverse_utils.prediction_imports import *

        #---------------------------------------------------

        # Get the selected model based on the weights file's MODEL key provided
        # create the model with the data loaded in from the weights file
        print('.....................................')
        print('Creating and compiling model {}...'.format(counter))

        # if 'h5' in weights:
        model = get_model()
        try:
            model.load_weights(weights.replace('.h5','_fullmodel.h5'))
        except:
            model.load_weights(weights)
        # else:
        #     model = tf.keras.models.load_model(weights)

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
        print("TESTTIMEAUG not found in config file(s). Setting to False")
        TESTTIMEAUG = False
    if not 'WRITE_MODELMETADATA' in locals():
        print("WRITE_MODELMETADATA not found in config file(s). Setting to False")
        WRITE_MODELMETADATA = False
    if not 'OTSU_THRESHOLD' in locals():
        print("OTSU_THRESHOLD not found in config file(s). Setting to False")
        OTSU_THRESHOLD = False


    ## # Import do_seg() from doodleverse_utils to perform the segmentation on the images
    for f in tqdm(sample_filenames):
        try:
            do_seg(f, M, metadatadict, MODEL, sample_direc,NCLASSES,N_DATA_BANDS,TARGET_SIZE,TESTTIMEAUG, WRITE_MODELMETADATA,OTSU_THRESHOLD, profile)
        except:
            print("{} failed. Check config file, and check the path provided contains valid imagery".format(f))
