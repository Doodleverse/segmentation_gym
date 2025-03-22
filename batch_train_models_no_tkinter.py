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

import argparse
import sys, os, json
from random import shuffle
import pandas as pd
from skimage.transform import resize
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from doodleverse_utils.imports import *

# python batch_train_models_no_tkinter.py -t /path/to/training -v /path/to/validation -c /path/to/config.json
# -t / --train: Directory of TRAINING data files
# -v / --val: Directory of VALIDATION data files
# -c / --config: Path to the config file

def setup_environment_variables(config):
    """
    Sets up the environment variables for GPU/CPU usage based on the provided configuration.
    
    Parameters:
        config (dict): A dictionary containing configuration options. Expected keys include:
            - 'SET_PCI_BUS_ID' (bool): Optional. If True, sets the CUDA device order to PCI_BUS_ID.
            - 'SET_GPU' (str): Optional. A string representing the GPU device(s) to use. If '-1', CPU is used.
            - 'MODEL' (str): Optional. Model name; used to determine mixed precision setting.
    
    Modifies:
        config (dict): Updates the dictionary with the following keys:
            - 'USE_GPU' (bool): True if GPU is used, False otherwise.
            - 'USE_MULTI_GPU' (bool): True if multiple GPUs are used, False otherwise.
            - 'PHYSICAL_DEVICES' (list): List of physical GPU devices detected.
            - 'DISTRIBUTION_STRATEGY' (tf.distribute.Strategy, optional): The distribution strategy if multiple GPUs are used.
    
    Environment Variables:
        - CUDA_VISIBLE_DEVICES: Set to the value of 'SET_GPU' or '-1' for CPU.
        - CUDA_DEVICE_ORDER: Set to 'PCI_BUS_ID' if 'SET_PCI_BUS_ID' is True.
    
    Prints:
        - Messages indicating whether GPU or CPU is being used.
        - Warnings if 'SET_GPU' is not specified or if CPU is being used.
    """
    # Default for 'SET_PCI_BUS_ID' if not provided
    if 'SET_PCI_BUS_ID' not in config:
        config['SET_PCI_BUS_ID'] = False

    # Ensure SET_GPU is a string; default to CPU if not specified
    config['SET_GPU'] = str(config.get('SET_GPU', '-1'))

    # Determine GPU usage
    if config['SET_GPU'] != '-1':
        config['USE_GPU'] = True
        print('Using GPU')
    else:
        config['USE_GPU'] = False
        print('Warning: using CPU - model training will be slow')
    
    # Determine if multiple GPUs are being used
    if len(config['SET_GPU'].split(',')) > 1:
        config['USE_MULTI_GPU'] = True
        print('Using multiple GPUs')
    else:
        config['USE_MULTI_GPU'] = False
        if config['USE_GPU']:
            print('Using single GPU device')
        else:
            print('Using single CPU device')
    
    # Set environment variables and list physical devices
    if config['USE_GPU']:
        if config['SET_PCI_BUS_ID']:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = config['SET_GPU']
        print(f"CUDA_VISIBLE_DEVICES : {config['SET_GPU']}")
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        print("Physical devices:", physical_devices)
        if physical_devices:
            try:
                tf.config.experimental.set_visible_devices(physical_devices, 'GPU')
            except RuntimeError as e:
                print("RuntimeError while setting visible devices:", e)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        print("Physical devices:", physical_devices)
    
    # Optionally set mixed precision if the model is not "segformer"
    if config.get('MODEL', '').lower() != 'segformer':
        from tensorflow.keras import mixed_precision
        try:
            mixed_precision.set_global_policy('mixed_float16')
        except Exception:
            mixed_precision.experimental.set_policy('mixed_float16')
    
    # Enable memory growth on each GPU device
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print("Error setting memory growth on device", device, ":", e)
    
    print("Visible devices:", tf.config.get_visible_devices())
    
    # If using multiple GPUs, establish a distribution strategy
    if config['USE_MULTI_GPU']:
        strategy = tf.distribute.MirroredStrategy(
            [p.name.split('/physical_device:')[-1] for p in physical_devices],
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
        print("Number of distributed devices: {}".format(strategy.num_replicas_in_sync))
        config['DISTRIBUTION_STRATEGY'] = strategy
    
    # Save the list of physical devices in the config
    config['PHYSICAL_DEVICES'] = physical_devices

def load_config(configfile):
    """
    Loads a configuration file in JSON format, sets the 'USE_GPU' key to False by default,
    and prints the configuration in a pretty-printed format.

    Args:
        configfile (str): The path to the configuration file.

    Returns:
        dict: The loaded configuration dictionary with 'USE_GPU' set to False.
    """
    with open(configfile) as f:
        config = json.load(f)
        config['USE_GPU'] = False  # set the GPU to False by default
        pretty_config = json.dumps(config, indent=4)
        print(f"pretty_config: {pretty_config}")
    return config

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Batch train models using specified data directories and a config file."
    )
    parser.add_argument("-t", "--train", type=str, required=True,
                        help="Directory of TRAINING data files")
    parser.add_argument("-v", "--val", type=str, required=True,
                        help="Directory of VALIDATION data files")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to config file (JSON format)")
    args = parser.parse_args()

    # Normalize paths
    data_path = os.path.normpath(args.train)
    val_data_path = os.path.normpath(args.val)
    configfile = os.path.normpath(args.config)
    weights = configfile.replace('.json', '.h5').replace('config', 'weights')

    print("Training directory:", data_path)
    print("Validation directory:", val_data_path)
    print("Config file:", configfile)

    # Prepare job lists (the original script allowed multiple jobs via an interactive loop;
    # here we process just a single job per run)
    W = [weights]
    C = [configfile]
    D = [data_path]
    V = [val_data_path]

    # Process each job (only one job is supported in this version)
    for counter, (configfile, weights, data_path, val_data_path) in enumerate(zip(C, W, D, V)):
        print('Working on job {} of {}'.format(counter+1, len(C)))
        print("Using config file:", configfile)

        try:
            os.mkdir(os.path.dirname(weights))
        except Exception:
            # Directory may already exist; ignore the error
            pass

        # Load the JSON config file
        with open(configfile) as f:
            config = json.load(f)

        # Create a dictionary to hold all configuration variables
        config_vars = {}
        config_vars = load_config(configfile)
        setup_environment_variables(config_vars)
        # for k in config.keys():
        #     config_vars[k] = config[k]

        if config_vars['NCLASSES'] > 1:
            pass
        else:
            print("NCLASSES must be > 1. Use NCLASSES==2 for binary problems")
            sys.exit(2)

        if config_vars['N_DATA_BANDS'] < 3:
            config_vars['N_DATA_BANDS'] = 3


    ##########################################
    ##### set up output variables
    #######################################

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices detected that were GPU: {physical_devices}")

    if config_vars['USE_GPU']:
        if physical_devices:
            try:
                tf.config.experimental.set_visible_devices(physical_devices, 'GPU')
                print(f"physical_devices after set visible devices : {physical_devices}")
            except RuntimeError as e:
                print(e)

    trainsamples_fig = weights.replace('.h5','_train_sample_batch.png').replace('weights', 'modelOut')
    valsamples_fig = weights.replace('.h5','_val_sample_batch.png').replace('weights', 'modelOut')

    hist_fig = weights.replace('.h5','_trainhist_'+str(config_vars['BATCH_SIZE'])+'.png').replace('weights', 'modelOut')

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
        return lr(epoch, config_vars['START_LR'], config_vars['MIN_LR'], config_vars['MAX_LR'], config_vars['RAMPUP_EPOCHS'], config_vars['SUSTAIN_EPOCHS'], config_vars['EXP_DECAY'])

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

        if config_vars['NCLASSES']==2:
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
        
        image = tf.transpose(image, (2, 0, 1))

        image.set_shape([config_vars['N_DATA_BANDS'], imdim, imdim])
        
        label.set_shape([imdim, imdim])

        label = tf.squeeze(tf.argmax(tf.squeeze(label),-1))

        return {"pixel_values": image, "labels": label}


    #-----------------------------------
    def plotcomp_n_metrics(ds,model,NCLASSES, DOPLOT, test_samples_fig, subset,MODEL,num_batches=10):

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
        if config['MODEL'] =='resunet':
            model =  custom_resunet((config['TARGET_SIZE'][0], config['TARGET_SIZE'][1], config['N_DATA_BANDS']),
                            config['FILTERS'],
                            nclasses=config['NCLASSES'], 
                            kernel_size=(config['KERNEL'],config['KERNEL']),
                            strides=config['STRIDE'],
                            dropout=config['DROPOUT'],
                            dropout_change_per_layer=config['DROPOUT_CHANGE_PER_LAYER'],
                            dropout_type=config['DROPOUT_TYPE'],
                            use_dropout_on_upsampling=config['USE_DROPOUT_ON_UPSAMPLING']
                            )
        elif config['MODEL']=='unet':
            model =  custom_unet((config['TARGET_SIZE'][0], config['TARGET_SIZE'][1], config['N_DATA_BANDS']),
                            config['FILTERS'],
                            nclasses=config['NCLASSES'], 
                            kernel_size=(config['KERNEL'],config['KERNEL']),
                            strides=config['STRIDE'],
                            dropout=config['DROPOUT'],
                            dropout_change_per_layer=config['DROPOUT_CHANGE_PER_LAYER'],
                            dropout_type=config['DROPOUT_TYPE'],
                            use_dropout_on_upsampling=config['USE_DROPOUT_ON_UPSAMPLING'],
                            )

        elif config['MODEL'] =='simple_resunet':

            model = simple_resunet((config['TARGET_SIZE'][0], config['TARGET_SIZE'][1], config['N_DATA_BANDS']),
                        kernel = (2, 2),
                        num_classes=config['NCLASSES'],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=config['DROPOUT'],
                        dropout_change_per_layer=config['DROPOUT_CHANGE_PER_LAYER'],
                        dropout_type=config['DROPOUT_TYPE'],
                        use_dropout_on_upsampling=config['USE_DROPOUT_ON_UPSAMPLING'],
                        filters=config['FILTERS'],
                        num_layers=4,
                        strides=(1,1))

        elif config['MODEL']=='simple_unet':
            model = simple_unet((config['TARGET_SIZE'][0], config['TARGET_SIZE'][1], config['N_DATA_BANDS']),
                        kernel = (2, 2),
                        num_classes=config['NCLASSES'],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=config['DROPOUT'],
                        dropout_change_per_layer=config['DROPOUT_CHANGE_PER_LAYER'],
                        dropout_type=config['DROPOUT_TYPE'],
                        use_dropout_on_upsampling=config['USE_DROPOUT_ON_UPSAMPLING'],
                        filters=config['FILTERS'],
                        num_layers=4,
                        strides=(1,1))

        elif config['MODEL']=='satunet':
            model = custom_satunet((config['TARGET_SIZE'][0], config['TARGET_SIZE'][1], config['N_DATA_BANDS']),
                        kernel = (2, 2),
                        num_classes=config['NCLASSES'], 
                        activation="relu",
                        use_batch_norm=True,
                        dropout=config['DROPOUT'],
                        dropout_change_per_layer=config['DROPOUT_CHANGE_PER_LAYER'],
                        dropout_type=config['DROPOUT_TYPE'],
                        use_dropout_on_upsampling=config['USE_DROPOUT_ON_UPSAMPLING'],
                        filters=config['FILTERS'],
                        num_layers=4,
                        strides=(1,1))

        elif config['MODEL']=='segformer':
            id2label = {}
            for k in range(config['NCLASSES']):
                id2label[k]=str(k)
            model = segformer(id2label,num_classes=config['NCLASSES'])
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

    train_filenames = tf.io.gfile.glob(data_path+os.sep+config['ROOT_STRING']+'*.npz')

    val_filenames = tf.io.gfile.glob(val_data_path+os.sep+config['ROOT_STRING']+'*.npz')

    #----------------------------------------------------------

    shuffle(train_filenames) ##shuffle here
    shuffle(val_filenames) ##shuffle here

    list_ds = tf.data.Dataset.list_files(train_filenames, shuffle=False) ##dont shuffle here

    val_list_ds = tf.data.Dataset.list_files(val_filenames, shuffle=False) ##dont shuffle here


    validation_steps = len(val_filenames) // config_vars['BATCH_SIZE']
    steps_per_epoch =  len(train_filenames) // config_vars['BATCH_SIZE']

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

    # weights = weights.replace('.h5','.weights.h5')

    ######################
    #### set up data throughput pipeline
    #####################
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

    if 'LOAD_DATA_WITH_CPU' not in config_vars:
        config_vars['LOAD_DATA_WITH_CPU'] = False #default
        print('LOAD_DATA_WITH_CPU not specified in config file. Setting to "False"')

    if config_vars['LOAD_DATA_WITH_CPU']:
        with tf.device("CPU"):
            if config_vars['MODEL']=='segformer':
                train_ds = list_ds.map(read_seg_dataset_multiclass_segformer, num_parallel_calls=AUTO)

                val_ds = val_list_ds.map(read_seg_dataset_multiclass_segformer, num_parallel_calls=AUTO)

            else:
                train_ds = list_ds.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
                val_ds = val_list_ds.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
    else:
        if config_vars['MODEL']=='segformer':
            train_ds = list_ds.map(read_seg_dataset_multiclass_segformer, num_parallel_calls=AUTO)

            val_ds = val_list_ds.map(read_seg_dataset_multiclass_segformer, num_parallel_calls=AUTO)

        else:
            train_ds = list_ds.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
            val_ds = val_list_ds.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)


    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(config_vars['BATCH_SIZE'], drop_remainder=True) # drop_remainder will be needed on TPU (and possible with distributed gpus)
    train_ds = train_ds.prefetch(AUTO) #

    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(config_vars['BATCH_SIZE'], drop_remainder=True) # drop_remainder will be needed on TPU
    val_ds = val_ds.prefetch(AUTO) #


    # ## check bad batches
    # L=[]
    # for counter, samples in enumerate(train_ds.take(len(train_filenames)//BATCH_SIZE)):
    #     sample_image, sample_mask = samples["pixel_values"][0], samples["labels"][0]
    #     lstack_gt = np.zeros((TARGET_SIZE[0],TARGET_SIZE[1],NCLASSES))
    #     lstack_gt[:,:,:NCLASSES+1] = (np.arange(NCLASSES) == 1+sample_mask.numpy()[...,None]-1).astype(int) 
    #     L.append(lstack_gt.shape) 
    # ## these should all the same size

    ### the following code is for troubleshooting, when do_viz=True
    do_viz = False
    # do_viz=True

    if do_viz == True:
        ## to view examples
        class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477',
                                '#66AA00','#B82E2E', '#316395','#0d0887', '#46039f', '#7201a8',
                                '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']

        class_label_colormap = class_label_colormap[:config_vars['NCLASSES']]

        if config_vars['MODEL']=='segformer':

            for counter, samples in enumerate(train_ds.take(100)):
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

                plt.imshow(color_label,  alpha=0.75, vmin=0, vmax=config_vars['NCLASSES'])

                plt.axis('off')
                plt.savefig('example-{}.png'.format(counter), dpi=300)
                plt.close()

        else:

            for counter, (imgs,lbls) in enumerate(train_ds.take(100)):
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

                    plt.imshow(color_label,  alpha=0.75, vmin=0, vmax=config_vars['NCLASSES'])

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
    if config_vars['USE_MULTI_GPU']:
        with strategy.scope():
            model = get_model(for_model_save=False)

    else: ## single GPU
        model = get_model(for_model_save=False)

    if config_vars['MODEL']!='segformer':
        if config_vars['LOSS']=='dice':
            if 'LOSS_WEIGHTS' in config_vars:
                if config_vars['LOSS_WEIGHTS'] is True:

                        print("Computing loss weights per class ...")
                        N = []
                        # compute class-frequency distribution for 30 batches of training images 
                        for _,lbls in train_ds.take(30):
                            for _,lab in zip(_, lbls):
                                lab = np.argmax(lab.numpy().squeeze(),-1).flatten()
                                # make sure bincount is same length as number of classes
                                N.append(np.bincount(lab,minlength=config_vars['NCLASSES'])) #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0]))
                        # mean of class-frequencies
                        class_weights = np.mean(np.vstack(N),axis=0)
                        # inverse weighting
                        class_weights = 1-(class_weights/np.sum(class_weights))

                elif type(config_vars['LOSS_WEIGHTS']) is list:
                        class_weights = np.array(config_vars['LOSS_WEIGHTS'])
                        # inverse weighting
                        class_weights = 1-(class_weights/np.sum(class_weights))
                        print("Model compiled with class weights {}".format(class_weights)) 
                else:
                    class_weights = np.ones(config_vars['NCLASSES'])

                    model.compile(optimizer = 'adam', loss =weighted_dice_coef_loss(config_vars['NCLASSES'],class_weights), metrics = [iou_multi(config_vars['NCLASSES']), dice_multi(config_vars['NCLASSES'])], run_eagerly=True)
            else:
                    model.compile(optimizer = 'adam', loss =dice_coef_loss(config_vars['NCLASSES']), metrics = [iou_multi(config_vars['NCLASSES']), dice_multi(config_vars['NCLASSES'])], run_eagerly=True)

        else:
                if config_vars['LOSS']=='hinge':
                    model.compile(optimizer = 'adam', loss =tf.keras.losses.CategoricalHinge(), metrics = [iou_multi(config_vars['NCLASSES']), dice_multi(config_vars['NCLASSES'])], run_eagerly=True) 
                elif config_vars['LOSS'].startswith('cat'):
                    model.compile(optimizer = 'adam', loss =tf.keras.losses.CategoricalCrossentropy(), metrics = [iou_multi(config_vars['NCLASSES']), dice_multi(config_vars['NCLASSES'])], run_eagerly=True)
                elif config_vars['LOSS'].startswith('k'):
                    model.compile(optimizer = 'adam', loss =tf.keras.losses.KLDivergence(), metrics = [iou_multi(config_vars['NCLASSES']), dice_multi(config_vars['NCLASSES'])], run_eagerly=True)

    #----------------------------------------------------------

    ##########################################
    ##### set up callbacks
    #######################################

    # Open the file
    with open(config_vars['MODEL']+'_report.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    if config_vars['MODEL'] =='resunet':
        try:
            tf.keras.utils.plot_model(model,to_file="residual_unet_test.png",dpi=200)
        except:
            pass
    elif config_vars['MODEL']=='unet':
        try:
            tf.keras.utils.plot_model(model,to_file="unet_test.png",dpi=200)
        except:
            pass

    earlystop = EarlyStopping(monitor="val_loss",
                            mode="min", patience=config['PATIENCE'])

    # set checkpoint file
    model_checkpoint = ModelCheckpoint(weights, monitor='val_loss',
                                    verbose=0, save_best_only=True, mode='min',
                                    save_weights_only = True)


    # models are sensitive to specification of learning rate. How do you decide? Answer: you don't. Use a learning rate scheduler
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

    if 'CLEAR_MEMORY' not in config_vars:
        config_vars['CLEAR_MEMORY'] = False

    if config_vars['CLEAR_MEMORY']:

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

    if config_vars['DO_TRAIN']:

        if 'HOT_START' in config_vars:
            if 'INITIAL_EPOCH' not in config_vars:
                print("if HOT_START is specified, INITIAL_EPOCH must also be specified in the config file. Exiting ...")
                sys.exit(2)
            model.load_weights(config_vars['HOT_START'])
            print('transfering model weights for hot start ...')
        else:
            if 'INITIAL_EPOCH' not in config_vars:
                config_vars['INITIAL_EPOCH']=0
                print("INITIAL_EPOCH not specified in the config file. Setting to default of 0 ...")

        print('.....................................')
        print('Training model ...')
        tf.config.run_functions_eagerly(True)
        history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=config_vars['MAX_EPOCHS'],
                            validation_data=val_ds, validation_steps=validation_steps, initial_epoch=config_vars['INITIAL_EPOCH'],
                            callbacks=callbacks)

        # Plot training history
        plot_seg_history_iou(history, hist_fig, config_vars['MODEL'])

        plt.close('all')
        K.clear_session()

        try:    
            np.savez_compressed(weights.replace('.h5','_model_history.npz'),**history.history)
        except: 
            print("model training history could not be saved")

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

    if config_vars['MODEL']!='segformer':
        print('loss={loss:0.4f}, Mean IOU={mean_iou:0.4f}, Mean Dice={mean_dice:0.4f}'.format(loss=scores[0], mean_iou=scores[1], mean_dice=scores[2]))
    else:
        print('loss={loss:0.4f}'.format(loss=scores))

    # # # ##########################################################
    IOUc, Dc, Kc, OA, MIOU, FWIOU, MCC = plotcomp_n_metrics(
                                    val_ds,model,config_vars['NCLASSES'],config['DOPLOT'],test_samples_fig,'val', config_vars['MODEL'])
    print('Mean of mean IoUs (validation subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(IOUc)))
    print('Mean of mean IoUs, confusion matrix (validation subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(MIOU)))
    print('Mean of mean frequency weighted IoUs, confusion matrix (validation subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(FWIOU)))
    print('Mean of Matthews Correlation Coefficients (validation subset)={mean_dice:0.3f}'.format(mean_dice=np.mean(MCC)))
    print('Mean of mean Dice scores (validation subset)={mean_dice:0.3f}'.format(mean_dice=np.mean(Dc)))
    print('Mean of mean KLD scores (validation subset)={mean_kld:0.3f}'.format(mean_kld=np.mean(Kc)))


    IOUc, Dc, Kc, OA, MIOU, FWIOU, MCC = plotcomp_n_metrics(
                                    train_ds,model,config_vars['NCLASSES'],config['DOPLOT'],test_samples_fig,'train', config_vars['MODEL'])
    print('Mean of mean IoUs (train subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(IOUc)))
    print('Mean of mean IoUs, confusion matrix (train subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(MIOU)))
    print('Mean of mean frequency weighted IoUs, confusion matrix (train subset)={mean_iou:0.3f}'.format(mean_iou=np.mean(FWIOU)))
    print('Mean of Matthews Correlation Coefficients (train subset)={mean_dice:0.3f}'.format(mean_dice=np.mean(MCC)))
    print('Mean of mean Dice scores (train subset)={mean_dice:0.3f}'.format(mean_dice=np.mean(Dc)))
    print('Mean of mean KLD scores (train subset)={mean_kld:0.3f}'.format(mean_kld=np.mean(Kc)))

##boo-yah



if __name__ == '__main__':
    main()
