# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020-24, Marda Science LLC
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

# utility to merge multiple coincident jpeg images into nd numpy arrays
import sys,os, json, shutil
import argparse

from skimage.io import imread
import numpy as np
from glob import glob
from skimage.morphology import dilation, disk #remove_small_objects, remove_small_holes
from tqdm import tqdm
from joblib import Parallel, delayed
from natsort import natsorted
import matplotlib.pyplot as plt

from doodleverse_utils.imports import *
import random
random.seed(0)



##========================================================
## USER INPUTS
##========================================================

parser = argparse.ArgumentParser(description='Process some directories and files.')
parser.add_argument('-o', '--output', required=True, help='Select directory for OUTPUT files')
parser.add_argument('-c', '--config', required=True, help='Select config file')
parser.add_argument('-l', '--label_dir', required=True, help='Select directory of LABEL files')
parser.add_argument('-i', '--image_dirs', required=True, nargs='+', help='Select directories of IMAGE files')
args = parser.parse_args()

output_data_path = args.output
print(output_data_path)

configfile = args.config
print(f"Config file: {configfile}")

label_data_path = args.label_dir
print(f"Label directory: {label_data_path}")

image_dirs = args.image_dirs
print(f"Image directories: {image_dirs}")

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
    print('Warning: using CPU - data making may be slower than GPU')

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
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.run_functions_eagerly(True)
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
    # from tensorflow.python.client import device_lib
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)


for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)
print(tf.config.get_visible_devices())

if USE_MULTI_GPU:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy([p.name.split('/physical_device:')[-1] for p in physical_devices], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print("Number of distributed devices: {}".format(strategy.num_replicas_in_sync))


###================================================
### get data files
###===============================================

W = image_dirs

##========================================================
## COLLATE FILES INTO LISTS
##========================================================

if len(W)>1:
    files = []
    for data_path in W:
        f = natsorted(glob(data_path+os.sep+'*.jpg')) + natsorted(glob(data_path+os.sep+'*.png')) + natsorted(glob(data_path+os.sep+'*.tif'))
        if len(f)<1:
            f = natsorted(glob(data_path+os.sep+'images'+os.sep+'*.jpg')) + natsorted(glob(data_path+os.sep+'images'+os.sep+'*.png')) + natsorted(glob(data_path+os.sep+'images'+os.sep+'*.tif'))
        files.append(f)
    # number of bands x number of samples
    files = np.vstack(files).T
else:
    data_path = W[0]
    files = natsorted(glob(data_path+os.sep+'*.jpg')) + natsorted(glob(data_path+os.sep+'*.png'))  + natsorted(glob(data_path+os.sep+'*.tif'))
    if len(files)<1:
        files = natsorted(glob(data_path+os.sep+'images'+os.sep+'*.jpg')) + natsorted(glob(data_path+os.sep+'images'+os.sep+'*.png')) + natsorted(glob(data_path+os.sep+'images'+os.sep+'*.tif'))

label_files = natsorted(glob(label_data_path+os.sep+'*.jpg')) + natsorted(glob(label_data_path+os.sep+'*.png')) + natsorted(glob(label_data_path+os.sep+'*.tif'))
if len(label_files)<1:
    label_files = natsorted(glob(label_data_path+os.sep+'labels'+os.sep+'*.jpg')) + natsorted(glob(label_data_path+os.sep+'labels'+os.sep+'*.png')) + natsorted(glob(label_data_path+os.sep+'labels'+os.sep+'*.tif'))


print("Found {} image and {} label files".format(len(files), len(label_files)))

##========================================================
## MAKING RESIZED COPIES OF IMAGERY
##========================================================

## make  direcs
for w in W:
    wend = w.split('/')[-1]
    newdirec = w.replace(wend,'resized_'+wend)

    try:
        os.mkdir(newdirec)
        os.mkdir(newdirec+os.sep+'images')
    except:
        pass

if USEMASK:
    newdireclabels = label_data_path.replace('mask','resized_mask')
else:
    newdireclabels = label_data_path.replace('label','resized_label')

# if directories already exist, skip them
if os.path.isdir(newdireclabels):
    print("{} already exists: skipping the image resizing step".format(newdireclabels))
else:

    try:
        os.mkdir(newdireclabels)
        os.mkdir(newdireclabels+os.sep+'images')
    except:
        pass

    if len(W)==1:
        try:
            w = Parallel(n_jobs=-2, verbose=0, max_nbytes=None)(delayed(do_resize_image)(os.path.normpath(f), TARGET_SIZE) for f in files)
            w = Parallel(n_jobs=-2, verbose=0, max_nbytes=None)(delayed(do_resize_label)(os.path.normpath(lfile), TARGET_SIZE) for lfile in label_files)
        except:

            ## cycle through, merge and padd/resize if need to
            for f in files:
                do_resize_image(os.path.normpath(f), TARGET_SIZE)

            for lfile in label_files:
                do_resize_label(os.path.normpath(lfile), TARGET_SIZE)

    else:
        ## cycle through, merge and padd/resize if need to
        # print("Several sets of input imagery -- resizing takes place in serial (slower)")
        for file,lfile in zip(files, label_files):

            for f in file:
                do_resize_image(f, TARGET_SIZE)
            do_resize_label(lfile, TARGET_SIZE)


## write padded labels to file
label_data_path = newdireclabels+os.sep+'images'

label_files = natsorted(glob(label_data_path+os.sep+'*.png')) + natsorted(glob(label_data_path+os.sep+'*.tif'))
if len(label_files)<1:
    label_files = natsorted(glob(label_data_path+os.sep+'images'+os.sep+'*.png')) +  natsorted(glob(label_data_path+os.sep+'images'+os.sep+'*.tif'))
print("{} label files".format(len(label_files)))

W2 = []
for w in W:
    wend = os.path.normpath(w).split(os.sep)[-1]
    w = w.replace(wend,'resized_'+wend)
    W2.append(w+os.sep+'images')
# W = W2
# del W2

files = []
for data_path in W2:
    f = natsorted(glob(os.path.normpath(data_path)+os.sep+'*.png')) + natsorted(glob(os.path.normpath(data_path)+os.sep+'*.tif'))
    if len(f)<1:
        f = natsorted(glob(os.path.normpath(data_path)+os.sep+'images'+os.sep+'*.png')) + natsorted(glob(os.path.normpath(data_path)+os.sep+'images'+os.sep+'*.tif'))
    files.append(f)

# number of bands x number of samples
files = np.vstack(files).T
print("{} sets of {} image files".format(len(W),len(files)))

###================================================
#----------------------------------------------------------

#make output direc structure
try:
    os.mkdir(output_data_path+os.sep+'train_data')
    os.mkdir(output_data_path+os.sep+'val_data')
except:
    pass

if N_DATA_BANDS>3:

    try:
        os.mkdir(output_data_path+os.sep+'train_data'+os.sep+'train_labels')
        os.mkdir(output_data_path+os.sep+'train_data'+os.sep+'train_npzs')
        os.mkdir(output_data_path+os.sep+'val_data'+os.sep+'val_labels')
        os.mkdir(output_data_path+os.sep+'val_data'+os.sep+'val_npzs')
    except:
        pass

    try:
        os.mkdir(output_data_path+os.sep+'train_data'+os.sep+'train_images')
        os.mkdir(output_data_path+os.sep+'val_data'+os.sep+'val_images')
    except:
        pass

    try:
        for counter in range(len(files[0])):
            os.mkdir(output_data_path+os.sep+'train_data'+os.sep+'train_images'+os.sep+'set'+str(counter))
            os.mkdir(output_data_path+os.sep+'val_data'+os.sep+'val_images'+os.sep+'set'+str(counter))
    except:
        pass

else: ##N_DATA_BANDS<=3
    try:
        os.mkdir(output_data_path+os.sep+'train_data'+os.sep+'train_images')
        os.mkdir(output_data_path+os.sep+'train_data'+os.sep+'train_labels')
        os.mkdir(output_data_path+os.sep+'train_data'+os.sep+'train_npzs')
        os.mkdir(output_data_path+os.sep+'val_data'+os.sep+'val_images')
        os.mkdir(output_data_path+os.sep+'val_data'+os.sep+'val_labels')
        os.mkdir(output_data_path+os.sep+'val_data'+os.sep+'val_npzs')
    except:
        pass

## shuffle
for k in range(3):
    temp = list(zip(files, label_files))
    random.shuffle(temp)
    files, label_files = zip(*temp)
    files, label_files = list(files), list(label_files)

if N_DATA_BANDS>3:
    list_of_list_ds_images = []
    for counter in range(len(files[0])): 
        list_of_list_ds_images.append(tf.data.Dataset.list_files( [f[counter] for f in files], shuffle=False))
else:
    files = [f[0] for f in files]
    list_ds_images = tf.data.Dataset.list_files(files, shuffle=False)

list_ds_labels = tf.data.Dataset.list_files(label_files, shuffle=False)

val_size = int(len(files) * VALIDATION_SPLIT)

if N_DATA_BANDS>3:
    list_of_train_ds = []
    list_of_val_ds = []
    for counter in range(len(files[0])):     
        list_of_train_ds.append(list_of_list_ds_images[counter].skip(val_size))
        list_of_val_ds.append(list_of_list_ds_images[counter].take(val_size))
else:
    train_ds = list_ds_images.skip(val_size)
    val_ds = list_ds_images.take(val_size)

if N_DATA_BANDS>3:
    list_of_train_file_lists = []
    for counter in range(len(files[0])):
        train_files = []
        for i in list_of_train_ds[counter]:
            train_files.append(i.numpy().decode().split(os.sep)[-1])        
        list_of_train_file_lists.append(train_files)

    list_of_val_file_lists = []
    for counter in range(len(files[0])):
        val_files = []
        for i in list_of_val_ds[counter]:
            val_files.append(i.numpy().decode().split(os.sep)[-1])        
        list_of_val_file_lists.append(val_files)

    for counter,train_files in enumerate(list_of_train_file_lists):
        for i in train_files:
            ii = i.split(os.sep)[-1]
            # shutil.copyfile(os.path.normpath(W2[counter])+os.sep+i,output_data_path+os.sep+'train_data'+os.sep+'train_images'+os.sep+ii)
            shutil.copyfile(os.path.normpath(W2[counter])+os.sep+i,output_data_path+os.sep+'train_data'+os.sep+'train_images'+os.sep+'set'+str(counter)+os.sep+ii)

    for counter,val_files in enumerate(list_of_val_file_lists):
        for i in val_files:
            ii = i.split(os.sep)[-1]
            # shutil.copyfile(os.path.normpath(data_path)+os.sep+i,output_data_path+os.sep+'val_data'+os.sep+'val_images'+os.sep+ii)
            shutil.copyfile(os.path.normpath(W2[counter])+os.sep+i,output_data_path+os.sep+'val_data'+os.sep+'val_images'+os.sep+'set'+str(counter)+os.sep+ii)

else:
    train_files = []
    for i in train_ds:
        train_files.append(i.numpy().decode().split(os.sep)[-1])

    val_files = []
    for i in val_ds:
        val_files.append(i.numpy().decode().split(os.sep)[-1])

    for i in train_files:
        ii = i.split(os.sep)[-1]
        shutil.copyfile(os.path.normpath(data_path)+os.sep+i,output_data_path+os.sep+'train_data'+os.sep+'train_images'+os.sep+ii)

    for i in val_files:
        ii = i.split(os.sep)[-1]
        shutil.copyfile(os.path.normpath(data_path)+os.sep+i,output_data_path+os.sep+'val_data'+os.sep+'val_images'+os.sep+ii)


## labels
train_ds = list_ds_labels.skip(val_size)
val_ds = list_ds_labels.take(val_size)

train_label_files = []
for i in train_ds:
    train_label_files.append(i.numpy().decode().split(os.sep)[-1])

val_label_files = []
for i in val_ds:
    val_label_files.append(i.numpy().decode().split(os.sep)[-1])


for i in train_label_files:
    ii = i.split(os.sep)[-1]
    try:
        shutil.copyfile(os.path.normpath(label_data_path)+os.sep+'images'+os.sep+i,output_data_path+os.sep+'train_data'+os.sep+'train_labels'+os.sep+ii)
    except:
        shutil.copyfile(os.path.normpath(label_data_path)+os.sep+i,output_data_path+os.sep+'train_data'+os.sep+'train_labels'+os.sep+ii)

for i in val_label_files:
    ii = i.split(os.sep)[-1]
    try:
        shutil.copyfile(os.path.normpath(label_data_path)+os.sep+'images'+os.sep+i,output_data_path+os.sep+'val_data'+os.sep+'val_labels'+os.sep+ii)
    except:
        shutil.copyfile(os.path.normpath(label_data_path)+os.sep+i,output_data_path+os.sep+'val_data'+os.sep+'val_labels'+os.sep+ii)


###================================================

##========================================================
## NON-AUGMENTED FILES
##========================================================

def get_lists_of_images(f,l):
    if type(f )== list:
        im=[] # read all images into a list
        for k in f:
            tmp = imread(k)
            try:
                tmp = tmp[:,:,:3]
            except:
                tmp = np.squeeze(tmp)
                tmp = np.dstack((tmp,tmp,tmp))[:,:,:3]
            im.append(tmp)
    else:
        im = imread(f)
        try:
            im = im[:,:,:3]
        except:
            im = np.squeeze(im)
            im = np.dstack((im,im,im))[:,:,:3]     

    return im 


def do_label_filter(lstack,FILTER_VALUE,NCLASSES):
    nx,ny,_ = lstack.shape

    #print("dilating labels with a radius of {}".format(FILTER_VALUE))
    initial_sum = np.sum(np.argmax(lstack,-1))
    lstack_copy = lstack.copy()
    for kk in range(lstack.shape[-1]):
        lab = dilation(lstack[:,:,kk].astype('uint8')>0, disk(FILTER_VALUE))
        lstack_copy[:,:,kk] = np.ceil(lab).astype(np.uint8)
        del lab
    final_sum = np.sum(np.argmax(lstack_copy,-1))
    if (final_sum < initial_sum) and (NCLASSES==2): ### this ambiguity can happen in 0/1 masks (NCLASSES=2)
        lstack_copy = lstack.copy()

        for kk in range(lstack.shape[-1]):
            lab = dilation(lstack[:,:,kk].astype('uint8')==0, disk(FILTER_VALUE))
            lstack_copy[:,:,kk] = np.round(lab).astype(np.uint8)
            del lab

        lab = ~np.argmax(lstack_copy,-1)
        if lab.min()<0:
            lab -= lab.min()
    else:
        lab = np.argmax(lstack_copy,-1)
        if lab.min()<0:
            lab -= lab.min()

    lstack = np.zeros((nx,ny,NCLASSES))
    lstack[:,:,:NCLASSES] = (np.arange(NCLASSES) == 1+lab[...,None]-1).astype(int) #one-hot encode 
    return lstack   

def get_lab_stack(lab, NCLASSES):
    if len(np.unique(lab))==1:
        nx,ny = lab.shape
        lstack = np.zeros((nx,ny,NCLASSES))
        try:
            lstack[:,:,np.unique(lab)[0]]=np.ones((nx,ny))
        except:
            lstack[:,:,0]=np.ones((nx,ny))

    else:
        nx,ny = lab.shape
        lstack = np.zeros((nx,ny,NCLASSES))
        lstack[:,:,:NCLASSES] = (np.arange(NCLASSES) == 1+lab[...,None]-1).astype(int) #one-hot encode
    return lstack

def doviz(datadict, counter, N_DATA_BANDS):
    if counter%10 ==0:
        if N_DATA_BANDS>3:
            plt.imshow(datadict['arr_0'][:,:,0], cmap='gray')
        else:
            plt.imshow(datadict['arr_0'])            
        plt.imshow(np.argmax(datadict['arr_1'], axis=-1), alpha=0.3); plt.axis('off')
        plt.savefig('ex{}.png'.format(counter),dpi=200)

#=============================
do_viz = False
# do_viz = True

print("Creating non-augmented train subset")
## make non-aug subset first
# cycle through pairs of files and labels

if N_DATA_BANDS>3:
    train_files = np.vstack(list_of_train_file_lists).T 

for counter,(f,l) in enumerate(zip(train_files,train_label_files)):

    if N_DATA_BANDS>3:
        ff=[]
        for cnt in range(len(f)):
            ff.append( output_data_path+os.sep+'train_data'+os.sep+'train_images'+os.sep+'set'+str(cnt)+os.sep+f[cnt] )
        f = ff 
        del ff
    else:
        f = output_data_path+os.sep+'train_data'+os.sep+'train_images'+os.sep+f
    l = output_data_path+os.sep+'train_data'+os.sep+'train_labels'+os.sep+l

    im = get_lists_of_images(f,l)
    # print(im.shape)

    datadict={}
    datadict['arr_0'] = im.astype(np.uint8)

    lab = imread(l) # reac the label)

    if 'REMAP_CLASSES' in locals():
        for k in REMAP_CLASSES.items():
            lab[lab==int(k[0])] = int(k[1])
    else:
        lab[lab>NCLASSES]=NCLASSES

    lstack = get_lab_stack(lab, NCLASSES)

    if FILTER_VALUE>1:
        lstack = do_label_filter(lstack,FILTER_VALUE,NCLASSES)

    datadict['arr_1'] = np.squeeze(lstack).astype(np.uint8)

    if do_viz == True:
        doviz(datadict, counter, N_DATA_BANDS)

    datadict['num_bands'] = im.shape[-1]
    datadict['files'] = [fi.split(os.sep)[-1] for fi in f]

    segfile = output_data_path+os.sep+'train_data'+os.sep+'train_npzs'+os.sep+ROOT_STRING+'_noaug_nd_data_000000'+str(counter)+'.npz'
    np.savez_compressed(segfile, **datadict)
    del datadict#, im, lstack


#######=============================================
print("Creating non-augmented validation subset")
## make non-aug subset first
# cycle through pairs of files and labels
if N_DATA_BANDS>3:
    val_files = np.vstack(list_of_val_file_lists).T 

for counter,(f,l) in enumerate(zip(val_files,val_label_files)):

    if N_DATA_BANDS>3:
        ff=[]
        for cnt in range(len(f)):
            ff.append( output_data_path+os.sep+'val_data'+os.sep+'val_images'+os.sep+'set'+str(cnt)+os.sep+f[cnt] )
        f = ff 
        del ff
    else:
        f = output_data_path+os.sep+'val_data'+os.sep+'val_images'+os.sep+f

    l = output_data_path+os.sep+'val_data'+os.sep+'val_labels'+os.sep+l

    im = get_lists_of_images(f,l)
    # print(im.shape)

    datadict={}
    datadict['arr_0'] = im.astype(np.uint8)

    lab = imread(l) # reac the label)

    if 'REMAP_CLASSES' in locals():
        for k in REMAP_CLASSES.items():
            lab[lab==int(k[0])] = int(k[1])
    else:
        lab[lab>NCLASSES]=NCLASSES

    lstack = get_lab_stack(lab, NCLASSES)

    if FILTER_VALUE>1:
        lstack = do_label_filter(lstack,FILTER_VALUE,NCLASSES)

    datadict['arr_1'] = np.squeeze(lstack).astype(np.uint8)

    if do_viz == True:
        doviz(datadict, counter, N_DATA_BANDS)

    datadict['num_bands'] = im.shape[-1]
    datadict['files'] = [fi.split(os.sep)[-1] for fi in f]

    segfile = output_data_path+os.sep+'val_data'+os.sep+'val_npzs'+os.sep+ROOT_STRING+'_noaug_nd_data_000000'+str(counter)+'.npz'
    np.savez_compressed(segfile, **datadict)
    del datadict

###================================

##========================================================
## READ, VERIFY and PLOT NON-AUGMENTED FILES
##========================================================

#-----------------------------------
def load_npz(example):
    with np.load(example.numpy()) as data:
        image = data['arr_0'].astype('uint8')
        image = standardize(image)
        label = data['arr_1'].astype('uint8')
        try:
            file = [''.join(f) for f in data['files']]
        except:
            file = [f]
    return image, label, file[0]

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
    image, label, file = tf.py_function(func=load_npz, inp=[example], Tout=[tf.float32, tf.uint8, tf.string])

    return image, label, file

###================================

filenames = tf.io.gfile.glob(output_data_path+os.sep+'train_data'+os.sep+'train_npzs'+os.sep+ROOT_STRING+'_noaug*.npz')
dataset = tf.data.Dataset.list_files(filenames, shuffle=False)

print('{} non-aug. training files'.format(len(filenames)))

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
dataset = dataset.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
dataset = dataset.repeat()
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
dataset = dataset.prefetch(AUTO)

try:
    os.mkdir(output_data_path+os.sep+'train_data'+os.sep+'train_npzs'+os.sep+'noaug_sample')
except:
    pass

#blue,red, yellow,green, etc
class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477',
                        '#66AA00','#B82E2E', '#316395','#0d0887', '#46039f', '#7201a8',
                        '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']

class_label_colormap = class_label_colormap[:NCLASSES]

print('.....................................')
print('Printing examples to file ...')

counter=0
for imgs,lbls,files in dataset.take(10):

  for count,(im,lab, file) in enumerate(zip(imgs, lbls, files)):

     im = rescale_array(im.numpy(), 0, 1)
     if im.shape[-1]:
         im = im[:,:,:3]
        #  print(im.shape)

     if N_DATA_BANDS==1:
         plt.imshow(im, cmap='gray')
     else:
         plt.imshow(im)

     lab = np.argmax(lab.numpy().squeeze(),-1)

     color_label = label_to_colors(np.squeeze(lab), tf.cast(im[:,:,0]==0,tf.uint8),
                                    alpha=128, colormap=class_label_colormap,
                                     color_class_offset=0, do_alpha=False)

     plt.imshow(color_label,  alpha=0.5)

     file = file.numpy()

     plt.axis('off')
     plt.title(file)
     plt.savefig(output_data_path+os.sep+'train_data'+os.sep+'train_npzs'+os.sep+'noaug_sample'+os.sep+ ROOT_STRING + 'noaug_ex'+str(counter)+'.png', dpi=200, bbox_inches='tight')
     plt.close('all')
     counter += 1

##========================================================
## AUGMENTED FILES
##========================================================

# -train images are augmented depending on existing aug configs, and put into train data folder

print("Creating augmented files")

# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=AUG_ROT,
                     width_shift_range=AUG_WIDTHSHIFT,
                     height_shift_range=AUG_HEIGHTSHIFT,
                     fill_mode='reflect', #'nearest',
                     zoom_range=AUG_ZOOM,
                     horizontal_flip=AUG_HFLIP,
                     vertical_flip=AUG_VFLIP)

null_data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=0,
                     width_shift_range=0,
                     height_shift_range=0,
                     fill_mode='reflect',
                     zoom_range=0,
                     horizontal_flip=False,
                     vertical_flip=False)

#get image dimensions
NX = TARGET_SIZE[0]
NY = TARGET_SIZE[1]

null_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**null_data_gen_args)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
null_mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**null_data_gen_args)

# important that each band has the same image generator
image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

### here, we are only going to augment the training data

W = []
for it in os.scandir(output_data_path+os.sep+'train_data'+os.sep+'train_images'):
    if it.is_dir():
        W.append(it.path)
W = sorted(W)

if len(W)<1:
    W = [output_data_path+os.sep+'train_data'+os.sep+'train_images']


## put TRAIN images in subfolders
for counter,w in enumerate(W):
    n_im = len(glob(w+os.sep+'*.png')+glob(w+os.sep+'*.jpg')+glob(w+os.sep+'*.tif'))
    if n_im>0:
        try:
            os.mkdir(w+os.sep+'images')
        except:
            pass
        for file in glob(w+os.sep+'*.png')+glob(w+os.sep+'*.jpg')+glob(w+os.sep+'*.tif'):
            try:
                shutil.move(file,w+os.sep+'images')
            except:
                pass

    n_im = len(glob(w+os.sep+'images'+os.sep+'*.*'))


label_data_path = output_data_path+os.sep+'train_data'+os.sep+'train_labels'


## put label images in subfolders
n_im = len(glob(label_data_path+os.sep+'*.png')+glob(label_data_path+os.sep+'*.jpg')+glob(label_data_path+os.sep+'*.tif'))
if n_im>0:
    try:
        os.mkdir(label_data_path+os.sep+'images')
    except:
        pass

for file in glob(label_data_path+os.sep+'*.png')+glob(label_data_path+os.sep+'*.jpg')+glob(label_data_path+os.sep+'*.tif'):
    try:
        shutil.move(file,label_data_path+os.sep+'images')
    except:
        pass   

n_im = len(glob(label_data_path+os.sep+'images'+os.sep+'*.*'))


#### make training generators directly, and in advance
train_generators = []
null_train_generators = []
for counter,w in enumerate(W):
    print("folder: {}".format(w.split(os.sep)[-1]))
    img_generator = image_datagen.flow_from_directory(
        w,
        target_size=(NX, NY),
        batch_size=int(n_im/AUG_LOOPS),
        class_mode=None, seed=SEED, shuffle=False)

    null_img_generator = null_image_datagen.flow_from_directory(
            w,
            target_size=(NX, NY),
            batch_size=int(n_im/AUG_LOOPS),
            class_mode=None, seed=SEED, shuffle=False)

    print("folder: {}".format(label_data_path.split(os.sep)[-1]))
    #the seed must be the same as for the training set to get the same images
    mask_generator = mask_datagen.flow_from_directory(
            label_data_path,
            target_size=(NX, NY),
            batch_size=int(n_im/AUG_LOOPS),
            class_mode=None, seed=SEED, shuffle=False, color_mode="grayscale", interpolation="nearest")

    null_mask_generator = null_mask_datagen.flow_from_directory(
            label_data_path,
            target_size=(NX, NY),
            batch_size=int(n_im/AUG_LOOPS),
            class_mode=None, seed=SEED, shuffle=False, color_mode="grayscale", interpolation="nearest")

    train_generator = (pair for pair in zip(img_generator, mask_generator))
    train_generators.append([img_generator,mask_generator,train_generator])

    null_train_generator = (pair for pair in zip(null_img_generator, null_mask_generator))
    null_train_generators.append([null_img_generator, null_mask_generator,null_train_generator])

######################## generate and print files

i = 0
for copy in tqdm(range(AUG_COPIES)):
    for k in range(AUG_LOOPS):

        X=[]; Y=[]; F=[]
        for counter,train_generator in enumerate(train_generators):
            #grab a batch of images and label images
            x, y = next(train_generator[-1])
            y = np.round(y)

            idx = np.maximum((train_generator[0].batch_index - 1) * train_generator[0].batch_size, 0)
            filenames = train_generator[0].filenames[idx : idx + train_generator[0].batch_size]
            X.append(x)
            del x
            Y.append(y)
            del y
            F.append(filenames)
            del filenames


        Y = Y[0]
        # wrute them to file and increment the counter
        for counter,lab in enumerate(Y):

            im = np.dstack([x[counter] for x in X])## X3])
            # print(im.shape)
            files = np.dstack([x[counter] for x in F])

            ##============================================ label
            l = np.round(lab[:,:,0]).astype(np.uint8)
            # print(l.shape)

            if 'REMAP_CLASSES' in locals():
                for k in REMAP_CLASSES.items():
                    l[l==int(k[0])] = int(k[1])
            else:
                l[l>NCLASSES]=NCLASSES

            if len(np.unique(l))==1:
                nx,ny = l.shape
                lstack = np.zeros((nx,ny,NCLASSES))

                lstack[:,:,np.unique(l)[0]]=np.ones((nx,ny))
            else:
                nx,ny = l.shape
                lstack = np.zeros((nx,ny,NCLASSES))
                lstack[:,:,:NCLASSES] = (np.arange(NCLASSES) == 1+l[...,None]-1).astype(int) #one-hot encode

            if FILTER_VALUE>1:
                ##print("dilating labels with a radius of {}".format(FILTER_VALUE))
                for kk in range(lstack.shape[-1]):
                    if FILTER_VALUE<0:
                        lab = dilation(lstack[:,:,kk].astype('uint8')<1, disk(np.abs(FILTER_VALUE)))
                    else:
                        lab = dilation(lstack[:,:,kk].astype('uint8')>0, disk(FILTER_VALUE))
                    lstack[:,:,kk] = np.round(lab).astype(np.uint8)
                    del lab

            datadict={}
            datadict['arr_0'] = im.astype(np.uint8)
            datadict['arr_1'] =  np.squeeze(lstack).astype(np.uint8)
            datadict['num_bands'] = im.shape[-1]
            try:
                datadict['files'] = [fi.split(os.sep)[-1] for fi in files.squeeze()]
            except:
                datadict['files'] = [files]

            np.savez_compressed(output_data_path+os.sep+'train_data'+os.sep+'train_npzs'+os.sep+ROOT_STRING+'_aug_nd_data_000000'+str(i),
                                **datadict)

            del lstack, l, im

            i += 1

##========================================================
## READ, VERIFY and PLOT AUGMENTED FILES
##========================================================

filenames = tf.io.gfile.glob(output_data_path+os.sep+'train_data'+os.sep+'train_npzs'+os.sep+ROOT_STRING+'_aug*.npz')
dataset = tf.data.Dataset.list_files(filenames, shuffle=False)

print('{} files made'.format(len(filenames)))

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
dataset = dataset.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
dataset = dataset.repeat()
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
dataset = dataset.prefetch(AUTO)

try:
    os.mkdir(output_data_path+os.sep+'train_data'+os.sep+'train_npzs'+os.sep+'aug_sample')
except:
    pass


print('.....................................')
print('Printing examples to file ...')

counter=0
for imgs,lbls,files in dataset.take(10):

  for count,(im,lab, file) in enumerate(zip(imgs, lbls, files)):

     im = rescale_array(im.numpy(), 0, 1)
     if im.shape[-1]:
         im = im[:,:,:3] #just show the first 3 bands
        #  print(im.shape)

     if N_DATA_BANDS==1:
         plt.imshow(im, cmap='gray')
     else:
         plt.imshow(im)

     lab = np.argmax(lab.numpy().squeeze(),-1)

     color_label = label_to_colors(np.squeeze(lab), tf.cast(im[:,:,0]==0,tf.uint8),
                                    alpha=128, colormap=class_label_colormap,
                                     color_class_offset=0, do_alpha=False)

     plt.imshow(color_label,  alpha=0.5)

     try:
         file = file.numpy().split(os.sep)[-1]
         plt.title(file)
         del file
     except:
         pass

     plt.axis('off')

     plt.savefig(output_data_path+os.sep+'train_data'+os.sep+'train_npzs'+os.sep+'aug_sample'+os.sep+ ROOT_STRING + 'aug_ex'+str(counter)+'.png', dpi=200, bbox_inches='tight')
     plt.close('all')
     counter += 1

#boom.
