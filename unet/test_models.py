
import os
import json
from tkinter import filedialog
from tkinter import *
from random import shuffle


USE_GPU = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if USE_GPU == True:
    if 'SET_GPU' in locals():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(SET_GPU)
    else:
        #use the first available GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from imports import *


TARGET_SIZE = [768,1024]
N_DATA_BANDS=3
NCLASSES=1
DROPOUT=0.0
DROPOUT_TYPE='standard'
TARGET_SIZE = [768,768]
N_DATA_BANDS=3
NCLASSES=1
DROPOUT_CHANGE_PER_LAYER=0.0
USE_DROPOUT_ON_UPSAMPLING=False

FILTERS=8
KERNEL=7
STRIDE=2

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
model.compile(optimizer = 'adam', loss =dice_coef_loss, metrics = [mean_iou, dice_coef])
model.summary()



FILTERS=4
KERNEL=11
STRIDE=2

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
model.compile(optimizer = 'adam', loss =dice_coef_loss, metrics = [mean_iou, dice_coef])
model.summary()



FILTERS=18
KERNEL=7
STRIDE=2

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
model.compile(optimizer = 'adam', loss =dice_coef_loss, metrics = [mean_iou, dice_coef])
model.summary()




FILTERS=8
KERNEL=4#4#2
STRIDE=1

model = simple_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
            kernel = (KERNEL, KERNEL),
            num_classes=NCLASSES,#[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
            activation="relu",
            use_batch_norm=True,
            dropout=DROPOUT,#0.1,
            dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
            dropout_type=DROPOUT_TYPE,#"standard",
            use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
            filters=FILTERS,#8,
            num_layers=4,
            strides=(STRIDE,STRIDE))
model.compile(optimizer = 'adam', loss =dice_coef_loss, metrics = [mean_iou, dice_coef])
model.summary()

FILTERS=8
KERNEL=4#4#2
STRIDE=1
model = simple_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
            kernel = (KERNEL, KERNEL),
            num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
            activation="relu",
            use_batch_norm=True,
            dropout=DROPOUT,#0.1,
            dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
            dropout_type=DROPOUT_TYPE,#"standard",
            use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
            filters=FILTERS,#8,
            num_layers=4,
            strides=(STRIDE,STRIDE))
model.compile(optimizer = 'adam', loss =dice_coef_loss, metrics = [mean_iou, dice_coef])
model.summary()
