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

import sys,os, time
sys.path.insert(1, 'src')

USE_GPU = True #False

if USE_GPU == True:
   ##use the first available GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '1' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from prediction_imports import *
#====================================================

#====================================================

root = Tk()
root.filename =  filedialog.askopenfilename(title = "Select file",filetypes = (("weights file","*.h5"),("all files","*.*")))
weights = root.filename
print(weights)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(title = "Select directory of images to segment")
sample_direc = root.filename
print(sample_direc)
root.withdraw()

configfile = weights.replace('.h5','.json').replace('weights', 'config')


with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')


from imports import *

#=======================================================

print('.....................................')
print('Creating and compiling model ...')

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


# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])
model.compile(optimizer = 'adam', loss = dice_coef_loss, metrics = [mean_iou, dice_coef])

model.load_weights(weights)

metadatadict = {}
metadatadict['model_weights'] = weights
metadatadict['config_files'] = configfile
metadatadict['model_types'] = MODEL


### predict
print('.....................................')
print('Using model for prediction on images ...')

# sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))
sample_filenames = sorted(glob(sample_direc+os.sep+'*.jpg'))
if len(sample_filenames)==0:
    # sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.png'))
    sample_filenames = sorted(glob(sample_direc+os.sep+'*.png'))

print('Number of samples: %i' % (len(sample_filenames)))

#look for TTA config
if not 'TESTTIMEAUG' in locals():
    TESTTIMEAUG = False

for counter,f in enumerate(sample_filenames):
    do_seg(f, [model], metadatadict, sample_direc,NCLASSES, N_DATA_BANDS,TARGET_SIZE,TESTTIMEAUG)
    print('%i out of %i done'%(counter,len(sample_filenames)))
<<<<<<< HEAD

=======
>>>>>>> main
