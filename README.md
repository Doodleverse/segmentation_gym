# Coastal Image Segmentation Zoo

> Daniel Buscombe, Marda Science daniel@mardascience.com. Developed for the USGS Coastal Marine Geology program, as part of the Florence Supplemental project


A toolbox to segment imagery using a residual UNet model. This repository allows you to do three things:

* Use an existing (i.e. pre-trained) model to segment new sample imagery using provided model weights
* Create a tfrecords dataset to train a new model for a particular task
* Train a new model using your new tfrecords dataset

## Navigation

* [Residual U-Net model](#model)
* [Implementation](#implementation)
* [Installation](#install)
* [Provided Datasets](#data)
* [Use a Pre-Trained Residual UNet for Image Segmentation](#resunet)
* [Creation of `config` files for model retraining and training](#config)
* [Train a model for image segmentation using provided datasets](#retrain)
* [Train a model for image segmentation using your own dataset](#train)
* [Roadmap](#roadmap)


## <a name="model"></a>Residual U-Net model

UNet with residual (or lateral/skip connections). The UNet model framework is a type of fully convolutional neural network that is used for binary segmentation i.e foreground and background pixel-wise classification. It is easily adapted to multiclass segmentation workflows by representing each class as a binary mask, creating a stack of binary masks for each potential class (so-called one-hot encoded label data). A UNet is symmetrical (hence the U in the name) and uses concatenation instead of addition to merge feature maps.

The fully convolutional model framework consists of two parts, the encoder and the decoder. The encoder receives the N x N x M (M=1, 3 or 4 in this implementation) input image and applies a series of convolutional layers and pooling layers to reduce the spatial size and condense features. Six banks of convolutional filters, each using filters that double in size to the previous, thereby progressively downsampling the inputs as features are extracted through pooling. The last set of features (or so-called bottleneck) is a very low-dimensional feature representation of the input imagery. The decoder upsamples the bottleneck into a N x N x 1 label image progressively using six banks of convolutional filters, each using filters half in size to the previous, thereby progressively upsampling the inputs as features are extracted through transpose convolutions and concatenation. A transposed convolution convolves a dilated version of the input tensor, consisting of interleaving zeroed rows and columns between each pair of adjacent rows and columns in the input tensor, in order to upscale the output. The sets of features from each of the six levels in the encoder-decoder structure are concatenated, which allows learning different features at different levels and leads to spatially well-resolved outputs. The final classification layer maps the output of the previous layer to a single 2D output based on a sigmoid activation function. The difference between ours and the original implementation is in the use of three residual-convolutional encoding and decoding layers instead of regular six convolutional encoding and decoding layers. Residual or 'skip' connections have been shown in numerous contexts to facilitate information flow, which is why we have halved the number of convolutional layers but can still achieve good accuracy on the segmentation tasks. The skip connections essentially add the outputs of the regular convolutional block (sequence of convolutions and ReLu activations) with the inputs, so the model learns to map feature representations in context to the inputs that created those representations.

## <a name="implementation"></a>Implementation
Designed for 1,3, or 4-band imagery, and up to 4 classes

The program supports both `binary` (one class of interest and a null class) and `multiclass` (several classes of interest) image segmentation using a custom implementation of a Residual U-Net, a deep learning framework for image segmentation. The implementation is prescribed to facilitate a wide variety of workflows but is limited to up to 4 image bands and up to 4 classes. The latter limitation is because of the nature of the image file storage capabilities of png decoding being limited to 4 bands.

We write image datasets to tfrecord format files for 'analysis ready data' that is highly compressed and easy to share. One of the motivations for using TFRecords for data is to ensure a consistency in what images get allocated as training ad which get allocated as validation. These images are already randomized, and are not randomized further during training. Another advantage is the way in which it facilitates efficient data throughput from file to to GPU memory where the numerical calculations carried out during model training, and making out-of-memory errors caused by variation in data input and output throughput to and from the GPU. Protocol buffers are also a very good compression technique for sharing large amounts of labeled data.

* Images are augmented using Keras image augmentation generator functions
* Each augmented image is written to file (jpeg for images and png for labels - one image per band of the on-hot encoded label)
* Images are read back in and written to TF-Record format files
* Models are trained on the augmented data encoded in the tf-records, so the original data is a hold-out or test set. This is ideal because although the validation dataset (drawn from augmented data) doesn't get used to adjust model weights, it does influence model training by triggering early stopping if validation loss is not improving. Testing on an untransformed set is also a further check/reassurance of model performance and evaluation metric


## <a name="install"></a>Installation
I advise creating a new conda environment to run the program.

1. Clone the repo:

```
git clone --depth 1 https://github.com/dbuscombe-usgs/segmentation_zoo.git
```

(`--depth 1` means "give me only the present code, not the whole history of git commits" - this saves disk space, and time)

2. Create a conda environment called `imageseg`

```
conda env create --file install/imageseg.yml
conda activate imageseg
```

If you get errors associated with loading the model weights you may need to:

```
pip install 'h5py==2.10.0' --force-reinstall
```

and just ignore the errors.


## <a name="data"></a>Provided datasets

### Oblique aircraft coastal imagery (R, G, B)

This dataset is used here to demonstrate `binary segmentation` (i.e. 1 class of interest, and 1 null class). The classes are `water` and `null`

732 (as of 3/9/2021) images and associated binary (2-class land and water) masks. Prototype version, research in progress. Full version forthcoming

Thanks to Andy Ritchie and Jon Warrick for creating label images. Additional labels were created by Daniel Buscombe


### Nadir aircraft/UAV coastal imagery (R, G, B)

This dataset is used here to demonstrate `binary segmentation` (i.e. 1 class of interest, and 1 null class). The classes are `water` and `null`

2564 (as of 3/9/2021) images and associated binary (2-class land and water) masks. Prototype version, research in progress. Full version forthcoming

Thanks to Stephen Bosse, Jin-Si Over, Christine Kranenberg, Chris Sherwood, and Phil Wernette for creating label images. Additional labels were created by Daniel Buscombe


### Sentinel2 satellite coastal imagery (R, G, B)
This dataset is used here to demonstrate `multiclass segmentation` (i.e. more than 1 class of interest, and 1 null class). The classes are `blue water` (unbroken water), `white water` (active wave breaking), `wet sand` (swash, lower intertidal), and `dry land`

Labels were created by Daniel Buscombe. Prototype version (72 labeled images from Santa Cruz, CA), research in progress. Full version forthcoming


## <a name="resunet"></a>Use a Pre-Trained Residual UNet for Image Segmentation

1. Change directory to the `res_unet` directory (perhaps some day this repo will contain other models, in which case this top-level directory will contain additional folders)

```
cd res_unet
```

2. Run the program like so to use a model that you have weights for (either provided with this repository or generated yourself using a procedure described below) o a directory of images

```
python seg_images_in_folder.py
```

You will be prompted to select a weights file (with file extension `*.h5`), then a directory containing the images you wish to segment

When the program has completed, go to the folder of samples you asked the model to segment and you will see a model predictions as new images (*_predseg.png). If the segmentation is binary (i.e. NCLASSES = 1 in the config file), the program will additionally create a composite of the sample image and its estimated mask.


### Example: Watermasker for oblique aircraft coastal imagery (R, G, B)
* Select the weights file 'weights/oblique_coast_watermask/watermask_oblique_2class_batch_4.h5'

* Select the sample folder 'sample/oblique_coast_watermask', or whatever folder of appropriate you may have


### Example: Watermasker for nadir aircraft/UAV coastal imagery (R, G, B)
Forthcoming


### Example: Watermasker for Sentinel2 satellite coastal imagery (R, G, B)
Prototype version, research in progress

* Select the weights file 'weights/sentinel2_coast_watermask/s2_4class_batch_12.h5'

* Select the sample folder 'sample/sentinel2_coast_watermask', or whatever folder of appropriate you may have


## <a name="config"></a>Creation of `config` files for model retraining and training
Configuration or `config` files in [JSON format](https://en.wikipedia.org/wiki/JSON) where all relevant parameters are set. There are a few of them

An example config file (saved as `res_unet/model_training/config/oblique_coast_watermask/watermask_oblique_2class_batch_4.json`):

```
{
  "TARGET_SIZE": [768,1024],
  "KERNEL_SIZE": [7,7],
  "NCLASSES": 1,
  "BATCH_SIZE": 4,
  "N_DATA_BANDS": 3,
  "DO_CRF_REFINE": true,
  "DO_TRAIN": false,
  "PATIENCE": 25,
  "IMS_PER_SHARD": 50,
  "MAX_EPOCHS": 200,
  "VALIDATION_SPLIT": 0.75,
  "RAMPUP_EPOCHS": 20,
  "SUSTAIN_EPOCHS": 0.0,
  "EXP_DECAY": 0.9,
  "START_LR":  1e-7,
  "MIN_LR": 1e-7,
  "MAX_LR": 1e-5,
  "MEDIAN_FILTER_VALUE": 3,
  "DOPLOT": true,
  "ROOT_STRING": "watermask-oblique-data",
  "USEMASK": true,
  "AUG_ROT": 5,
  "AUG_ZOOM": 0.05,
  "AUG_WIDTHSHIFT": 0.05,
  "AUG_HEIGHTSHIFT": 0.05,
  "AUG_HFLIP": true,
  "AUG_VFLIP": true,
  "AUG_LOOPS": 1,
  "AUG_COPIES": 3 ,
  "REMAP_CLASSES": {"0": 0, "1": 0, "2": 0, "3":1, "4":1}
}
```

Notice the last entry does *NOT* have a comma. It does not matter what order the variables are specified as, but you must use the names of the variables exactly as is described here. A description of the variables is provided below


### Model training and retraining

#### General settings
* `DO_TRAIN`: (bool) `true` to retrain model from scratch. Otherwise, program will use existing model weights and evaluate the model based on the validation set
* `MEDIAN_FILTER_VALUE`: (integer) radius of disk used to apply median filter, if > 1
* `DOPLOT`: (bool) `true` to make plots
* `DO_CRF_REFINE`: (bool) `true` to apply CRF post-processing to model outputs

#### Model training
Model training and performance is sensitive to these hyperparameters. Use a `TARGET_SIZE` that makes sense for your problem, that conforms roughly with the dimensions of the imagery and labels you have for model training, and that fits in available GPU memory. You might be very surprised at the accuracy and utility of models trained with significantly downsized imagery

* `TARGET_SIZE`: list of integer image dimensions to write to tfrecord and use to build and use models. This doesn't have to be the sample image dimension (it would typically be significantly smaller due to memory constraints) but it should ideally have the same aspect ratio. The target size must be compatible with the cardinality of the model
* `KERNEL_SIZE`: (integer) convolution kernel dimension
* `NCLASSES`: (integer) number of classes (1 = binary e.g water/no water). For multiclass segmentations, enumerate the number of classes not including a null class
* `BATCH_SIZE`: (integer) number of images to use in a batch. Typically better to use larger batch sizes but also uses more memory
* `N_DATA_BANDS`: (integer) number of input image bands. Typically 3 (for an RGB image, for example) or 4 (e.g. near-IR or DEM, or other relevant raster info you have at coincident resolution and coverage)
* `PATIENCE`: (integer) the number of epochs with no improvement in validation loss to wait before exiting model training
* `IMS_PER_SHARD`: (integer) the number of images to encode in each tfrecord file
* `MAX_EPOCHS`: (integer) the maximum number of epochs to train the model over. Early stopping should ensure this maximum is never reached
* `VALIDATION_SPLIT`: (float) the proportion of the dataset to use for validation. The rest will be used for model training. Typically in the range 0.5 -- 0.9 for model training on large datasets

#### Learning rate scheduler
The model training script uses a learning rate scheduler to cycle through a range of learning rates at every training epoch using a prescribed function. Model training can sometimes be sensitive to the specification of these parameters, especially the `MAX_LR`, so be prepared to try a few values if the model is not performing optimally

* `RAMPUP_EPOCHS`: (integer) The number of epochs to increase from `START_LR` to `MAX_LR`
* `SUSTAIN_EPOCHS`: (float) The number of epochs to remain at `MAX_LR`
* `EXP_DECAY`: (float) The rate of decay in learning rate from `MAX_LR`
* `START_LR`: (float) The starting learning rate
* `MIN_LR`: (float) The minimum learning rate, usually equals `START_LR`, must be < `MAX_LR`
* `MAX_LR`: (float) The maximum learning rate, must be > `MIN_LR`

#### Label pre-processing (optional)
* `REMAP_CLASSES`: (dict) A dictionary of values in the data and what values you'd like to replace them with, for example `{"0": 0, "1": 0, "2": 0, "3":1, "4":1}` says "recode ones and twos as zeros and threes and fours as ones". Used to reclassify data on the fly without written new files to disk


### TF-Record dataset creation

#### General settings

* `ROOT_STRING`: (string) string to prepend the tfrecords with, if running `make_tfrecords.py`
* `USEMASK`: (bool) `true` if the files use 'mask' instead of 'label' in the folder/filename. if `false`, 'label' is assumed
* `IMS_PER_SHARD`: (integer) the number of images to encode in each tfrecord file

#### Image augmentation
This program is structured to carry out augmentation of labeled training/validation datasets. The program `make_tfrecords.py` first generates a new set of augmented imagery and encodes those data (only) into TFRecords. The model therefore is trained using the augmented data only; they are split into train and validation subsets. The original imagery is therefore free to be used as a 'hold-out' test set to further evaluate the performance of the model. Augmentation is designed to regularize the model (i.e. prevent it from overfitting) by transforming imagery and label pairs in random ways within limits. Those limits are set using the parameters below.

* `AUG_ROT`: (integer) the maximum amount of random image rotation in degrees, typically <10
* `AUG_ZOOM`: (float) the maximum amount of random image zoom as a proportion, typically <.2
* `AUG_WIDTHSHIFT`:  (float) the maximum amount of random horizontal shift, typically <.2
* `AUG_HEIGHTSHIFT`: (float) the maximum amount of random horizontal shift, typically <.2
* `AUG_HFLIP`: (bool) `true` to randomly horizontally flip the image
* `AUG_VFLIP`: (bool) `true` to randomly vertically flip the image  
* `AUG_LOOPS`: (integer) number of batches to use for augmented imagery generation (typically 1, >1 for RAM limited machines and large datasets)
* `AUG_COPIES`: (integer) number of augmented datasets to create. Each dataset will contain the same number of samples as in the original image set, typically <5


## <a name="retrain"></a>Train a model for image segmentation using provided datasets

This section is to retrain a model using one of the datasets provided


*Note*: you require an NVIDIA GPU with >6GB memory to train models from scratch using TFrecords

1. Change directory

```
cd res_unet/model_training
```

Make a configuration file that will contain all the information the program needs to train a model and subsequently use it for prediction

2. Run training like this

```
python train_resunet.py
```

### Example: Watermasker for oblique aircraft coastal imagery (R, G, B)

Change directory to the appropriate tf records directory and run the `download_tfrecords.py` script to get the data

```
cd res_unet/model_training/data/watermask_oblique_tfrecords
python download_tfrecords.py
cd ../..
```

Run the program

```
python train_resunet.py
```

* Select the config file `weights/sentinel2_coast_watermask/watermask_oblique_2class_batch_4.json`
* Select the tfrecords data folder `data/watermask_oblique_tfrecords`

The program will first print some example training and validation samples. See the file `watermask_oblique_2class_batch_4_train_sample_batch.png` and `watermask_oblique_2class_batch_4_val_sample_batch.png` in `model_training/examples/oblique_coast_watermask` directory.

Then the model will begin training. You will see output similar to ...

```
reating and compiling model ...
.....................................
Training model ...

Epoch 00001: LearningRateScheduler reducing learning rate to 1e-07.
Epoch 1/200
2021-03-03 11:47:03.934177: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2021-03-03 11:47:04.670713: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
75/75 [==============================] - 55s 733ms/step - loss: 0.5727 - mean_iou: 0.1747 - dice_coef: 0.4273 - val_loss: 0.5493 - val_mean_iou: 3.3207e-05 - val_dice_coef: 0.4507 - lr: 1.0000e-07

Epoch 00002: LearningRateScheduler reducing learning rate to 5.95e-07.
Epoch 2/200
75/75 [==============================] - 56s 745ms/step - loss: 0.5632 - mean_iou: 0.1840 - dice_coef: 0.4368 - val_loss: 0.6005 - val_mean_iou: 9.7821e-05 - val_dice_coef: 0.3995 - lr: 5.9500e-07

Epoch 00003: LearningRateScheduler reducing learning rate to 1.0900000000000002e-06.
Epoch 3/200
75/75 [==============================] - 56s 751ms/step - loss: 0.5403 - mean_iou: 0.2212 - dice_coef: 0.4597 - val_loss: 0.6413 - val_mean_iou: 8.7021e-04 - val_dice_coef: 0.3587 - lr: 1.0900e-06

(etc)
```

The above is for `MAX_EPOCHS`=200, `BATCH_SIZE`=4 (i.e. the default, set/changed in the config file `weights/sentinel2_coast_watermask/watermask_oblique_2class_batch_4.json`), and the default learning rate parameters.

When model training is complete, the model is evaluated on the validation set. You'll see output like this printed to screen

```
Epoch 00126: LearningRateScheduler reducing learning rate to 1.001552739802484e-07.
Epoch 126/200
75/75 [==============================] - 56s 750ms/step - loss: 0.1206 - mean_iou: 0.8015 - dice_coef: 0.8794 - val_loss: 0.1222 - val_mean_iou: 0.7998 - val_dice_coef: 0.8778 - lr: 1.0016e-07
.....................................
Evaluating model ...
225/225 [==============================] - 26s 117ms/step - loss: 0.1229 - mean_iou: 0.7988 - dice_coef: 0.8771
loss=0.1229, Mean IOU=0.7988, Mean Dice=0.8771
```

The IOU and Dice coefficients are accuracy metrics. The model then prints several model output examples from the validation set to files located in the `examples` folder (for this example, `res_unet/model_training/examples/oblique_coast_watermask`)



## <a name="train"></a>Train a model for image segmentation using your own datasets

*Note*: you require an NVIDIA GPU with >6GB memory to train models from scratch using TFrecords

### Example: Watermasker for oblique aircraft coastal imagery (R, G, B)

Example workflow using the data encoded in the tfrecords for the "oblique coastal watermasker" dataset below. Note that the most difficult aspect of this is usually creating the TF-REcords properly. You should be prepared to modify the `make_tfrecords.py` script to deal with specific data folder structures, and file naming conventions. I therefore urge you to prepare your data in a similar way to the provided examples

1. Change directory, download images

```
cd res_unet/model_training/data/watermask_oblique_jpegs
python download_jpegs.py
cd ../..
```

2. create TF-Records from your images/labels files

```
python make_tfrecords.py
```

This program will ask you for 4 things:
* the location of the `config` file
* the location of the folder of images (it expects a single subdirectory containing .jpg extension image files)
* the location of the folder of corresponding label images (it expects a single subdirectory containing .jpg extension image files)
* the location where to write tfrecord files for model training

It expects the same number of images and labels (i.e. you should provide 1 label image per image). It creates augmented images and corresponding labels in batches to save memory. Once all augmented images have been made, they are written to tfrecords. Note that the original images are not used in model training, only augmented images. That provides an opportunity to use the original image/label set as a hold-out or independent test set. All augmented images will have the same size, i.e. `TARGET_SIZE`. Augmentation is controlled by the `AUG_*` parameters in the `config` file. I advise you use only small shifts and zooms. Use vertical and horizontal flips often.


3. Create a config file for model training (see appropriate section above)

4. Run the following to train a new model on your data from scratch:

```
python train_resunet.py
```

(see above for explanation of this script/process). The model takes a long time to train using the `config` file `weights/sentinel2_coast_watermask/watermask_oblique_2class_batch_4.json` and the tfrecords in `data/watermask_oblique_tfrecords`. After possibly a few hours (depending on your GPU speed) the model training finishes with an output like this:

```
Epoch 00141: LearningRateScheduler reducing learning rate to 1.0003196953557818e-07.
Epoch 141/200
125/125 [==============================] - 92s 739ms/step - loss: 0.0791 - mean_iou: 0.8631 - dice_coef: 0.9209 - val_loss: 0.0751 - val_mean_iou: 0.8687 - val_dice_coef: 0.9249 - lr: 1.0003e-07
.....................................
Evaluating model on entire validation set ...
350/350 [==============================] - 42s 119ms/step - loss: 0.0751 - mean_iou: 0.8686 - dice_coef: 0.9249
loss=0.0751, Mean IOU=0.8686, Mean Dice=0.9249
sys:1: UserWarning: Possible precision loss converting image of type float32 to uint8 as required by rank filters. Convert manually using skimage.util.img_as_ubyte to silence this warning.
Mean IoU (validation subset)=0.855
```

The mean IOU is 86% and the mean Dice coefficient is 92% using 500 image files for training and 1400 for validation


## <a name="roadmap"></a>Roadmap

Plans:

* add "4D" satellite shorelines example
* add "3D" dune segmentation example
* add "4D" dune segmentation example
* continue to refine data and train models for existing dataset examples
* eventually ... some form of a graphical user interface
* adaptive learning: rank validation imagery by loss (label those with highest loss)
* add "2D" NIR sentinel exmaple
* more than 4 bands will require removing all decode_png and encode_png, replacing with a format that can support >4 bands
