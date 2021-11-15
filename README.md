# Zoo (Coastal Image Segmentation Zoo)

*Warning* this is alpha software, i.e. not finished with several known bugs. Please be patient, thanks. We welcome pull requests and posting issues - please don't be shy!

![Zoo](./zoo-logo.png)

We are building a toolbox to segment imagery with a variety of models. Current work is focused on building a family of UNet models. This repository allows you to do three things:

* Use an existing (i.e. pre-trained) model to segment new imagery (by using provided code and model weights)
* Use images & masks to develp a 'model-ready' dataset
* Train a new model using this new dataset

This toolbox is designed to work seamlessly with [Doodler](https://github.com/dbuscombe-usgs/dash_doodler), a human-in-the loop labeling tool that will help you make training data for Zoo.


## Table of Contents:

### The basics:

* [Generic Workflow](#workflow)
* [Models Included in this toolbox](#model)
* [Provided Datasets](#data)

### Code use:

* [Installation](#install)
* [Directory Structure and Tests](#dir)
* [Creation of `config` files for model retraining and training](#config)
* [Train an image segmentation model using the provided dataset](#retrain)
* [Train an image segmentation model using your own dataset](#newdata)
* [Changelog](#changelog)

## <a name="workflow"></a>Generic workflow
This toolbox is designed for 1,3, or 4-band imagery, and supports both `binary` (one class of interest and a null class) and `multiclass` (several classes of interest).

We recommend a 6 part workflow:

1. Download & Install Zoo
2. Decide on which data to use and move them into the appropriate part of the Zoo [directory structure](#dir). *(We recommend that you first use the included data as a test of Zoo on your machine. After you have confirmed that this works, you can import your own data, or make new data using [Doodler](https://github.com/dbuscombe-usgs/dash_doodler))*
3. Write a `config` file for your data. You will need to make some decisions about the model and hyperparameters.
4. Run `make_dataset.py` to augment and package your images into npz files for training the model.  
5. Run `train_model.py` to train a segmentation model.
6. Run `seg_images_in_folder.py` to segment images with your newly trained model, or `ensemble_seg_images_in_folder.py` to point more than one trained model at the same imagery and ensemble the model outputs


* Here at Zoo HQ we advocate training models on the augmented data encoded in the datasets, so the original data is a hold-out or test set. This is ideal because although the validation dataset (drawn from augmented data) doesn't get used to adjust model weights, it does influence model training by triggering early stopping if validation loss is not improving. Testing on an untransformed set is also a further check/reassurance of model performance and evaluation metric

* Here ate Zoo HQ we advocate use of `ensemble` models where possible, which requires training multiple models each with a config file, and model weights file

## <a name="model"></a>Models

There are currently 5 models included in this toolbox: a 2 [UNets](unet), 2 [Residual UNets](resunet), and a [Satellite UNet](satunet).

*Note that the Residual UNet is a new model, and will be described more fully in a forthcoming paper.*

### <a name="unet"></a>UNet model

The [UNet model](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) is a fully convolutional neural network that is used for binary segmentation i.e foreground and background pixel-wise classification. It is easily adapted to multiclass segmentation workflows by representing each class as a binary mask, creating a stack of binary masks for each potential class (so-called one-hot encoded label data). A UNet is symmetrical (hence the U in the name) and uses concatenation instead of addition to merge feature maps.

The fully convolutional model framework consists of two parts, the encoder and the decoder. The encoder receives the N x N x M (M=1, 3 or 4 in this implementation) input image and applies a series of convolutional layers and pooling layers to reduce the spatial size and condense features. Six banks of convolutional filters, each using filters that double in size to the previous, thereby progressively downsampling the inputs as features are extracted through pooling. The last set of features (or so-called bottleneck) is a very low-dimensional feature representation of the input imagery. The decoder upsamples the bottleneck into a N x N x 1 label image progressively using six banks of convolutional filters, each using filters half in size to the previous, thereby progressively upsampling the inputs as features are extracted through transpose convolutions and concatenation. A transposed convolution convolves a dilated version of the input tensor, consisting of interleaving zeroed rows and columns between each pair of adjacent rows and columns in the input tensor, in order to upscale the output. The sets of features from each of the six levels in the encoder-decoder structure are concatenated, which allows learning different features at different levels and leads to spatially well-resolved outputs. The final classification layer maps the output of the previous layer to a single 2D output based on a sigmoid activation function.

There are two options with the Unet architecture in this repository: a simple version and a highly configurable version... *more detail coming soon*

### <a name="resunet"></a>Residual UNet model
UNet with residual (or lateral/skip connections).

![Res-UNet](./unet/res-unet-diagram.png)

 The difference between our Res Unet and the original UNet is in the use of three residual-convolutional encoding and decoding layers instead of regular six convolutional encoding and decoding layers. Residual or 'skip' connections have been shown in numerous contexts to facilitate information flow, which is why we have halved the number of convolutional layers but can still achieve good accuracy on the segmentation tasks. The skip connections essentially add the outputs of the regular convolutional block (sequence of convolutions and ReLu activations) with the inputs, so the model learns to map feature representations in context to the inputs that created those representations.

There are two options with the Res-Unet architecture in this repository: a simple version and a highly configurable version... *more detail coming soon*

### <a name="satunet"></a>Satellite UNet model

[Satellite Unet](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/)
*Coming Soon*


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
pip install "h5py==2.10.0" --force-reinstall
```

and just ignore any errors.

Also, tensorflow version 2.2.0 or higher is now required, which means you may need to

```
pip install tensorflow-gpu=2.2.0 --user
```

and just ignore any errors. When you run any script, the tensorflow version should be printed to screen.


## <a name="dir"></a>Directory Structure and Tests

After Zoo downloads, we recommend you make a directory structure that mirrors below, *entirely separate from* the src directory contains the Zoo source code. A config folder will store `*.json` config files, which determine how the dataset is made and the model is trained. The data folder stores data used to make a zoo-compatible dataset (i.e., images & labels), and also serves as a useful directory for storing model ready data (which is *.npz) and images that are used by a trained model for prediction. The weights folder holds *.h5 files _ the weights and biases for your trained tensorflow model.


```{sh}
/Users/Someone/my_segmentation_zoo_datasets
                    │   ├── config
                    │   |    └── *.json
                    │   ├── data
                    |   |   ├── fromDoodler
                    |   |   |     ├──images
                    │   |   |     └──labels
                    |   |   ├──npzForModel
                    │   |   └──toPredict
                    │   ├── modelOut
                    │   └── weights
                    │       └── *.h5

```

A full dataset is available from [here](https://github.com/dbuscombe-usgs/segmentation_zoo/releases/download/v0.0.4-testdata/my_segmentation_zoo_datasets.zip) that is organized in the above way. We strongly suggest downloading and testing Zoo with this dataset before attempting to use Zoo with your own data.

We strongly recomment that you Download this dataset, unzip it, and test Zoo with it. Once you have confirmed that Zoo works with this test data, you can then go on to test zoo with your own data with the confidence tyhat Zoo works on your machine. This sampel data is also useful as a template for your own data, which must be organized in the same way as the test data.

## <a name="config"></a>Creation of `config` files for model training
Configuration or `config` files are [JSON format](https://en.wikipedia.org/wiki/JSON) and are the place where all relevant parameters are set.

An example config file:

```
{
    "TARGET_SIZE": [768,768],
    "MODEL": "resunet",
    "NCLASSES": 4,
    "BATCH_SIZE": 7,
    "N_DATA_BANDS": 3,
    "DO_TRAIN": true,
    "PATIENCE": 10,
    "MAX_EPOCHS": 100,
    "VALIDATION_SPLIT": 0.2,
    "FILTERS":8,
    "KERNEL":7,
    "STRIDE":1,
    "DROPOUT":0.1,
    "DROPOUT_CHANGE_PER_LAYER":0.0,
    "DROPOUT_TYPE":"standard",
    "USE_DROPOUT_ON_UPSAMPLING":false,
    "ROOT_STRING": "hatteras_l8_aug_768",
    "FILTER_VALUE": 3,
    "DOPLOT": true,
    "USEMASK": false,
    "RAMPUP_EPOCHS": 10,
    "SUSTAIN_EPOCHS": 0.0,
    "EXP_DECAY": 0.9,
    "START_LR":  1e-7,
    "MIN_LR": 1e-7,
    "MAX_LR": 1e-4,
    "AUG_ROT": 0,
    "AUG_ZOOM": 0.05,
    "AUG_WIDTHSHIFT": 0.05,
    "AUG_HEIGHTSHIFT": 0.05,
    "AUG_HFLIP": false,
    "AUG_VFLIP": false,
    "AUG_LOOPS": 1,
    "AUG_COPIES": 3
  }
```

Notice the last entry does *NOT* have a comma. It does not matter what order the variables are specified as, but you must use the names of the variables exactly as is described here. A description of the variables is provided below


### Model Description configs:
* `TARGET_SIZE`: list of integer image dimensions to write to dataset and use to build and use models. This doesn't have to be the sample image dimension (it would typically be significantly smaller due to memory constraints) but it should ideally have the same aspect ratio. The target size must be compatible with the cardinality of the model. Use a `TARGET_SIZE` that makes sense for your problem, that conforms roughly with the dimensions of the imagery and labels you have for model training, and that fits in available GPU memory. You might be very surprised at the accuracy and utility of models trained with significantly downsized imagery.
* `MODEL` : (string) specify which model you want to use, options are "unet","resunet", "simple_unet", "simple_resunet", and "satunet".
* `NCLASSES`: (integer) number of classes (1 = binary e.g water/no water). For multiclass segmentations, enumerate the number of classes not including a null class. For example, for 4 classes, use `NCLASSES`=4
* `BATCH_SIZE`: (integer) number of images to use in a batch. Typically better to use larger batch sizes but also uses more memory
* `N_DATA_BANDS`: (integer) number of input image bands. Typically 3 (for an RGB image, for example) or 4 (e.g. near-IR or DEM, or other relevant raster info you have at coincident resolution and coverage). Currently cannot be more than 4.
* `DO_TRAIN`: (bool) `true` to retrain model from scratch. Otherwise, program will use existing model weights and evaluate the model based on the validation set

### Model Training configs:

* `PATIENCE`: (integer) the number of epochs with no improvement in validation loss to wait before exiting model training
* `MAX_EPOCHS`: (integer) the maximum number of epochs to train the model over. Early stopping should ensure this maximum is never reached
* `VALIDATION_SPLIT`: (float) the proportion of the dataset to use for validation. The rest will be used for model training. Typically in the range 0.5 -- 0.9 for model training on large datasets
* `LOSS`: one of `cat` (categorical cross-entropy), `dice` (Dice loss), `hinge` (hinge loss), or `kld` (Kullback-Leibler divergence)

### Model Architecture configs:
* `FILTERS` : (integer) number of initial filters per convolutional block, doubled every layer
* `KERNEL` : (integer) the size of the Conv kernel
* `STRIDE` : (integer) the Conv stride
* `DROPOUT` : (integer) the fraction of dropout.
* `DROPOUT_CHANGE_PER_LAYER` : (integer) changes dropout by addition/ subtraction on encoder/decoder layers
* `DROPOUT_TYPE` : (string) "standard" or "spatial"
* `USE_DROPOUT_ON_UPSAMPLING` : (bool) if True, dropout is used on upsampling, otherwise it is not

### General configs
* `ROOT_STRING`: (string) the prefix used when writing data for use with the model e.g., "coastal_5class_",
* `FILTER_VALUE`: (integer) radius of disk used to apply median filter, if > 1
* `DOPLOT`: (bool) `true` to make plots
* `USEMASK`: (bool) `true` if the files use 'mask' instead of 'label' in the folder/filename. if `false`, 'label' is assumed
* `SET_GPU`: (int; optional) for machines with mutiple GPUs, this sets the GPU to use (note that GPU count begins with 0).

### Learning rate scheduler configs:
The model training script uses a learning rate scheduler to cycle through a range of learning rates at every training epoch using a prescribed function. Model training can sometimes be sensitive to the specification of these parameters, especially the `MAX_LR`, so be prepared to try a few values if the model is not performing optimally

* `RAMPUP_EPOCHS`: (integer) The number of epochs to increase from `START_LR` to `MAX_LR`
* `SUSTAIN_EPOCHS`: (float) The number of epochs to remain at `MAX_LR`
* `EXP_DECAY`: (float) The rate of decay in learning rate from `MAX_LR`
* `START_LR`: (float) The starting learning rate
* `MIN_LR`: (float) The minimum learning rate, usually equals `START_LR`, must be < `MAX_LR`
* `MAX_LR`: (float) The maximum learning rate, must be > `MIN_LR`


## Dataset creation and Image augmentation configs:
This program is structured to carry out augmentation of labeled training/validation datasets. The program `make_dataset.py` first generates a new set of augmented imagery and encodes those data (only) into datasets. The model therefore is trained using the augmented data only; they are split into train and validation subsets. The original imagery is therefore free to be used as a 'hold-out' test set to further evaluate the performance of the model. Augmentation is designed to regularize the model (i.e. prevent it from overfitting) by transforming imagery and label pairs in random ways within limits. Those limits are set using the parameters below.

* `AUG_ROT`: (integer) the maximum amount of random image rotation in degrees, typically <10
* `AUG_ZOOM`: (float) the maximum amount of random image zoom as a proportion, typically <.2
* `AUG_WIDTHSHIFT`:  (float) the maximum amount of random horizontal shift, typically <.2
* `AUG_HEIGHTSHIFT`: (float) the maximum amount of random horizontal shift, typically <.2
* `AUG_HFLIP`: (bool) `true` to randomly horizontally flip the image
* `AUG_VFLIP`: (bool) `true` to randomly vertically flip the image  
* `AUG_LOOPS`: (integer) number of batches to use for augmented imagery generation (>=2)
* `AUG_COPIES`: (integer) number of augmented datasets to create. Each dataset will contain the same number of samples as in the original image set, typically 2--10
* `REMAP_CLASSES`: (dict; optional) A dictionary of values in the data and what values you'd like to replace them with, for example `{"0": 0, "1": 0, "2": 0, "3":1, "4":1}` says "recode ones and twos as zeros and threes and fours as ones". Used to reclassify data on the fly without written new files to disk

## <a name="retrain"></a>Train an image segmentation model using provided datasets

This section is to retrain a model using the provided datasets.

*Note*: you require an NVIDIA GPU with >6GB memory to train models from scratch using datasets

1. Make sure you have the dataset and the config file, which should be in  `/Users/Someone/my_segmentation_zoo_datasets` in the appropraite directories.

2. Now navigate to the directory with the code ( ` cd /segmentation_zoo/unet` and train the model with:

```
python train_model.py
```

You will be prompted via a GUI to provide the `config` file, images, and labels. Then the program will print some example training and validation samples in a `sample/` folder in the directory with the data.

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

## <a name="newdata"></a>Train a model for image segmentation using your own data.

*Coming Soon*


## <a name="changelog"></a>CHANGELOG


## version 0.0.1, 06/10/21
* fixed some bugs in make_datasets.py
* first named version before watermasking dev branch
* for 1-class problems, creates 4-band image (band 4 is the probability of clas)

##version 0.0.2, 09/03/21
* fixed some bugs in make_datasets.py
* no median filter on 2d label image, now a morphology holes/islands on the one-hot stack (much better)
* removed CRF pre or post processing option

##version 0.0.4, 10/07/21
* no USE_LOCATION
* 1 example dataset, shipped separately
* code cleaned, sharing all functions through `imports.py`
* seg-images-in-folder working ok for multiclass Imagery
* code and directory structure greatly simplified
* standarized imagery is now [-1, 1], rather than [0,1] - old models will break with the new implementation - you should retrain your old model


##version 0.0.5, 10/21/21
* implements 3 unets, in a consistent way, and with much bigger options
* fixes bug in prior resunet implementation that used BATCH size for the number of filters
* vanilla unet , residual unet, and 'satellite unet'
* satellite unet is a reworking of  https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/
* nicer color overlay outputs in training script
* imports and model_imports are separated, the former for non-tf/keras, and the latter for tf/keras
* cleaned up import functions and and called dependencies

##version 0.0.6, 10/21/28
* re-implements res-unet and vanilla unet, in a consistent way, and with much bigger options, but with the original keras codes for model arechitecture, facilitating:
  * any kernel size up to 7x7 or perhaps more?
  * any stride size
  * with dropout
  * upsampling deconv and simple
* previous unets from 0.0.5 are kept, called 'satunet,' 'simple_unet,' and 'simple_resunet'
