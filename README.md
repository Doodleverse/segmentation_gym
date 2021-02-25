# Segmentation Zoo

> Daniel Buscombe, Marda Science daniel@mardascience.com

> Developed for the USGS Coastal Marine Geology program, as part of the Florence Supplemental project


A place to segment imagery using a residual UNet model. This repository allows you to do three things:

* Use existing (pre-trained) model weights to segment new sample imagery
* Create a tfrecords dataset to train a new model for a particular task
* Train a new model using your new tfrecords dataset

* [Installation](#install)
* [Provided Datasets](#data)
* [Use a Pre-Trained Residual UNet for Image Segmentation](#resunet)
* [Train a model for image segmentation using provided datasets](#retrain)
* [Train a model for image segmentation using your own dataset](#train)
* [Roadmap](#roadmap)


## <a name="install"></a>Installation

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
Thanks to Andy Ritchie and Jon Warrick for creating label images. Additional labels were created by Daniel Buscombe

### Nadir aircraft/UAV coastal imagery (R, G, B)
Thanks to Stephen Bosse, Jin-Si Over, Christine Kranenberg, Chris Sherwood, and Phil Wernette for creating label images. Additional labels were created by Daniel Buscombe


### Sentinel2 satellite coastal imagery (R, G, B)
Labels were created by Daniel Buscombe



## <a name="resunet"></a>Use a Pre-Trained Residual UNet for Image Segmentation

1. Change directory to the `res_unet` directory (perhaps some day this repo will contain other models, in which case this top-level directory will contain additional folders)

```
cd res_unet
```

2. Run the program like so to use a model that you have weights for (either provided with this repository or generated yourself using a procedure described below) o a directory of images

```
python seg_images_in_folder.py
```

You will be prompted to select a weights file (with file extension "*.h5"), then a directory containing the images you wish to segment

When the program has completed, go to the folder of samples you asked the model to segment and you will see a model predictions as new images (*_predseg.png). If the segmentation is binary (i.e. NCLASSES = 1 in the config file), the program will additionally create a composite of the sample image and its estimated mask.


### Example: Watermasker for oblique aircraft coastal imagery (R, G, B)

Prototype version, research in progress

* Select the weights file 'weights/oblique_coast_watermask/watermask_oblique_2class_batch_4.h5'

* Select the sample folder 'sample/oblique_coast_watermask', or whatever folder of appropriate you may have



### Example: Watermasker for nadir aircraft/UAV coastal imagery (R, G, B)

Forthcoming



### Example: Watermasker for Sentinel2 satellite coastal imagery (R, G, B)

Prototype version, research in progress

* Select the weights file 'weights/sentinel2_coast_watermask/s2_4class_batch_12.h5'

* Select the sample folder 'sample/sentinel2_coast_watermask', or whatever folder of appropriate you may have


## <a name="retrain"></a>Train a model for image segmentation using provided datasets

This section is to retrain a model using one of the datasets provided

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

* Select the config file 'weights/sentinel2_coast_watermask/watermask_oblique_2class_batch_4.json'
* Select the tfrecords data folder 'data/watermask_oblique_tfrecords'


## <a name="train"></a>Train a model for image segmentation using your own datasets

Change directory, download images

```
cd res_unet/model_training/data/watermask_oblique_jpegs
python download_jpegs.py
cd ../..
```

create TF-Records from your images/labels files

```
python make_tfrecords.py
```



## <a name="train"></a>Train a Residual UNet for Image Segmentation

* add "4D" satellite shorelines example
* add "3D" dune segmentation example
* add "4D" dune segmentation example
* tfrecord generation workflows
