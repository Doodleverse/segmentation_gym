# 📦 Segmentation Gym :muscle:
[![Last Commit](https://img.shields.io/github/last-commit/Doodleverse/segmentation_gym)](
https://github.com/Doodleverse/segmentation_gym/commits/main)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Doodleverse/segmentation_gym/graphs/commit-activity)
[![Wiki](https://img.shields.io/badge/wiki-documentation-forestgreen)](https://github.com/Doodleverse/segmentation_gym/wiki)
![GitHub](https://img.shields.io/github/license/Doodleverse/segmentation_gym)
[![Wiki](https://img.shields.io/badge/discussion-active-forestgreen)](https://github.com/Doodleverse/segmentation_gym/discussions)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

<!-- ![](https://user-images.githubusercontent.com/3596509/153691733-1fe98e37-5379-4122-8d02-adbcb0ab0db3.png) -->
![gym](https://user-images.githubusercontent.com/3596509/153696396-0b3148c5-77e4-48b2-b3ce-fd9038ba21ab.png)

## 🌟 Highlights

- Gym is for training, evaluating, and deploying deep learning models for image segmentation
- We take transferability seriously; Gym is designed to be a "one stop shop" for image segmentation on N-D imagery (i.e. any number of coincident bands). It is tailored to Earth Observation and aerial remote sensing imagery.
- Gym encodes relatively powerful models like UNets, and provides lots of ways to manipulate data, model training, and model architectures that should yield good results with some informed experimentation
- Gym works seamlessly with [Doodler](https://github.com/Doodleverse/dash_doodler), a human-in-the loop labeling tool that will help you make training data for Gym. 
- It would also work on any imagery in jpg or png format that has corresponding 2d greyscale integer label images (jpg or png), however acquired.
- Gym implements models based on the U-Net. Despite being one of the "original" deep learning segmentation models (dating to [2016](https://arxiv.org/abs/1505.04597)), UNets have proven themselves enormously flexible for a wide range of image segmentation tasks and spatial regression tasks in the natural sciences. So, we expect these models, and, perhaps more importantly, the training and implementation of those models in an end-to-end pipeline, to work for a very wide variety of cases. Additional models may be added later.
- You can read more about the models [here](https://github.com/Doodleverse/segmentation_gym/wiki/Models-in-Zoo) but be warned! We at Doodleverse HQ have discovered - often the hard way - that success is more about the data than the model. Gym helps you wrangle and tame your data, and makes your data work hard for you (nothing fancy, we just use augmentation)

## ℹ️ Overview

Gym is a toolbox to segment imagery with a variety of a family of UNet models, which are supervised deep-learning models for image segmentation. Gym supports segmentation of image with any number of bands, and any number of classes (memory limited). We have built an end-to-end workflow that facilitates a fully reproducible label-to-model workflow when used in conjunction with companion program [Doodler](https://github.com/Doodleverse/dash_doodler), however pairs of images and corresponding labels however-acquired may be used with Gym.

* Preprocessing of imagery for deep learning model training and prediction, such as image padding and/or resizing
* Coupling of N-dimensional imagery, perhaps stored across multiple files, with corresponding integer label images
* Use of an existing (i.e. pre-trained) model to segment new imagery (by using provided code and model weights)
* Use of images and corresponding label images, or 'labels', to develop a 'model-ready' dataset. A model-ready dataset is a set of images and corresponding labels in a serial binary archive format (we use `.npz`) that contain all your data for model training and validation, and that can be unpacked directory as tensorflow tensors. We initially used tfrecord format files, but abandoned the approach because of the relative complexity, and because the npz format is more familiar to Earth scientists who code with python.
* Training a new model from scratch using this new dataset
* Evaluating the model against a validation subset
* Applying the model (or ensemble of models) on sample imagery, i.e. model deployment

We have tested on a variety of Earth and environmental imagery of coastal, river, and other natural environments. However, we expect the toolbox to be useful for all types of imagery when properly applied.

### ✍️ Authors

Package maintainers:
* [@dbuscombe-usgs](https://github.com/dbuscombe-usgs)
* [@ebgoldstein](https://github.com/ebgoldstein)

Contributions:
* [@2320sharon](https://github.com/2320sharon)
* `doodleverse_utils` functions in `model_metrics.py` use minimally modified code from [here](https://github.com/zhiminwang1/Remote-Sensing-Image-Segmentation/blob/master/seg_metrics.py)

## 🚀 Usage

This toolbox is designed for 1,3, or 4-band imagery, and supports both `binary` (one class of interest and a null class) and `multiclass` (several classes of interest).

We recommend a 6 part workflow:

1. Download & Install Gym
2. Decide on which data to use and move them into the appropriate part of the Gym [directory structure](#dir). *(We recommend that you first use the included data as a test of Gym on your machine. After you have confirmed that this works, you can import your own data, or make new data using [Doodler](https://github.com/Doodleverse/dash_doodler))*
3. Write a `config` file for your data. You will need to make some decisions about the model and hyperparameters.
4. Run `make_dataset.py` to augment and package your images into npz files for training the model.  
5. Run `train_model.py` to train a segmentation model.
6. Run `seg_images_in_folder.py` to segment images with your newly trained model, or `ensemble_seg_images_in_folder.py` to point more than one trained model at the same imagery and ensemble the model outputs


* Here at Doodleverse HQ we advocate training models on the augmented data encoded in the datasets, so the original data is a hold-out or test set. This is ideal because although the validation dataset (drawn from augmented data) doesn't get used to adjust model weights, it does influence model training by triggering early stopping if validation loss is not improving. Testing on an untransformed set is also a further check/reassurance of model performance and evaluation metric

* Doodleverse HQ also advocates the use of `ensemble` models where possible, which requires training multiple models each with a config file, and model weights file


## ⬇️ Installation

We advise creating a new conda environment to run the program.

1. Clone the repo:

```
git clone --depth 1 https://github.com/Doodleverse/segmentation_gym.git
```

(`--depth 1` means "give me only the present code, not the whole history of git commits" - this saves disk space, and time)

2. Create a conda environment called `gym`

First, and optionally, you may want to do some conda housekeeping (recommended)

```
conda update conda
conda clean --all
```

Then:

```
conda env create --file install/gym.yml
conda activate gym
```

Alternatively, you could install using mamba, which should be significantly faster

```
conda install mamba -c conda-forge
mamba env create --file install/gym.yml
conda activate gym
```

If you get errors associated with loading the model weights you may need to:

```
pip install "h5py==2.10.0" --force-reinstall
```

and just ignore any errors.


## How to use
Check out the [wiki](https://github.com/dbuscombe-usgs/segmentation_gym/wiki) for a guide of how to use Gym

1. Organize your files according to [this guide](https://github.com/Doodleverse/segmentation_gym/wiki/Directory-Structure-and-Tests)
2. Create a configuration file according to [this guide](https://github.com/Doodleverse/segmentation_gym/wiki/Creation-of-%60config%60-files)
3. Create a model-ready dataset from your pairs of images and labels. We hope you find [this guide](https://github.com/Doodleverse/segmentation_gym/wiki/Create-a-model-ready-dataset) helpful
4. Train and evaluate an image segmentation model according to [this guide](https://github.com/Doodleverse/segmentation_gym/wiki/Train-an-image-segmentation-model)
5. Deploy / evaluate model on unseen sample imagery  *more detail coming soon*

## Test Dataset

A test data set, including a set of images/labels, model config files, and a dataset and models created with Gym, are available [here](https://zenodo.org/record/5895128/files/hatteras_RGB_zenodo_data_release_jan2022.zip?download=1) and [described on the zenodo page](https://zenodo.org/record/5895128#.Ye4AgPuIZH4)


## 💭 Feedback and Contributing

Please read our [code of conduct](https://github.com/Doodleverse/segmentation_gym/blob/main/CODE_OF_CONDUCT.md)

Please contribute to the [Discussions tab](https://github.com/Doodleverse/segmentation_gym/discussions) - we welcome your ideas and feedback.

We also invite all to open issues for bugs/feature requests using the [Issues tab](https://github.com/Doodleverse/segmentation_gym/issues)
