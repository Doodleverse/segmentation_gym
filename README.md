# 📦 Segmentation Zoo
[![Last Commit](https://img.shields.io/github/last-commit/dbuscombe-usgs/segmentation_zoo)](
https://github.com/dbuscombe-usgs/segmentation_zoo/commits/main)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/dbuscombe-usgs/segmentation_zoo/graphs/commit-activity)
[![Wiki](https://img.shields.io/badge/wiki-documentation-forestgreen)](https://github.com/dbuscombe-usgs/segmentation_zoo/wiki)
![GitHub](https://img.shields.io/github/license/dbuscombe-usgs/segmentation_zoo)
[![Wiki](https://img.shields.io/badge/discussion-active-forestgreen)](hhttps://github.com/dbuscombe-usgs/segmentation_zoo/discussions)

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

![Zoo Logo](./zoo-logo.png)

## 🌟 Highlights

- Zoo is for training, evaluating, and deploying deep learning models for image segmentation
- We take transferability seriously; Zoo is designed to be a "one stop shop" for image segmentation on N-D imagery (well, at least Earth Observation and aerial remote sensing imagery)
- Zoo encodes relatively powerful models like UNets, and provides lots of ways to manipulate data, model training, and model architectures that should yield good results with some informed experimentation
- Zoo works seamlessly with [Doodler](https://github.com/dbuscombe-usgs/dash_doodler), a human-in-the loop labeling tool

## ℹ️ Overview

We are building a toolbox to segment imagery with a variety of supervised deep-learning models for image segmentation. Current work is focused on building a family of UNet models. We have built an end-to-end workflow that facilitates

* Preprocessing of imagery for deep learning model training and prediction, such as image padding and/or resizing
* Coupling of N-dimensional imagery, perhaps stored across multiple files, with corresponding integer label images
* Use of an existing (i.e. pre-trained) model to segment new imagery (by using provided code and model weights)
* Use of images and corresponding label images, or 'labels', to develop a 'model-ready' dataset
* Training a new model from scratch using this new dataset
* Evaluating the model against a validation subset
* Applying the model (or ensemble of models) on sample imagery, i.e. model deployment

We have tested on a variety of Earth and environmental imagery of coastal environments, which is the authors' motivation for creating this toolbox. However, we expect the toolbox to be useful for all types of imagery when properly applied.

This toolbox is designed to work seamlessly with [Doodler](https://github.com/dbuscombe-usgs/dash_doodler), a human-in-the loop labeling tool that will help you make training data for Zoo. It would also work on any imagery in jpg or png format that has corresponding 2d greyscale integer label images (jpg or png), however acquired.


### ✍️ Authors

Package maintainers:
* [@dbuscombe-usgs](https://github.com/dbuscombe-usgs)
* [@ebgoldstein](https://github.com/ebgoldstein)

Contributions:
* [@2320sharon](https://github.com/2320sharon)


## 🚀 Usage

This toolbox is designed for 1,3, or 4-band imagery, and supports both `binary` (one class of interest and a null class) and `multiclass` (several classes of interest).

We recommend a 6 part workflow:

1. Download & Install Zoo
2. Decide on which data to use and move them into the appropriate part of the Zoo [directory structure](#dir). *(We recommend that you first use the included data as a test of Zoo on your machine. After you have confirmed that this works, you can import your own data, or make new data using [Doodler](https://github.com/dbuscombe-usgs/dash_doodler))*
3. Write a `config` file for your data. You will need to make some decisions about the model and hyperparameters.
4. Run `make_dataset.py` to augment and package your images into npz files for training the model.  
5. Run `train_model.py` to train a segmentation model.
6. Run `seg_images_in_folder.py` to segment images with your newly trained model, or `ensemble_seg_images_in_folder.py` to point more than one trained model at the same imagery and ensemble the model outputs


* Here at Zoo HQ we advocate training models on the augmented data encoded in the datasets, so the original data is a hold-out or test set. This is ideal because although the validation dataset (drawn from augmented data) doesn't get used to adjust model weights, it does influence model training by triggering early stopping if validation loss is not improving. Testing on an untransformed set is also a further check/reassurance of model performance and evaluation metric

* Zoo HQ also advocates the use of `ensemble` models where possible, which requires training multiple models each with a config file, and model weights file


## ⬇️ Installation

We advise creating a new conda environment to run the program.

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


( add other minimum requirements like Python versions or operating systems)


## How to use
Check out the [wiki](https://github.com/dbuscombe-usgs/segmentation_zoo/wiki) for a guide of how to use Zoo

1. Organize your files according to [this guide](https://github.com/dbuscombe-usgs/segmentation_zoo/wiki/Directory-Structure-and-Tests)
2. Create a configuration file according to [this guide](https://github.com/dbuscombe-usgs/segmentation_zoo/wiki/Creation-of-%60config%60-files)
3. Create a model-ready dataset from your pairs of images and labels. A model-ready dataset is a set of images and corresponding labels in `.npz` format that contain all your data for model training and validation. We hope you find [this guide]() helpful
4. Train and evaluate an image segmentation model according to [this guide](https://github.com/dbuscombe-usgs/segmentation_zoo/wiki/Train-an-image-segmentation-model)
5. Deploy / evaluate model on unseen sample imagery  *more detail coming soon*


## 💭 Feedback and Contributing

Please read our [code of conduct](https://github.com/dbuscombe-usgs/segmentation_zoo/blob/main/CODE_OF_CONDUCT.md)

Please contribute to the [Discussions tab](https://github.com/dbuscombe-usgs/segmentation_zoo/discussions) - we welcome your ideas and feedback.

We also invite all to open issues for bugs/feature requests using the [Issues tab](https://github.com/dbuscombe-usgs/segmentation_zoo/issues)
