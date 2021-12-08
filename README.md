# 📦 Segmentation Zoo

(add your badges here)

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
* @dbuscombe-usgs
* @ebgoldstein

Contributions:
* @2320sharon


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

* Zoo HQ also advocate use of `ensemble` models where possible, which requires training multiple models each with a config file, and model weights file

### <a name="model"></a>Models

There are currently 5 models included in this toolbox: a 2 [UNets](unet), 2 [Residual UNets](resunet), and a [Satellite UNet](satunet).

*Note that the Residual UNet is a new model, and will be described more fully in a forthcoming paper.*

#### <a name="unet"></a>UNet model

The [UNet model](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) is a fully convolutional neural network that is used for binary segmentation i.e foreground and background pixel-wise classification. It is easily adapted to multiclass segmentation workflows by representing each class as a binary mask, creating a stack of binary masks for each potential class (so-called one-hot encoded label data). A UNet is symmetrical (hence the U in the name) and uses concatenation instead of addition to merge feature maps.

The fully convolutional model framework consists of two parts, the encoder and the decoder. The encoder receives the N x N x M (M=1, 3 or 4 in this implementation) input image and applies a series of convolutional layers and pooling layers to reduce the spatial size and condense features. Six banks of convolutional filters, each using filters that double in size to the previous, thereby progressively downsampling the inputs as features are extracted through pooling. The last set of features (or so-called bottleneck) is a very low-dimensional feature representation of the input imagery. The decoder upsamples the bottleneck into a N x N x 1 label image progressively using six banks of convolutional filters, each using filters half in size to the previous, thereby progressively upsampling the inputs as features are extracted through transpose convolutions and concatenation. A transposed convolution convolves a dilated version of the input tensor, consisting of interleaving zeroed rows and columns between each pair of adjacent rows and columns in the input tensor, in order to upscale the output. The sets of features from each of the six levels in the encoder-decoder structure are concatenated, which allows learning different features at different levels and leads to spatially well-resolved outputs. The final classification layer maps the output of the previous layer to a single 2D output based on a sigmoid activation function.

There are two options with the Unet architecture in this repository: a simple version and a highly configurable version... *more detail coming soon*

#### <a name="resunet"></a>Residual UNet model
UNet with residual (or lateral/skip connections).

![Res-UNet](./unet/res-unet-diagram.png)

 The difference between our Res Unet and the original UNet is in the use of three residual-convolutional encoding and decoding layers instead of regular six convolutional encoding and decoding layers. Residual or 'skip' connections have been shown in numerous contexts to facilitate information flow, which is why we have halved the number of convolutional layers but can still achieve good accuracy on the segmentation tasks. The skip connections essentially add the outputs of the regular convolutional block (sequence of convolutions and ReLu activations) with the inputs, so the model learns to map feature representations in context to the inputs that created those representations.

There are two options with the Res-Unet architecture in this repository: a simple version and a highly configurable version... *more detail coming soon*

#### <a name="satunet"></a>Satellite UNet model

[Satellite Unet](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/)
*Coming Soon*


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

1. Organize your files according to [this guide](https://github.com/dbuscombe-usgs/segmentation_zoo/wiki/Directory-Structure-and-Tests)
2. Create a configuration file according to [this guide](https://github.com/dbuscombe-usgs/segmentation_zoo/wiki/Creation-of-%60config%60-files)
3. Train and evaluate an image segmentation model according to [this guide](https://github.com/dbuscombe-usgs/segmentation_zoo/wiki/Train-an-image-segmentation-model)
4. Deploy / evaluate model on unseen sample imagery  *more detail coming soon*


## 💭 Feedback and Contributing

Please read our [code of conduct](https://github.com/dbuscombe-usgs/segmentation_zoo/blob/main/CODE_OF_CONDUCT.md)

Please contribute to the [Discussions tab](https://github.com/dbuscombe-usgs/segmentation_zoo/discussions) - we welcome your ideas and feedback.

We also invite all to open issues for bugs/feature requests using the [Issues tab](https://github.com/dbuscombe-usgs/segmentation_zoo/issues)
