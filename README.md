# Segmentation Zoo
A collection of trained models for coastal image segmentation

* [Installation](#install)
* [Use a Pre-Trained Residual UNet for Image Segmentation](#resunet)
* [Train a Residual UNet for Image Segmentation](#train)
* [Roadmap](#roadmap)

## <a name="install"></a>Installation

Clone the repo:

```
git clone --depth 1 https://github.com/dbuscombe-usgs/segmentation_zoo.git
```

Create a conda environment called `imageseg`

```
conda env create --file install/imageseg.yml
conda activate imageseg
```

If you get errors associated with loading the model weights you may need to:

```
pip install 'h5py==2.10.0' --force-reinstall
```

and just ignore the errors. Worked for me anyway when I ran into it on a windows machine


## <a name="resunet"></a>Use a Pre-Trained Residual UNet for Image Segmentation

Usage:

```
cd res_unet
python seg_images_in_folder.py
```

### Watermasker for oblique aircraft coastal imagery (R, G, B)

Prototype version, research in progress

* Select the weights file 'weights/oblique_coast_watermask/watermask_oblique_2class_best_weights_batch_4.h5'

* Select the sample folder 'sample/oblique_coast_watermask', or whatever folder of appropriate you may have

### Watermasker for nadir aircraft/UAV coastal imagery (R, G, B)

Forthcoming

### Watermasker for Sentinel2 satellite coastal imagery (R, G, B)

Prototype version, research in progress

* Select the weights file 'weights/sentinel2_coast_watermask/s2_4class_best_weights_batch_12.h5'

* Select the sample folder 'sample/sentinel2_coast_watermask', or whatever folder of appropriate you may have


## <a name="train"></a>Train a Residual UNet for Image Segmentation


```
cd res_unet/model_training
python train_resunet.py
```


## <a name="train"></a>Train a Residual UNet for Image Segmentation

* add "4D" satellite shorelines example
* add "3D" dune segmentation example
* add "4D" dune segmentation example
* tfrecord generation workflows
