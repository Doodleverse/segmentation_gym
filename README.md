# Segmentation Zoo
A collection of trained models for coastal image segmentation

* [Installation](#install)
* [Residual UNet for Image Segmentation](#resunet)


## <a name="install"></a>Installation

```
conda env create --file install/imageseg.yml
conda activate imageseg
```


## <a name="resunet"></a>Residual UNet for Image Segmentation

Usage:

```
cd res_unet
python seg_images_in_folder.py
```

### Watermasker for oblique aircraft coastal imagery (R, G, B)

Prototype version, research in progress

Select the weights file 'weights/oblique_coast_watermask/watermask_oblique_2class_best_weights_batch_4.h5'

Select the sample folder 'sample/oblique_coast_watermask', or whatever folder of appropriate you may have

### Watermasker for nadir aircraft/UAV coastal imagery (R, G, B)

Forthcoming

### Watermasker for Sentinel2 satellite coastal imagery (R, G, B)

Prototype version, research in progress

Select the weights file 'weights/sentinel2_coast_watermask/s2_4class_best_weights_batch_12.h5'

Select the sample folder 'sample/sentinel2_coast_watermask', or whatever folder of appropriate you may have
