
import os, shutil
from glob import glob
from skimage.io import imread, imsave
import numpy as np


label_files = sorted(glob("G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\labels_all\\*.tif"))

orig_label_dir = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\labels_all"
orig_image_dir = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\images_all"

target_dir_images = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\images"
target_dir_labels = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\labels"

for k in label_files:

    lab = imread(k)
    if np.all( np.unique(lab)==np.array([0,1,2,3,4])):
        shutil.copyfile(k,k.replace(orig_label_dir, target_dir_labels))
        shutil.copyfile(k.replace(orig_label_dir,orig_image_dir),k.replace(orig_label_dir, target_dir_images))



