
import os, shutil
from glob import glob

label_files = sorted(glob("G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\labels.orig\\*.tif"))

orig_files = sorted(glob("G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\images.orig\\*.tif"))

good_files = sorted(glob("G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\overlays\\good\\*.png"))

orig_label_dir = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\labels.orig"

orig_image_dir = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\images.orig"
good_image_dir = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\overlays\\good"
target_dir_images = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\images"
target_dir_labels = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\labels"

for k in good_files:
    shutil.copyfile(k.replace(good_image_dir, orig_image_dir).replace(".png", ".tif"), k.replace(good_image_dir, target_dir_images).replace(".png", ".tif"))
    shutil.copyfile(k.replace(good_image_dir, orig_label_dir).replace(".png", ".tif"), k.replace(good_image_dir, target_dir_labels).replace(".png", ".tif"))
