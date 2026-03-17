#!/bin/bash

# python /home/jdelvecchio/image_segmentation/segmentation_gym/make_dataset_no_tkinter.py \
python /home/gaimholte/Imholte_segmentation_gym_ubuntu/segmentation_gym-main/make_dataset_no_tkinter.py \
 --output /home/jdelvecchio/image_segmentation/my_segmentation_gym_datasets/modelOut \
 --config /home/jdelvecchio/image_segmentation/my_segmentation_gym_datasets/config/hatteras_l8_segformer.json \
 --label_dir /home/jdelvecchio/image_segmentation/my_segmentation_gym_datasets/capehatteras_data/fromDoodler/labels \
 --image_dirs /home/jdelvecchio/image_segmentation/my_segmentation_gym_datasets/capehatteras_data/fromDoodler/images

python /home/jdelvecchio/image_segmentation/segmentation_gym/train_model_script_no_tkinter.py \
 --train_data_dir /home/jdelvecchio/image_segmentation/my_segmentation_gym_datasets/modelOut/train_data/train_npzs \
 --val_data_dir /home/jdelvecchio/image_segmentation/my_segmentation_gym_datasets/modelOut/val_data/val_npzs \
 --config_file /home/jdelvecchio/image_segmentation/my_segmentation_gym_datasets/modelOut \
 --config /home/jdelvecchio/image_segmentation/my_segmentation_gym_datasets/config/hatteras_l8_segformer.json



