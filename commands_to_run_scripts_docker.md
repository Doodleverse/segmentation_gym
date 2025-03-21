# Commands to train model
Guide : https://github.com/Doodleverse/segmentation_gym/wiki/02_Case-Study-Demo

## Train Model
python train_model_script_no_tkinter.py --config_file /gym/my_segmentation_gym_datasets_v5/config/hatteras_l8_resunet.json --train_data_dir /gym/model_from_scratch_test_v5/train_data/train_npzs --val_data_dir /gym/model_from_scratch_test_v5/val_data/val_npzs

    Parameters
    --config_file
    Description: Path to the configuration JSON file that defines the model architecture and training settings.
    Example: --config_file /gym/my_segmentation_gym_datasets_v5/config/hatteras_l8_resunet.json

    --train_data_dir
    Description: Directory where the training data (e.g., npz files) is stored.
    Example: --train_data_dir /gym/model_from_scratch_test_v5/train_data/train_npzs

    --val_data_dir
    Description: Directory where the validation data is stored.
    Example: --val_data_dir /gym/model_from_scratch_test_v5/val_data/val_npzs


## Apply segmentation model to folder
python seg_images_in_folder_no_tkinter.py --sample_direc /gym/my_segmentation_gym_datasets/capehatteras_data/toPredict  --weights /gym/my_segmentation_gym_datasets/weights/hatteras_l8_resunet_both_fullmodel.h5

    Parameters:

    --sample_direc
    Description: Directory containing images to which the segmentation model will be applied.
    Example: --sample_direc /gym/my_segmentation_gym_datasets/capehatteras_data/toPredict

    --weights
    Description: Path to the model weights file used for performing segmentation.
    Example: --weights /gym/my_segmentation_gym_datasets/weights/hatteras_l8_resunet_both_fullmodel.h5

## Make dataset
python make_dataset_no_tkinter.py -o /gym/model_from_scratch_test_v5 -c /gym/my_segmentation_gym_datasets_v5/config/hatteras_l8_resunet.json -i /gym/my_segmentation_gym_datasets/capehatteras_data/fromDoodler/images -l /gym/my_segmentation_gym_datasets/capehatteras_data/fromDoodler/labels


    Parameters:

    -o
    Description: Output directory where the dataset and subsequent model training files will be stored.
    Example: -o /gym/model_from_scratch_test_v5

    -c
    Description: Path to the configuration JSON file.
    Example: -c /gym/my_segmentation_gym_datasets_v5/config/hatteras_l8_resunet.json

    -i
    Description: Directory containing the source images.
    Example: -i /gym/my_segmentation_gym_datasets/capehatteras_data/fromDoodler/images

    -l
    Description: Directory containing the label images corresponding to the source images.
    Example: -l /gym/my_segmentation_gym_datasets/capehatteras_data/fromDoodler/labels

## Batch Train Model
python batch_train_models_no_tkinter.py -c /gym/my_segmentation_gym_datasets_v5/config/hatteras_l8_resunet.json -t /gym/model_from_scratch_test_v5/train_data/train_npzs -v /gym/model_from_scratch_test_v5/val_data/val_npzs

    Parameters
    -c
    Description: Configuration JSON file specifying model training settings.
    Example: -c /gym/my_segmentation_gym_datasets_v5/config/hatteras_l8_resunet.json

    -t
    Description: Directory containing the training data files (npz format).
    Example: -t /gym/model_from_scratch_test_v5/train_data/train_npzs

    -v
    Description: Directory containing the validation data files (npz format).
    Example: -v /gym/model_from_scratch_test_v5/val_data/val_npzs