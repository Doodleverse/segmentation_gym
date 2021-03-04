
import os, zipfile
import tensorflow as tf


url = "https://github.com/dbuscombe-usgs/segmentation_zoo/releases/download/v0.0.3/santacruz_sentinel2-4class-tfrecords.zip"
filename = os.path.join(os.getcwd(), "santacruz_sentinel2-4class-tfrecords.zip")
tf.keras.utils.get_file(filename, url)

with zipfile.ZipFile("santacruz_sentinel2-4class-tfrecords.zip", "r") as z_fp:
    z_fp.extractall("./")
