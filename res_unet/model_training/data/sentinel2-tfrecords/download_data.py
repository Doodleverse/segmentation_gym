
import os, zipfile
import tensorflow as tf


url = "https://github.com/segmentation_zoo/releases/download/v0.0.3/sentinel2-tfrecords.zip"
filename = os.path.join(os.getcwd(), "sentinel2-tfrecords.zip")
tf.keras.utils.get_file(filename, url)


with zipfile.ZipFile("sentinel2-tfrecords.zip", "r") as z_fp:
    z_fp.extractall("./")
