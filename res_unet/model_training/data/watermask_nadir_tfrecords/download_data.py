
import os, zipfile
import tensorflow as tf


url = "https://github.com/segmentation_zoo/releases/download/v0.0.2/watermask-nadir-tfrecords.zip"
filename = os.path.join(os.getcwd(), "watermask-nadir-tfrecords.zip")
tf.keras.utils.get_file(filename, url)


with zipfile.ZipFile("watermask-nadir-tfrecords.zip", "r") as z_fp:
    z_fp.extractall("./")
