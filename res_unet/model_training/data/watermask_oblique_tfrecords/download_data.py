
import os, zipfile
import tensorflow as tf


url = "https://github.com/segmentation_zoo/releases/download/v0.0.1/watermask-oblique-tfrecords.zip"
filename = os.path.join(os.getcwd(), "watermask-oblique-tfrecords.zip")
tf.keras.utils.get_file(filename, url)


with zipfile.ZipFile("watermask-oblique-tfrecords.zip", "r") as z_fp:
    z_fp.extractall("./")
