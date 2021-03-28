
import os, zipfile, shutil
import tensorflow as tf
from glob import glob

# not available yet


# for k in np.arange(1,10):
#     url = "https://github.com/dbuscombe-usgs/segmentation_zoo/releases/download/0.0.2/watermask-nadir-datasets"+str(k)+".zip"
#     filename = os.path.join(os.getcwd(), "watermask-nadir-datasets"+str(k)+".zip")
#     tf.keras.utils.get_file(filename, url)
#
#     with zipfile.ZipFile("watermask-nadir-datasets"+str(k)+".zip", "r") as z_fp:
#         z_fp.extractall("./")
