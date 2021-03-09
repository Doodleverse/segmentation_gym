# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os, zipfile, shutil
import tensorflow as tf
from glob import glob


##====nadir
url = "https://github.com/dbuscombe-usgs/segmentation_zoo/releases/download/0.3.1/nadir_sample_images1.zip"

filename = os.path.join(os.getcwd(), "nadir_sample_images1.zip")
tf.keras.utils.get_file(filename, url)

with zipfile.ZipFile("nadir_sample_images1.zip", "r") as z_fp:
    z_fp.extractall("./")
os.remove(filename)

url = "https://github.com/dbuscombe-usgs/segmentation_zoo/releases/download/0.3.1/nadir_sample_images2.zip"

filename = os.path.join(os.getcwd(), "nadir_sample_images2.zip")
tf.keras.utils.get_file(filename, url)

with zipfile.ZipFile("nadir_sample_images2.zip", "r") as z_fp:
    z_fp.extractall("./")
os.remove(filename)

os.mkdir('nadir_coast_watermask')
for file in glob('*.jpg'):
    shutil.move(file,'nadir_coast_watermask/')


##====oblique
url = "https://github.com/dbuscombe-usgs/segmentation_zoo/releases/download/0.3.1/oblique_samples.zip"

filename = os.path.join(os.getcwd(), "oblique_samples.zip")
tf.keras.utils.get_file(filename, url)

with zipfile.ZipFile("oblique_samples.zip", "r") as z_fp:
    z_fp.extractall("./")
os.remove(filename)


os.mkdir('oblique_coast_watermask')
for file in glob('*.jpg'):
    shutil.move(file,'oblique_coast_watermask/')


##====satellite
url = "https://github.com/dbuscombe-usgs/segmentation_zoo/releases/download/0.3.1/sentinel2_sample_images.zip"

filename = os.path.join(os.getcwd(), "sentinel2_sample_images.zip")
tf.keras.utils.get_file(filename, url)

with zipfile.ZipFile("sentinel2_sample_images.zip", "r") as z_fp:
    z_fp.extractall("./")
os.remove(filename)


os.mkdir('sentinel2_sample')
for file in glob('*.jpg'):
    shutil.move(file,'sentinel2_sample/')
