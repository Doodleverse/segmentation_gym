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

url = "https://github.com/dbuscombe-usgs/segmentation_zoo/releases/download/0.1.1/oblique_2class_watermask_masks.zip"

filename = os.path.join(os.getcwd(), "oblique_2class_watermask_masks.zip")
tf.keras.utils.get_file(filename, url)

with zipfile.ZipFile("oblique_2class_watermask_masks.zip", "r") as z_fp:
    z_fp.extractall("./")
os.remove(filename)

for k in range(1,6):
    url = "https://github.com/dbuscombe-usgs/segmentation_zoo/releases/download/0.1.1/oblique_2class_watermask_images"+str(k)+".zip"
    filename = os.path.join(os.getcwd(), "oblique_2class_watermask_images"+str(k)+".zip")
    tf.keras.utils.get_file(filename, url)

    with zipfile.ZipFile("oblique_2class_watermask_images"+str(k)+".zip", "r") as z_fp:
        z_fp.extractall("./")
    os.remove(filename)


#==== move files
os.mkdir('images')

for file in glob(os.getcwd()+os.sep+'*.jpg'):
    shutil.move(file,'images/')
