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
from skimage.io import imsave, imread


url = "https://github.com/dbuscombe-usgs/segmentation_zoo/releases/download/0.1.3/santacruz_sentinel2_rgb.zip"

filename = os.path.join(os.getcwd(), "santacruz_sentinel2_rgb.zip")
tf.keras.utils.get_file(filename, url)

with zipfile.ZipFile("santacruz_sentinel2_rgb.zip", "r") as z_fp:
    z_fp.extractall("./")
os.remove(filename)


url = "https://github.com/dbuscombe-usgs/segmentation_zoo/releases/download/0.1.3/santacruz_sentinel2_labels.zip"

filename = os.path.join(os.getcwd(), "santacruz_sentinel2_labels.zip")
tf.keras.utils.get_file(filename, url)

with zipfile.ZipFile("santacruz_sentinel2_labels.zip", "r") as z_fp:
    z_fp.extractall("./")
os.remove(filename)

# os.rename('labels', 'masks')


url = "https://github.com/dbuscombe-usgs/segmentation_zoo/releases/download/0.1.3/santacruz_sentinel2_nir.zip"

filename = os.path.join(os.getcwd(), "santacruz_sentinel2_nir.zip")
tf.keras.utils.get_file(filename, url)

with zipfile.ZipFile("santacruz_sentinel2_nir.zip", "r") as z_fp:
    z_fp.extractall("./")
os.remove(filename)



#==== move files
os.mkdir('nir')
for file in glob('sc/*ir*.jpg'):
    shutil.move(file,'nir/')


os.mkdir('labels')
for file in glob('sc/*label.png'):
    shutil.move(file,'labels/')

shutil.move('sc/', 'images/')


##convert pngs to jpegs
for file in glob('labels/*label.png'):
    im = imread(file)
    imsave(file.replace('.png','.jpg'), im, quality=100)

for file in glob('labels/*label.png'):
    os.remove(file)
