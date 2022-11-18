# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021-2022, Marda Science LLC
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

from numpy.lib.npyio import load
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from skimage.io import imsave, imread
from glob import glob 
import os
from tqdm import tqdm

##========================================================
def fromhex(n):
    """ hexadecimal to integer """
    return int(n, base=16)

##========================================================
def label_to_colors(
    img,
    mask,
    alpha,#=128,
    colormap,#=class_label_colormap, #px.colors.qualitative.G10,
    color_class_offset,#=0,
    do_alpha,#=True
):
    """
    Take MxN matrix containing integers representing labels and return an MxNx4
    matrix where each label has been replaced by a color looked up in colormap.
    colormap entries must be strings like plotly.express style colormaps.
    alpha is the value of the 4th channel
    color_class_offset allows adding a value to the color class index to force
    use of a particular range of colors in the colormap. This is useful for
    example if 0 means 'no class' but we want the color of class 1 to be
    colormap[0].
    """

    colormap = [
        tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
        for h in [c.replace("#", "") for c in colormap]
    ]

    cimg = np.zeros(img.shape[:2] + (3,), dtype="uint8")
    minc = np.min(img)
    maxc = np.max(img)

    for c in range(minc, maxc + 1):
        cimg[img == c] = colormap[(c + color_class_offset) % len(colormap)]

    cimg[mask==1] = (0,0,0)

    if do_alpha is True:
        return np.concatenate(
            (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
        )
    else:
        return cimg



root = Tk()
root.filename =  askdirectory(title = "Select directory of output npz files")
folder = root.filename
print(folder)
root.withdraw()

files = glob(folder+os.sep+'*.npz')
print("Found {} files".format(len(files)))

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
classfile = askopenfilename(title='Select file containing class (label) names', filetypes=[("Pick classes.txt file","*.txt")])

with open(classfile) as f:
    classes = f.readlines()

for anno_file in tqdm(files):

    data = dict()
    with load(anno_file, allow_pickle=True) as dat:
        #create a dictionary of variables
        #automatically converted the keys in the npz file, dat to keys in the dictionary, data, then assigns the arrays to data
        for k in dat.keys():
            data[k] = dat[k]
        del dat

    class_label_names = [c.strip() for c in classes]

    NUM_LABEL_CLASSES = len(class_label_names)

    if NUM_LABEL_CLASSES<=10:
        class_label_colormap = px.colors.qualitative.G10
    else:
        class_label_colormap = px.colors.qualitative.Light24

    cimg = label_to_colors(data['grey_label'], data['grey_label']==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)

    imsave(anno_file.replace('.npz','.npz_cl.jpg'),
                cimg, quality=100, chroma_subsampling=False, check_contrast=False)

    
    gimg = data['av_prob_stack'][:,:,1]>.05

    imsave(anno_file.replace('.npz','.npz_gl.jpg'),
                gimg.astype('uint8'), quality=100, chroma_subsampling=False, check_contrast=False)


#