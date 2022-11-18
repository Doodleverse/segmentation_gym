# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2022, Marda Science LLC
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

import numpy as np 
from glob import glob 
import os, shutil
from tqdm import tqdm
from tkinter import filedialog, messagebox
from tkinter import *

# Request the folder containing the imagery/npz to segment 
# sample_direc: full path to the directory
root = Tk()
root.filename =  filedialog.askdirectory(title = "Select directory of class-imbalanced npzs")
indirec = root.filename
print(indirec)
root.withdraw()

outdirec = os.path.normpath(os.path.dirname(indirec)+os.sep+os.path.basename(indirec)+'_subset')
print(outdirec)

# set minimum threshold for any proportion
# if any normalized class frequency distribution is less than this number
# it is discounted (moved to 'no_use')
# thres = 1e-2
print("Input threshold for minor class [0 - 1], typically <0.25")
print("This is the smallest acceptable proportion of the minority class. Samples were minority < threshold will not be used")
print("The smaller the threshold, the fewer the number of samples used in the subset")
# print("\n")
thres = float(input())

print("Threshold chosen: {}".format(thres))

# read files
files = glob(indirec+os.sep+'*.npz')

try:
    # make directory for outputs
    os.mkdir(outdirec)
    os.mkdir(outdirec+os.sep+'no_use')
except:
    pass

# read files one by one
for file in tqdm(files):
    with np.load(file) as data:
        label = data['arr_1'].astype('uint8')
        label = np.argmax(label,-1)
        # get normalized class distributions
        norm_class_dist = np.bincount(label.flatten())/np.sum(label>-1)
        # if below thres
        if np.any(norm_class_dist<thres):
            shutil.copyfile(file,file.replace(indirec,outdirec+os.sep+'no_use'))
            # print('below threshold')
        elif not len(norm_class_dist)==1:
            shutil.copyfile(file,file.replace(indirec,outdirec))            
        else: # if length > 1, copy the file
            shutil.copyfile(file,file.replace(indirec,outdirec+os.sep+'no_use'))
