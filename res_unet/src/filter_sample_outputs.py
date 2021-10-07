
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

import numpy as np
from glob import glob
import pandas as pd
import os
from tqdm import tqdm
from tkinter import filedialog
from tkinter import *
from tkinter import simpledialog
# import matplotlib.pyplot as plt

do_conversion = False

#### root = "F:\\planecam\\2018-1006-NC-pix\\jpg_adobe\\conf_var\\"

tkp = Tk()
tkp.filename =  filedialog.askdirectory(initialdir = "/samples",title = "Select directory of npz (conf/var) files")
sample_direc = tkp.filename
print(sample_direc)
tkp.withdraw()

tkp = Tk()
rootfilename = simpledialog.askstring("Input", "Root of filename for output statistics csv files",
                                parent=tkp)
tkp.withdraw()

###confidence
sample_direc = os.path.normpath(sample_direc)
filenames = sorted(glob(sample_direc+os.sep+'*conf.npz'))

dictobj = {}
dictobj['filenames'] = filenames

M = []; m = []; Mn = []; Md = []
for f in tqdm(filenames):
    #print(f)
    with np.load(f) as data:
        dat = data['arr_0'].astype('float16')
    dat[np.isnan(dat)] = 0
    dat[np.isinf(dat)] = 0
    M.append(np.max(dat))
    m.append(np.min(dat))
    Mn.append(np.mean(dat))
    Md.append(np.median(dat))
    if do_conversion:
        np.savez(f, dat)

dictobj['min_conf'] = m
del m
dictobj['max_conf'] = M
del M
dictobj['mean_conf'] = Mn
del Mn
dictobj['median_conf'] = Md
del Md

df = pd.DataFrame(dictobj)
df.to_csv(rootfilename+'_conf_stats.csv') #'2018-1006-NC-pix-conf_stats.csv')

##stdev
filenames = sorted(glob(sample_direc+os.sep+'*var.npz'))

dictobj2 = {}
dictobj2['filenames'] = filenames

M = []; m = []; Mn = []; Md = []
for f in tqdm(filenames):
    #print(f)
    with np.load(f) as data:
        dat = data['arr_0'].astype('float16')
    dat[np.isnan(dat)] = 0
    dat[np.isinf(dat)] = 0

    M.append(np.max(dat))
    m.append(np.min(dat))
    Mn.append(np.mean(dat))
    Md.append(np.median(dat))
    if do_conversion:
        np.savez(f, dat)

dictobj2['min_stdev'] = m
del m
dictobj2['max_stdev'] = M
del M
dictobj2['mean_stdev'] = Mn
del Mn
dictobj2['median_stdev'] = Md
del Md


df = pd.DataFrame(dictobj2)
df.to_csv(rootfilename+'_stdev_stats.csv') #'2018-1006-NC-pix-stdev_stats.csv')


# plt.plot(dictobj['median_conf'] , dictobj2['median_stdev'],'ko')
# plt.show()
