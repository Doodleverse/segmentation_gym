
from glob import glob 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import matplotlib 
from joblib import delayed, Parallel
import skimage.io as io
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from numpy.lib.npyio import load


##========================================================
def fromhex(n):
    """ hexadecimal to integer """
    return int(n, base=16)

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
direc = askdirectory(title='Select directory of results (npz)', initialdir=os.getcwd()+os.sep+'results')
files = sorted(glob(direc+'/*.npz'))

len(files)

outpath = os.path.dirname(files[0])

### edit below!
# 0=null, 1=water, 2=sed, 3=veg, 4=wood
NUM_LABEL_CLASSES = 5
gr = "#696A6C"
g = "#17BF39"
b = "#1F47C5"
y = "#EAF51E"
br = "#B78D2D"
class_label_colormap = [gr,b,y,g,br]

colormap = [
    tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
    for h in [c.replace("#", "") for c in class_label_colormap]
]

cmap = matplotlib.colors.ListedColormap(class_label_colormap[:NUM_LABEL_CLASSES])
cmap2 = matplotlib.colors.ListedColormap(['#000000']+class_label_colormap[:NUM_LABEL_CLASSES])


def do_it(file, outpath, cmap):
    try:

        data = dict()
        with load(file, allow_pickle=True) as dat:
            #create a dictionary of variables
            #automatically converted the keys in the npz file, dat to keys in the dictionary, data, then assigns the arrays to data
            for k in dat.keys():
                data[k] = dat[k]
            del dat

        infile = str(data['input_file'])
        outfile = infile.split(os.sep)[-1].replace('.jpg','_overlay.png')

        im = io.imread(infile)
        fig = plt.figure(figsize=(6,6))
        plt.imshow(im)
        plt.imshow(data['grey_label'], cmap=cmap, alpha=0.5, vmin=0, vmax=NUM_LABEL_CLASSES)
        # plt.show()
        plt.savefig(outpath+os.sep+outfile, dpi=300, bbox_inches='tight')
        plt.close('all')
    except:
        pass


Parallel(n_jobs=-2)(delayed(do_it)(i,outpath,cmap) for i in files)
