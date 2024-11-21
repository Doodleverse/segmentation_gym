
from glob import glob 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import matplotlib 
from joblib import delayed, Parallel
import skimage.io as io

##========================================================
def fromhex(n):
    """ hexadecimal to integer """
    return int(n, base=16)

image_path = r"D:\CDI_runup\AK_runup_timestacks\r2"
image_path = os.path.normpath(image_path)

im_files = sorted(glob(image_path+os.sep+"*.png"))
len(im_files)


# label_path = r"D:\CDI_runup\AK_runup_timestacks\r1\meta"
# label_path = os.path.normpath(label_path)

# lab_files = sorted(glob(label_path+os.sep+"*.png"))
# len(lab_files)

lab_files = [i.replace('runup.','runup_predseg.').replace('r2\\','r2\\meta\\') for i in im_files]

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


def do_it(im, lab, cmap):
    try:
        l = io.imread(lab) 
        l[l==3] = 0
        fig = plt.figure(figsize=(6,6))
        plt.imshow(io.imread(im))
        plt.imshow(l, cmap=cmap, alpha=0.5, vmin=0, vmax=NUM_LABEL_CLASSES)
        # plt.show()
        plt.savefig(im.replace("runup.","overlay.").replace('.png','.jpg'), dpi=300, bbox_inches='tight')
        plt.close('all')
    except:
        pass

# for im,lab in zip(im_files,lab_files):
    
#     l = io.imread(lab) 
#     l[l==3] = 0
#     fig = plt.figure(figsize=(6,6))
#     plt.imshow(io.imread(im))
#     plt.imshow(l, cmap=cmap, alpha=0.5, vmin=0, vmax=NUM_LABEL_CLASSES)
#     # plt.show()
#     plt.savefig(im.replace("images","overlays").replace('.tif','.png'), dpi=300, bbox_inches='tight')
#     plt.close('all')

Parallel(n_jobs=-2)(delayed(do_it)(i,l,cmap) for i,l in zip(im_files,lab_files))
