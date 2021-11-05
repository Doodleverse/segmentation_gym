
from skimage.io import imread, imsave
import numpy as np
from tkinter import filedialog
from tkinter import *
import os
from glob import glob
from skimage.transform import rescale

TARGET_SIZE = (512,512)

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select directory of image files")
data_path = root.filename
print(data_path)
root.withdraw()

files = sorted(glob(data_path+os.sep+'*.jpg'))

direc = os.path.dirname(files[0])
direclabels = direc.replace('images', 'labels')

labelfiles = sorted(glob(direclabels+os.sep+'*.jpg'))

print("{} image files".format(len(files)))
print("{} label image files".format(len(labelfiles)))


newdirec = direc.replace('images','padded_images')
try:
    os.mkdir(newdirec)
except:
    pass

newdireclabels = direclabels.replace('labels','padded_labels')
try:
    os.mkdir(newdireclabels)
except:
    pass

for file,lfile in zip(files, labelfiles):
    # read image
    img = imread(file)
    lab = imread(lfile)
    # print("before")
    print(np.unique(lab))

    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size and color (blue) for padding
    new_image_width = TARGET_SIZE[0]
    new_image_height = TARGET_SIZE[0]
    color = (0,0,0)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    try:
        # copy img image into center of result image
        result[y_center:y_center+old_image_height,
               x_center:x_center+old_image_width] = img
    except:
        sf = np.minimum(new_image_width/old_image_width,new_image_height/old_image_height)
        img = rescale(img,(sf,sf,1),anti_aliasing=True, preserve_range=True)
        old_image_height, old_image_width, channels = img.shape
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        result[y_center:y_center+old_image_height,
               x_center:x_center+old_image_width] = img.astype('uint8')

    # save result
    imsave(file.replace('images','padded_images'), result, check_contrast=False)


    color = (0)
    result = np.full((new_image_height,new_image_width), color, dtype=np.uint8)

    try:
        # copy img image into center of result image
        result[y_center:y_center+old_image_height,
               x_center:x_center+old_image_width] = lab+1
    except:
        lab2 =rescale(lab,(sf,sf),anti_aliasing=False, preserve_range=True, order=0)

        result[y_center:y_center+old_image_height,
               x_center:x_center+old_image_width] = lab2+1

        # print("after")
        # print(np.unique(lab2))

        del lab2

    # save result
    imsave(lfile.replace('labels','padded_labels'), result, check_contrast=False)
