
## run after running 'seg_images_in_folder.py'
# which creates a folder of 'predseg' outputs for mapping


import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import os 
from glob import glob

root = Tk()
root.filename =  askdirectory(title = "Select directory of model output (label) files")
folder = root.filename
print(folder)
root.withdraw()

files = glob(folder+os.sep+'*predseg.png')
print("Found {} files".format(len(files)))

root = Tk()
root.filename =  askdirectory(title = "Select directory of wld files")
wld = root.filename
print(wld)
root.withdraw()

root = Tk()
root.filename =  askdirectory(title = "Select directory of xml files")
xml = root.filename
print(xml)
root.withdraw()

root = Tk()
root.filename =  askdirectory(title = "Select directory to store outputs")
output = root.filename
print(output)
root.withdraw()

print('Copying flles to '+output)
os.system('cp '+xml+'/*.xml '+output)
os.system('cp '+wld+'/*.wld '+output)
os.system('cp '+folder+'/*predseg.png '+output)

print('Converting png to jpg files')
os.system('for f in '+output+'/*.png; do convert $f -quality 100 "${f%_predseg.png}.jpg"; done')

print('Converting jpg to geotiff files')
os.system('for k in '+output+'/*.jpg; do gdal_translate -of GTiff -b 1 $k "${k%jpg}tif"; done')

print('Making large mosaic geotiff file')
os.system('gdal_merge.py -o '+output+'/mosaic.tif -of GTiff -co "COMPRESS=JPEG" -co "TILED=YES"  '+output+'/*.tif')