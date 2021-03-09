
import os, shutil
from glob import glob

for file in glob('../watermask_nadir_tfrecords/*tfrec'):
    shutil.copy(file,'./')

for file in glob('../watermask_oblique_tfrecords/*tfrec'):
    shutil.copy(file,'./')
