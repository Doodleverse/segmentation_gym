
import os, shutil
from glob import glob

for file in glob('../watermask_nadir_datasets/*npz'):
    shutil.copy(file,'./')

for file in glob('../watermask_oblique_datasets/*npz'):
    shutil.copy(file,'./')
