import numpy as np 
from glob import glob 
import os, shutil
from tqdm import tqdm

indirec = 'v4'
outdirec = 'v5_subset'

# read files
files = glob('v4/*.npz')

# make directory for outputs
os.mkdir(outdirec)
os.mkdir(outdirec+os.sep+'no_use')

# set minimum threshold for any proportion
# if any normalized class frequency distribution is less than this number
# it is discounted (moved to 'no_use')
thres = 1e-2

# read files one by one
for file in tqdm(files):
    with np.load(file) as data:
        label = data['arr_1'].astype('uint8')
        label = np.argmax(label,-1)
        # get normalized class distributions
        norm_class_dist = np.bincount(label.flatten())/np.sum(label>-1)
        # if length > 1, copy the file
        # if below thres
        if np.any(norm_class_dist<thres):
            shutil.copyfile(file,file.replace(indirec,outdirec+os.sep+'no_use'))
            # print('below threshold')
        elif not len(norm_class_dist)==1:
            shutil.copyfile(file,file.replace(indirec,outdirec))            
        else:
            shutil.copyfile(file,file.replace(indirec,outdirec+os.sep+'no_use'))
