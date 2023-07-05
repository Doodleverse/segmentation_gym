
from glob import glob 
import os, shutil 
import numpy as np
from tqdm import tqdm 

indir = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\npz4gym\\val_data\\val_npzs"
outdir = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\npz4gym\\val_data\\val_npzs\\bad"

# indir = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\npz4gym\\train_data\\train_npzs"
# outdir = "G:\\elwha_ortho_segmentation\\seg_zoo_model_datasets\\elwha_aerial\\v8_june2023_all\\npz4gym\\train_data\\train_npzs\\bad"


files = glob(os.path.normpath(indir+os.sep+"*.npz"))
len(files)

for f in tqdm(files):
    dat = np.load(f)
    data = dict()
    for k in dat.keys():
        data[k] = dat[k]
    del dat

    if data['arr_1'].shape != (1024, 1024, 4):
        print(f)
        shutil.move(f,f.replace(indir,outdir))
