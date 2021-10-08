## A little utility script to check a config file for TARGET_SIZE compatible with the model
## (for use with training new models - note that in prediction mode, TARGET_SIZE stays the same 
## even if your sample imagery is a different size)

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

import json
from tqdm import tqdm
from tkinter import filedialog
from tkinter import *

from imports import *
    
def findNextPowerOf2(n):
    n = n - 1
    while n & n - 1:
        n = n & n - 1
    return n << 1

def is_odd(num):
    return num & 0x1


root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/data",title = "Select config file",filetypes = (("config files","*.json"),("all files","*.*")))
configfile = root.filename
print(configfile)
root.withdraw()

weights = configfile.replace('.json','.h5').replace('config', 'weights')

#---------------------------------------------------
with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')

from imports import *
#suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if is_odd(TARGET_SIZE[0]):
    print("Target size of %i is odd-valued ... using %i" % (TARGET_SIZE[0], TARGET_SIZE[0]-1))
    TARGET_SIZE[0] = TARGET_SIZE[0]-1

if is_odd(TARGET_SIZE[1]):
    print("Target size of %i is odd-valued ... using %i" % (TARGET_SIZE[1], TARGET_SIZE[1]-1))
    TARGET_SIZE[1] = TARGET_SIZE[1]-1
    
try:

    if NCLASSES==1:
        model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), BATCH_SIZE, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))
    else:
        model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), BATCH_SIZE, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))
    print("Target size of (%i, %i) works with the residual U-Net model" % (TARGET_SIZE[0], TARGET_SIZE[1]))
except:
    print("Target size of (%i, %i) does not work with the residual U-Net model" % (TARGET_SIZE[0], TARGET_SIZE[1]))
    print("Searching for nearest compatible image size ...")
    
    X = TARGET_SIZE[0]
    Y = TARGET_SIZE[1]

    L = []
    for i in tqdm(np.arange(X-64,X+64,16)):
        for j in np.arange(Y-64,Y+64,16):
            try:
                if NCLASSES==1:
                    model = res_unet((i, j, N_DATA_BANDS), BATCH_SIZE, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))
                else:
                    model = res_unet((i,j, N_DATA_BANDS), BATCH_SIZE, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))  
                L.append((i,j))
                print("%i,%i are compatible" % (i,j))
            except:
                K.clear_session()
                
    if L==[]:
        TARGET_SIZE = [findNextPowerOf2(TARGET_SIZE[0]), findNextPowerOf2(TARGET_SIZE[1])]
        print("%i,%i are compatible image sizes" % (TARGET_SIZE[0],TARGET_SIZE[1]))
