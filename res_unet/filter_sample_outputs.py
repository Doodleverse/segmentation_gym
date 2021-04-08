
import numpy as np
from glob import glob 
import pandas as pd
import os
from tqdm import tqdm

do_conversion = True #False

root = "F:\\planecam\\2018-1006-NC-pix\\jpg_adobe\\conf_var\\"

###confidence
root = os.path.normpath(root)
filenames = sorted(glob(root+os.sep+'*conf.npz'))

dictobj = {}
dictobj['filenames'] = filenames

M = []; m = []; Mn = []; Md = []
for f in tqdm(filenames):
	#print(f)
	with np.load(f) as data:	
		dat = data['arr_0'].astype('float16')
	M.append(np.nanmax(dat))
	m.append(np.nanmin(dat))
	Mn.append(np.nanmean(dat))
	Md.append(np.nanmedian(dat))
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
df.to_csv('2018-1006-NC-pix-conf_stats.csv')

##stdev
filenames = sorted(glob(root+os.sep+'*var.npz'))

dictobj = {}
dictobj['filenames'] = filenames

M = []; m = []; Mn = []; Md = []
for f in tqdm(filenames):
	#print(f)
	with np.load(f) as data:	
		dat = data['arr_0'].astype('float16')
	M.append(np.nanmax(dat))
	m.append(np.nanmin(dat))
	Mn.append(np.nanmean(dat))
	Md.append(np.nanmedian(dat))
	if do_conversion:
		np.savez(f, dat)

dictobj['min_stdev'] = m 
del m 
dictobj['max_stdev'] = M
del M
dictobj['mean_stdev'] = Mn
del Mn
dictobj['median_stdev'] = Md
del Md


df = pd.DataFrame(dictobj)
df.to_csv('2018-1006-NC-pix-stdev_stats.csv')

