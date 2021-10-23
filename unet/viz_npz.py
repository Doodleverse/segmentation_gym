
import numpy as np
import matplotlib.pyplot as plt

example='/media/marda/TWOTB1/USGS/SOFTWARE/Projects/UNets/pcmsc_watermasking/npz_formodel/watermask-oblique-planecam-data-oct2021augimage_000000431.npz'


with np.load(example) as data:
    image = data['arr_0'].astype('uint8')
    label = data['arr_1'].astype('uint8')

lab = np.argmax(label,-1)

plt.imshow(image)
plt.imshow(lab, alpha=0.5)
plt.show()
