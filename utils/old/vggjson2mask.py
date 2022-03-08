# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2022, Marda Science LLC
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
from tkinter import filedialog, messagebox
from tkinter import *
import json, os, glob, shutil
from PIL import Image, ImageDraw, ExifTags
import numpy as np
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt


#######===================================================
def make_dir(dirname):
    # check that the directory does not already exist
    if not os.path.isdir(dirname):
        # if not, try to create the directory
        try:
            os.mkdir(dirname)
        # if there is an exception, print to screen and try to continue
        except Exception as e:
            print(e)
    # if the dir already exists, let the user know
    else:
        print('{} directory already exists'.format(dirname))

def move_files(files, outdirec):
    for a_file in files:
        shutil.move(a_file, outdirec+os.sep+a_file.split(os.sep)[-1])

def fromhex(n):
    """ hexadecimal to integer """
    return int(n, base=16)

def get_data(data):
    X = []; Y = []; L=[] #pre-allocate lists to fill in a for loop
    for k in data['regions']: #cycle through each polygon
        # get the x and y points from the dictionary
        X.append(data['regions'][k]['shape_attributes']['all_points_x'])
        Y.append(data['regions'][k]['shape_attributes']['all_points_y'])
        L.append(data['regions'][k]['region_attributes']['label'])
    return Y,X,L #image coordinates are flipped relative to json coordinates
	
def get_mask(X, Y, L, class_dict, image):
    # get the dimensions of the image
    nx, ny, nz = np.shape(image)
    mask = np.zeros((nx,ny))
    codes = []
    
    for y,x,c in zip(X,Y,L):
        # the ImageDraw.Draw().polygon function we will use to create the mask
        # requires the x's and y's are interweaved, which is what the following
        # one-liner does    
        polygon = np.vstack((x,y)).reshape((-1,),order='F').tolist()
        
        # create a mask image of the right size and infill according to the polygon
        if nx>ny:
           x,y = y,x 
           img = Image.new('L', (ny, nx), 0)
        elif ny>nx:
           x,y = y,x 
           img = Image.new('L', (ny, nx), 0)            
        else:
           img = Image.new('L', (nx, ny), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        # turn into a numpy array
        #m = np.array(img)
        m = np.flipud(np.rot90(np.array(img)))
        m[m>0] = class_dict[c]
        codes.append(class_dict[c])
        try:
            mask = mask + m
        except:
            mask = mask + m.T
         
    return mask , np.unique(codes)
########==============================================


def make_jpegs(alpha):
   root = Tk()
   root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select directory of VGG JSON files")
   data_path = root.filename
   print(data_path)
   root.withdraw()

   root = Tk()
   root.filename =  filedialog.askdirectory(initialdir = data_path,title = "Select directory of image files")
   image_path = root.filename
   print(image_path)
   root.withdraw()

   out_path = image_path.replace(image_path.split(os.sep[-1])[-1],'images')
   try:
      os.mkdir(out_path)
   except:
      pass

   overlay_path = image_path.replace(image_path.split(os.sep[-1])[-1],'overlays')
   try:
      os.mkdir(overlay_path)
   except:
      pass

   label_path = image_path.replace(image_path.split(os.sep[-1])[-1],'labels')
   try:
      os.mkdir(label_path)
   except:
      pass


   all_labels = []
   for root, dirs, files in os.walk(data_path, topdown=False):
      tmp = []
      files = [f for f in files if f.endswith('json')]
      for name in files:
         print(os.path.join(root, name))
         tmp.append( json.load( open(os.path.join(root, name)) ) )
      all_labels.append(tmp)

   all_names = []
   for root, dirs, files in os.walk(image_path, topdown=False):
      for name in files:
         all_names.append(os.path.join(root, name))

   print("{} labels found".format(len(all_labels[0])))
   print("{} images found".format(len(all_names)))

#    ALL_CLASSES = []

   Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
   classfile = filedialog.askopenfilename(title='Select file containing class (label) names', initialdir=image_path, filetypes=[("Pick classes.txt file","*.txt")])

   with open(classfile) as f:
      classes = f.readlines()

   class_dict = {}
   for counter, i in enumerate(classes):
      class_dict[i.strip("\n")]=counter

   for tmp in all_labels:
      for label in tmp:
         for img in label.keys():
            print("Working on image {}".format(img))
            X, Y, L = get_data(label[img])
            # ALL_CLASSES.append(np.unique(L))
            try:
               rawfile = [f for f in all_names if f.endswith(img)][0]
            except:
               print("Image not found: {}. Skipping ...")
               rawfile = None

            if rawfile is not None:
               image = Image.open(rawfile)		 	

               for orientation in ExifTags.TAGS.keys():
                  # if ExifTags.TAGS[orientation]=='Orientation':
                  #    break
                  try:
                     exif=dict(image._getexif().items())
                     if exif[orientation] == 3:
                        image=image.rotate(180, expand=True)
                     elif exif[orientation] == 6:
                        image=image.rotate(270, expand=True)
                     elif exif[orientation] == 8:
                        image=image.rotate(90, expand=True)
                  except:
                     # print('no exif')
                     pass

               mask, codes = get_mask(X, Y, L, class_dict,image)
               mask = Image.fromarray(mask).convert('L')
               ext = rawfile.split('.')[-1]
               mask.save((rawfile.split("."+ext)[0]+'_label.jpg').replace(image_path,label_path), format='JPEG')

               image.save((rawfile.split("."+ext)[0]+'.jpg').replace(image_path,out_path), format='JPEG')

               # class_label_names = [c.strip() for c in L]
               # class_label_names = np.unique(class_label_names)

               NUM_LABEL_CLASSES = len(codes) #class_label_names)

               if NUM_LABEL_CLASSES<=10:
                  class_label_colormap = px.colors.qualitative.G10
               else:
                  class_label_colormap = px.colors.qualitative.Light24

               # we can't have fewer colors than classes
               assert NUM_LABEL_CLASSES <= len(class_label_colormap)

               colormap = [
                  tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
                  for h in [c.replace("#", "") for c in class_label_colormap]
               ]

               cmap = matplotlib.colors.ListedColormap(class_label_colormap[:NUM_LABEL_CLASSES+1])

               #Make an overlay
               plt.imshow(image)
               plt.imshow(mask, cmap=cmap, alpha=alpha, vmin=0, vmax=NUM_LABEL_CLASSES)
               plt.axis('off')
               plt.savefig((rawfile.replace(rawfile.split(".")[-1],'overlay.png')).replace(image_path,overlay_path), dpi=200, bbox_inches='tight')
               plt.close('all')


   # overdir = os.path.join(image_path, 'overlays')
   # make_dir(overdir)
   # ovfiles = glob.glob(image_path+'/*overlay.png')
   # outdirec = os.path.normpath(image_path + os.sep+'overlays')
   # move_files(ovfiles, outdirec)

   # overdir = os.path.join(image_path, 'labels')
   # make_dir(overdir)
   # ovfiles = glob.glob(image_path+'/*label.png')
   # outdirec = os.path.normpath(image_path + os.sep+'labels')
   # move_files(ovfiles, outdirec)


###==================================================================
#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:a:") #m:p:l:")
    except getopt.GetoptError:
        print('======================================')
        print('python vggjson2mask.py') #
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('======================================')
            print('Example usage: python vggjson2mask.py -a 0.4') 
            print('======================================')
            sys.exit()
        elif opt == 'a':
            alpha = arg
            alpha = float(alpha)
    #ok, dooo it
    if 'alpha' in locals():
      if (alpha < 0) or (alpha>1):
         alpha=0.5
         print("alpha outside of range, using alpha = 0.5")
    else:
      alpha = 0.5

    make_jpegs(alpha)


# boom.    


# ## if you need to rotate images before labeling them on makesense.ai
#          try:
#             image = Image.open(rawfile)
#             for orientation in ExifTags.TAGS.keys():
#                if ExifTags.TAGS[orientation]=='Orientation':
#                   break
#             exif=dict(image._getexif().items())

#             if exif[orientation] == 3:
#                image=image.rotate(180, expand=True)
#             elif exif[orientation] == 6:
#                image=image.rotate(270, expand=True)
#             elif exif[orientation] == 8:
#                image=image.rotate(90, expand=True)

#          except (AttributeError, KeyError, IndexError):
#             # cases: image don't have getexif
#             pass
#          image.save(rawfile, format='PNG')	


