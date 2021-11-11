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

import sys,os, time
sys.path.insert(1, 'src')

USE_GPU = True #False

if USE_GPU == True:
   ##use the first available GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from prediction_imports import *
#====================================================

#====================================================

root = Tk()
root.filename =  filedialog.askopenfilename(title = "Select file",filetypes = (("weights file","*.h5"),("all files","*.*")))
weights = root.filename
print(weights)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(title = "Select directory of images to segment")
sample_direc = root.filename
print(sample_direc)
root.withdraw()

configfile = weights.replace('.h5','.json').replace('weights', 'config')


with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')


from imports import *

#=======================================================

print('.....................................')
print('Creating and compiling model ...')

if MODEL =='resunet':
    model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    FILTERS,
                    nclasses=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    kernel_size=(KERNEL,KERNEL),
                    strides=STRIDE,
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    )
elif MODEL=='unet':
    model =  custom_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    FILTERS,
                    nclasses=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    kernel_size=(KERNEL,KERNEL),
                    strides=STRIDE,
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    )

elif MODEL =='simple_resunet':
    # num_filters = 8 # initial filters
    # model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), num_filters, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))

    model = simple_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                kernel = (2, 2),
                num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                activation="relu",
                use_batch_norm=True,
                dropout=DROPOUT,#0.1,
                dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                dropout_type=DROPOUT_TYPE,#"standard",
                use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                filters=FILTERS,#8,
                num_layers=4,
                strides=(1,1))
#346,564
elif MODEL=='simple_unet':
    model = simple_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                kernel = (2, 2),
                num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                activation="relu",
                use_batch_norm=True,
                dropout=DROPOUT,#0.1,
                dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                dropout_type=DROPOUT_TYPE,#"standard",
                use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                filters=FILTERS,#8,
                num_layers=4,
                strides=(1,1))
#242,812

elif MODEL=='satunet':
    #model = sat_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), num_classes=NCLASSES)

    model = custom_satunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                kernel = (2, 2),
                num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                activation="relu",
                use_batch_norm=True,
                dropout=DROPOUT,#0.1,
                dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                dropout_type=DROPOUT_TYPE,#"standard",
                use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                filters=FILTERS,#8,
                num_layers=4,
                strides=(1,1))

else:
    print("Model must be one of 'unet', 'resunet', or 'satunet'")
    sys.exit(2)


# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])
model.compile(optimizer = 'adam', loss = dice_coef_loss, metrics = [mean_iou, dice_coef])

model.load_weights(weights)

metadatadict = {}
metadatadict['model_weights'] = weights
metadatadict['config_files'] = configfile
metadatadict['model_types'] = MODEL


### predict
print('.....................................')
print('Using model for prediction on images ...')

# sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))
sample_filenames = sorted(glob(sample_direc+os.sep+'*.jpg'))
if len(sample_filenames)==0:
    # sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.png'))
    sample_filenames = sorted(glob(sample_direc+os.sep+'*.png'))

print('Number of samples: %i' % (len(sample_filenames)))

for counter,f in enumerate(sample_filenames):
    do_seg(f, [model], metadatadict, sample_direc,NCLASSES, N_DATA_BANDS,TARGET_SIZE)
    print('%i out of %i done'%(counter,len(sample_filenames)))


# w = Parallel(n_jobs=2, verbose=0, max_nbytes=None)(delayed(do_seg)(f) for f in tqdm(sample_filenames))



#
# # =========================================================
# def do_seg(f, M, metadatadict,temp=0):
#
#     if 'jpg' in f:
#     	segfile = f.replace('.jpg', '_predseg.png')
#     elif 'png' in f:
#     	segfile = f.replace('.png', '_predseg.png')
#
#     metadatadict['input_file'] = f
#
#     segfile = os.path.normpath(segfile)
#     segfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'out'))
#
#     try:
#     	os.mkdir(os.path.normpath(sample_direc+os.sep+'out'))
#     except:
#     	pass
#
#     metadatadict['nclasses'] = NCLASSES
#     metadatadict['n_data_bands'] = N_DATA_BANDS
#
#     datadict={}
#
#     if NCLASSES==1:
#
#         if N_DATA_BANDS<=3:
#             image, w, h, bigimage = seg_file2tensor_3band(f)#, resize=True)
#         if image is None:
#             image = bigimage#/255
#             #bigimage = bigimage#/255
#             w = w.numpy(); h = h.numpy()
#         else:
#             image, w, h, bigimage = seg_file2tensor_4band(f, f.replace('aug_images', 'aug_nir'), resize=True )
#             if image is None:
#                 image = bigimage#/255
#                 w = w.numpy(); h = h.numpy()
#
#         print("Working on %i x %i image" % (w,h))
#
#         image = standardize(image.numpy()).squeeze()
#
#         E0 = []; E1 = [];
#
#         for counter,model in enumerate(M):
#
#             est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
#             print('Model {} applied'.format(counter))
#             E0.append(resize(est_label[:,:,0],(w,h), preserve_range=True, clip=True))
#             E1.append(resize(est_label[:,:,1],(w,h), preserve_range=True, clip=True))
#             del est_label
#
#         K.clear_session()
#
#         e0 = np.average(np.dstack(E0), axis=-1)#, weights=np.array(MW))
#
#         del E0
#
#         e1 = np.average(np.dstack(E1), axis=-1)#, weights=np.array(MW))
#         del E1
#
#         est_label = (e1+(1-e0))/2
#
#         datadict['av_prob_stack'] = est_label
#
#         del e0, e1
#
#         thres = threshold_otsu(est_label)+temp
#         print("Class threshold: %f" % (thres))
#         est_label = (est_label>thres).astype('uint8')
#         metadatadict['otsu_threshold'] = thres
#
#     else: ###NCLASSES>1
#
#         if N_DATA_BANDS<=3:
#         	image, w, h, bigimage = seg_file2tensor_3band(f)#, resize=True)
#         	image = image#/255
#         	bigimage = bigimage#/255
#         	w = w.numpy(); h = h.numpy()
#         else:
#         	image, w, h, bigimage = seg_file2tensor_4band(f, f.replace('aug_images', 'aug_nir'), resize=True )
#         	image = image#/255
#         	bigimage = bigimage#/255
#         	w = w.numpy(); h = h.numpy()
#
#         print("Working on %i x %i image" % (w,h))
#
#         #image = tf.image.per_image_standardization(image)
#         image = standardize(image.numpy())
#
#         est_label = np.zeros((w,h, NCLASSES))
#         for counter,model in enumerate(M):
#
#             # est_label = model.predict(tf.expand_dims(image, 0 , batch_size=1).squeeze()
#             est_label += model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
#             K.clear_session()
#
#         est_label = resize(est_label,(w,h))
#
#         est_label /= counter+1
#
#         datadict['av_prob_stack'] = est_label
#
#         est_label = np.argmax(est_label, -1)
#         metadatadict['otsu_threshold'] = np.nan
#
#
#     class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477','#66AA00','#B82E2E', '#316395']
#     #add classes for more than 10 classes
#
#     if NCLASSES>1:
#         class_label_colormap = class_label_colormap[:NCLASSES]
#     else:
#         class_label_colormap = class_label_colormap[:2]
#
#     metadatadict['color_segmentation_output'] = segfile
#
#     try:
#         color_label = label_to_colors(est_label, bigimage.numpy()[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
#     except:
#         color_label = label_to_colors(est_label, bigimage[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
#
#     if 'jpg' in f:
#         imsave(segfile, (color_label).astype(np.uint8), check_contrast=False)
#     elif 'png' in f:
#         imsave(segfile, (color_label).astype(np.uint8), check_contrast=False)
#
#     metadatadict['color_segmentation_output'] = segfile
#
#     segfile = segfile.replace('.png','_meta.npz')
#
#     np.savez_compressed(segfile, **metadatadict)
#
#     segfile = segfile.replace('_meta.npz','_res.npz')
#
#     # datadict['color_label'] = color_label
#     datadict['grey_label'] = est_label
#     # datadict['image_fullsize'] = bigimage
#     # datadict['image_targetsize'] = image
#
#     np.savez_compressed(segfile, **datadict)
#
#     segfile = segfile.replace('_res.npz','_overlay.png')
#
#     plt.imshow(bigimage); plt.imshow(color_label, alpha=0.5);
#     plt.axis('off')
#     # plt.show()
#     plt.savefig(segfile, dpi=200, bbox_inches='tight')
#     plt.close('all')
#



#
# # =========================================================
# def do_seg(f, model, metadatadict):
#
# 	# model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), BATCH_SIZE, NCLASSES)
#
# 	# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])
#
# 	# model.load_weights(weights)
#
# 	if NCLASSES==1:
# 		if 'jpg' in f:
# 			segfile = f.replace('.jpg', '_seg.tif')
# 		elif 'png' in f:
# 			segfile = f.replace('.png', '_seg.tif')
#
# 		segfile = os.path.normpath(segfile)
# 		# segfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks'))
#
# 		if os.path.exists(segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'certainty'))):
# 			print('%s exists ... skipping' % (segfile))
# 			pass
# 		else:
# 			print('%s does not exist ... creating' % (segfile))
#
# 		start = time.time()
#
# 		if N_DATA_BANDS<=3:
# 			image, w, h, bigimage = seg_file2tensor_3band(f)#, resize=True)
# 			if image is None:
# 				image = bigimage#/255
# 				#bigimage = bigimage#/255
# 				w = w.numpy(); h = h.numpy()
# 		else:
# 			image, w, h, bigimage = seg_file2tensor_4band(f, f.replace('aug_images', 'aug_nir'), resize=True )
# 			if image is None:
# 				image = bigimage#/255
# 				w = w.numpy(); h = h.numpy()
#
# 		print("Working on %i x %i image" % (w,h))
#
# 		image = standardize(image.numpy()).squeeze()
#
# 		E = []; W = []
# 		est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
# 		est_label = np.argmax(est_label, -1)
# 		E.append(est_label)
# 		W.append(1)
# 		est_label = np.fliplr(model.predict(tf.expand_dims(np.fliplr(image), 0) , batch_size=1).squeeze())
# 		est_label = np.argmax(est_label, -1)
# 		E.append(est_label)
# 		W.append(.5)
# 		est_label = np.flipud(model.predict(tf.expand_dims(np.flipud(image), 0) , batch_size=1).squeeze())
# 		est_label = np.argmax(est_label, -1)
# 		E.append(est_label)
# 		W.append(.5)
#
# 		# for k in np.linspace(100,int(TARGET_SIZE[0]),10):
# 		#     #E.append(np.roll(model.predict(tf.expand_dims(np.roll(image, int(k)), 0) , batch_size=1).squeeze(), -int(k)))
# 		#     E.append(model.predict(tf.expand_dims(np.roll(image, int(k)), 0) , batch_size=1).squeeze())
# 		#     W.append(2*(1/np.sqrt(k)))
# 		#
# 		# for k in np.linspace(100,int(TARGET_SIZE[0]),10):
# 		#     #E.append(np.roll(model.predict(tf.expand_dims(np.roll(image, -int(k)), 0) , batch_size=1).squeeze(), int(k)))
# 		#     E.append(model.predict(tf.expand_dims(np.roll(image, -int(k)), 0) , batch_size=1).squeeze())
# 		#     W.append(2*(1/np.sqrt(k)))
#
# 		K.clear_session()
#
# 		#E = [maximum_filter(resize(e,(w,h)), int(w/200)) for e in E]
# 		E = [resize(e,(w,h), preserve_range=True, clip=True) for e in E]
#
# 		#est_label = np.median(np.dstack(E), axis=-1)
# 		est_label = np.average(np.dstack(E), axis=-1, weights=np.array(W))
#
# 		est_label /= est_label.max()
#
# 		var = np.std(np.dstack(E), axis=-1)
#
# 		if np.max(est_label)-np.min(est_label) > .5:
# 			thres = threshold_otsu(est_label)
# 			print("Probability of land threshold: %f" % (thres))
# 		else:
# 			thres = .9
# 			print("Default threshold: %f" % (thres))
#
# 		conf = 1-est_label
# 		conf[est_label<thres] = est_label[est_label<thres]
# 		conf = 1-conf
#
# 		conf[np.isnan(conf)] = 0
# 		conf[np.isinf(conf)] = 0
#
# 		model_conf = np.sum(conf)/np.prod(conf.shape)
# 		print('Overall model confidence = %f'%(model_conf))
#
# 		out_stack = np.dstack((est_label,conf,var))
# 		outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'prob_stack'))
#
# 		try:
# 			os.mkdir(os.path.normpath(sample_direc+os.sep+'prob_stack'))
# 		except:
# 			pass
#
# 		imsave(outfile.replace('.tif','.png'),(100*out_stack).astype('uint8'),compression=9)
#
# 		#yellow = high prob land , high confidence, low variability
# 		#green = low prob of land, high confidence, low variability
# 		#purple = high prob land, low confidence, high variability
# 		#blue = low prob land, low confidence, high variability
# 		#red = high probability of land, low confidence, low variability
#
# 		thres_conf = threshold_otsu(conf)
# 		thres_var = threshold_otsu(var)
# 		print("Confidence threshold: %f" % (thres_conf))
# 		print("Variance threshold: %f" % (thres_var))
#
# 		land = (est_label>thres) & (conf>thres_conf) & (var<thres_conf)
# 		water = (est_label<thres)
# 		certainty = np.average(np.dstack((np.abs(est_label-thres) , conf , (1-var))), axis=2, weights=[2,1,1])
# 		outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'certainty'))
# 		try:
# 			os.mkdir(os.path.normpath(sample_direc+os.sep+'certainty'))
# 		except:
# 			pass
#
# 		imsave(outfile,(100*certainty).astype('uint8'),photometric='minisblack',compress=0)
#
# 		#land = remove_small_holes(land.astype('uint8'), 5*w)
# 		#land = remove_small_objects(land.astype('uint8'), 5*w)
# 		outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks'))
#
# 		try:
# 			os.mkdir(os.path.normpath(sample_direc+os.sep+'masks'))
# 		except:
# 			pass
#
# 		imsave(outfile.replace('.tif','.jpg'),255*land.astype('uint8'),quality=100)
#
# 		elapsed = (time.time() - start)/60
# 		print("Image masking took "+ str(elapsed) + " minutes")
#
# 	else: ###NCLASSES>1
#
# 		if 'jpg' in f:
# 			segfile = f.replace('.jpg', '_predseg.png')
# 		elif 'png' in f:
# 			segfile = f.replace('.png', '_predseg.png')
#
# 		segfile = os.path.normpath(segfile)
# 		segfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'out'))
#
# 		try:
# 			os.mkdir(os.path.normpath(sample_direc+os.sep+'out'))
# 		except:
# 			pass
#
# 		if os.path.exists(segfile):
# 			print('%s exists ... skipping' % (segfile))
# 			pass
# 		else:
# 			print('%s does not exist ... creating' % (segfile))
#
# 		# start = time.time()
#
# 		if N_DATA_BANDS<=3:
# 			image, w, h, bigimage = seg_file2tensor_3band(f)#, resize=True)
# 			image = image#/255
# 			bigimage = bigimage#/255
# 			w = w.numpy(); h = h.numpy()
# 		else:
# 			image, w, h, bigimage = seg_file2tensor_4band(f, f.replace('aug_images', 'aug_nir'), resize=True )
# 			image = image#/255
# 			bigimage = bigimage#/255
# 			w = w.numpy(); h = h.numpy()
#
# 		print("Working on %i x %i image" % (w,h))
#
# 		#image = tf.image.per_image_standardization(image)
# 		image = standardize(image.numpy())
#
# 		# est_label = model.predict(tf.expand_dims(image, 0 , batch_size=1).squeeze()
# 		est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
# 		K.clear_session()
#
# 		est_label = resize(est_label,(w,h))
#
# 		est_label = np.argmax(est_label, -1)
#
# 		# conf = np.max(est_label, -1)
# 		# conf[np.isnan(conf)] = 0
# 		# conf[np.isinf(conf)] = 0
# 		#est_label = np.argmax(est_label,-1)
#         #print(est_label.shape)
#
# 		#est_label = np.squeeze(est_label[:w,:h])
#
# 		class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477','#66AA00','#B82E2E', '#316395']
# 		#add classes for more than 10 classes
#
# 		class_label_colormap = class_label_colormap[:NCLASSES]
#
# 		try:
# 			color_label = label_to_colors(est_label, bigimage.numpy()[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
# 		except:
# 			color_label = label_to_colors(est_label, bigimage[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
#
# 		if 'jpg' in f:
# 			imsave(segfile, (color_label).astype(np.uint8), check_contrast=False)
# 		elif 'png' in f:
# 			imsave(segfile, (color_label).astype(np.uint8), check_contrast=False)
