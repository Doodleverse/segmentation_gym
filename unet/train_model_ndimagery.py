



root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/segmentation_zoo",title = "Select directory of 'nd' npz files")
imdir = root.filename
print(imdir)
root.withdraw()

# Out[9]: '/media/marda/TWOTB/USGS/SOFTWARE/Projects/UNets/coast_train_datasets/datasetF_L8/all_data/npz_merged'
