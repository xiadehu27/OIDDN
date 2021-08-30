import h5py
import numpy as np
import matplotlib.image as mpimg
import os
import glob

# Generate color image training samples

sampleNum=37
block_size=96
step_size=47
img_folder = 'data/BSD500'

x = np.empty([0, 3, block_size, block_size])

filepaths=[]

file_formats = ('*.bmp', '*.png', '*.jpg')
for file_format in file_formats:
    filepaths.extend(glob.glob(img_folder + '/' + file_format))
ImgNum = len(filepaths)

def blockImags(imgData):
    shape = imgData.shape    
    blockHeightNum = int((shape[0]-block_size+step_size)/step_size)
    blockWidthNum = int((shape[1]-block_size+step_size)/step_size)
    x = np.empty([blockHeightNum*blockWidthNum, 3, block_size, block_size])

    idx = 0
    for i in range(blockWidthNum):
        for j in range(blockHeightNum):
            block = np.transpose(np.array(imgData[j*step_size:j*step_size+block_size,i*step_size:i*step_size+block_size,:]), (2,0,1))
            x[idx,:,:,:]=block
            idx=idx+1
    return x

i = 1 
for filepath in filepaths:    
    image = mpimg.imread(filepath)
    # Original image + Flip
    image_ = image
    x = np.concatenate((x,blockImags(image_)), axis=0)
    x = np.concatenate((x,blockImags(np.flip(image_,1))), axis=0)
    # Rotate 90 degrees + flip
    image_ = np.rot90(image,k=1,axes=(0,1))
    x = np.concatenate((x,blockImags(image_)), axis=0)
    x = np.concatenate((x,blockImags(np.flip(image_,1))), axis=0)
    # Rotate 180 degrees + flip
    image_ = np.rot90(image,k=2,axes=(0,1))
    x = np.concatenate((x,blockImags(image_)), axis=0)
    x = np.concatenate((x,blockImags(np.flip(image_,1))), axis=0)
    # Rotate 270 degrees + flip
    image_ = np.rot90(image,k=3,axes=(0,1))
    x = np.concatenate((x,blockImags(image_)), axis=0)
    x = np.concatenate((x,blockImags(np.flip(image_,1))), axis=0)

    print("image:%d,block_size:%d" %(i,x.shape[0]))

    i = i+1
    if i == sampleNum:
        break

if np.max(x) > 10:
    x = x / 255.0  

save_path = './data/py_%d_%d_%d_cf.hdf5' % (sampleNum,block_size,x.shape[0])      

hf = h5py.File(save_path, 'a') 

dset = hf.create_dataset('data', data=x)

hf.close()