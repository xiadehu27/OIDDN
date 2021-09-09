
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import math
from torch.nn import init
import copy
from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from OIDDN_Def import OIDDN

parser = ArgumentParser(description='OPINE-Net-plus')

parser.add_argument('--epoch_start', type=int, default=27, help='epoch number of model')
parser.add_argument('--epoch_num', type=int, default=35, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of OPINE-Net-plus') 
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cs_ratio', type=int, default=1, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='1', help='gpu index')
parser.add_argument('--block_size', type=str, default='32', help='basic block size of convolution')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set5', help='name of test set')
parser.add_argument('--net_name', type=str, default='OIDDN_withoutG', help='net name')


args = parser.parse_args()


epoch_start = args.epoch_start
epoch_num = args.epoch_num
learning_rate = args.learning_rate
layer_num = args.layer_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name
net_name = args.net_name
block_size  = int(args.block_size)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 5: 55, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}

N = block_size * block_size

if N==1089:
    M = ratio_dict[cs_ratio]
else:
    M = (int)(N * cs_ratio / 100)
nrtrain = 88912
batch_size = 64


model = OIDDN(layer_num, M,block_size)
model = nn.DataParallel(model)
model = model.to(device)
# model.eval()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/%s_layer_%d_ratio_%d" % (args.model_dir,net_name,layer_num, cs_ratio)


test_dir = os.path.join(args.data_dir, test_name)

filepaths=[]

file_formats = ('*.bmp', '*.png', '*.jpg')
for file_format in file_formats:
    filepaths.extend(glob.glob(test_dir + '/' + file_format))

result_dir = os.path.join(args.result_dir, test_name, net_name, str(layer_num))
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

log_file_name = "%s/ratio_%d_log.log" % (result_dir,cs_ratio)    

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
TIME_All = np.zeros([1, ImgNum], dtype=np.float32)

# Split the color image into non-overlapping image blocks by the specified block size and return the related information
def img_to_blocks(Iorg,block_size):    
    [row, col,channel] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad,3])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad,3])), axis=0)
    [row_new, col_new] = Ipad.shape[0:2]

    return [Iorg, row, col, Ipad, row_new, col_new]     

with torch.no_grad():     
        
    for epoch_loop_no in range(epoch_start,epoch_num+1,1):

        model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epoch_loop_no)),strict=False)

        # A small amount of randomly generated data is passed into the model for inference to avoid 
        # the time consuming initial execution of inference, 
        # which affects the accuracy of statistical data.
        model(torch.zeros([1,3,32,32],dtype=torch.float32, device=device))

        for img_no in range(0,ImgNum):        
        
            imgName = filepaths[img_no]

            Img = mpimg.imread(imgName)           

            [Iorg, row, col, Ipad, row_new, col_new] = img_to_blocks(Img,block_size)          
            
            Icol = Ipad   

            if np.max(Icol) > 10:
                Icol = Icol / 255.0
            
            icolTensor = torch.from_numpy(Icol)  
            input_x = torch.zeros([1,3,row_new,col_new],dtype=torch.float32, device=device) 

            # Prepare input data by model input shape
            input_x[:,0:1,:,:] = icolTensor[:,:,0:1].view(1,1,row_new,col_new)
            input_x[:,1:2,:,:] = icolTensor[:,:,1:2].view(1,1,row_new,col_new)
            input_x[:,2:3,:,:] = icolTensor[:,:,2:3].view(1,1,row_new,col_new)


            start = time()              
            [output_x, loss_layers_sym, phis] = model(input_x)
          
            end = time()                                       

            model_time = (end-start) 

            Prediction_value = output_x[:,:,0:row,0:col]
            
            X_rec = torch.zeros([row,col,3],dtype=torch.float32, device=device)  

            # Converting model output to color image three-channel structure
            X_rec[:,:,0:1] = Prediction_value[:,0:1,:,:].view(row,col,1)
            X_rec[:,:,1:2] = Prediction_value[:,1:2,:,:].view(row,col,1)
            X_rec[:,:,2:3] = Prediction_value[:,2:3,:,:].view(row,col,1)

            X_rec = X_rec.cpu().data.numpy()

            X_rgb = np.clip(X_rec*255.0,0,255).astype(np.uint8)

            if np.max(Img) < 10:
                Img = np.clip(Img * 255.0,0,255).astype(np.uint8)

            rec_PSNR = peak_signal_noise_ratio(X_rgb, Img,data_range=255)           
            rec_SSIM = structural_similarity(X_rgb, Img,data_range=255,multichannel=True)                     

            if epoch_loop_no == epoch_num:
                resultName = imgName.replace(os.path.join(args.data_dir, test_name), result_dir)            
                mpimg.imsave("%s_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.2f.png" % (resultName, cs_ratio, epoch_loop_no, rec_PSNR, rec_SSIM), X_rgb.astype(np.uint8))

            del output_x
            del input_x

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            TIME_All[0, img_no] = model_time

        log_content = "epoch %03d, PSNR is %.5f,SSIM is %.5f, time is %.4f" % (epoch_loop_no, np.mean(PSNR_All[:,:]),np.mean(SSIM_All[:,:]),np.mean(TIME_All[:,:]))
        print(log_content)
        output_file = open(log_file_name, 'a')
        output_file.write(log_content+"\n")
        output_file.close()

print("CS Reconstruction End")