import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
from time import time
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
from DeephomographyDataset import DeephomographyDataset
from OIDN_def import OIDN
import matplotlib.pyplot as plt

parser = ArgumentParser(description='OIDN')

parser.add_argument('--start_epoch', type=int, default=0 ,help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=300, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of OIDN')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=25, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--block_size', type=str, default='32', help='basic block size of convolution')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--save_interval', type=int, default=1, help='interval of saving model')
parser.add_argument('--dataset_name', type=str, default='trainAll_33_rgb_10.h5', help='trained or pre-trained model name')
parser.add_argument('--net_name', type=str, default='CS_OPINE_CI_Net_10_ista_binary', help='net name')


args = parser.parse_args()


start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
dataset_name = args.dataset_name
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
nrtrain = 88912   # number of training blocks
batch_size = 64

Training_data_Name = dataset_name


model = OIDN(layer_num, M, block_size)
model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
model = model.to(device)


num_count = 0
for para in model.parameters():
    num_count += 1
    print('Layer %d' % num_count)
    print(para.size())

if (platform.system() =="Windows"):
    
    trainDataset = DeephomographyDataset('%s/%s' % (args.data_dir, Training_data_Name),loadAllToMemory=True)

    rand_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, num_workers=0,shuffle=True,pin_memory=True)
    
else:
    
    Dataset1 = h5py.File('./%s/%s' % (args.data_dir, Training_data_Name),'r',swmr=True)

    rand_loader = DataLoader(dataset=Dataset1, batch_size=batch_size, num_workers=0, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/%s_layer_%d_group_%d_ratio_%d" % (args.model_dir,net_name,layer_num, group_num, cs_ratio)

log_file_name = "./%s/Log_CS_OPINE_Net_plus_layer_%d_group_%d_ratio_%d.txt" % (args.log_dir, layer_num, group_num, cs_ratio)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)),strict=False)


Eye_I = torch.eye(M).to(device)

gamma = torch.Tensor([0.01]).to(device)
mu = torch.Tensor([0.01]).to(device)
# Training loop
for epoch_i in range(start_epoch+1, end_epoch+1):
    for data in rand_loader:

        start = time()  
        batch_x = data.to(device)
        
        [x_final, x_loss,loss_layers_sym, phi,loss_orth] = model(batch_x)
       

        # Compute and print loss
        loss_discrepancy = torch.mean(x_loss)
        loss_orth = torch.mean(loss_orth)
                
        loss_symmetry = 0
        for k in range(len(loss_layers_sym)):
            loss_symmetry += torch.mean(torch.pow(loss_layers_sym[k], 2))                                

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        torch.autograd.backward((loss_discrepancy,torch.mul(mu,loss_symmetry),torch.mul(mu,loss_orth)))
        optimizer.step()

        model_time = (time()-start) 

        output_data = "[%02d/%02d] Discrepancy Loss: %.8f, Symmetry Loss: %.8f, Orth Loss: %.8f, time: %.4f" % (epoch_i, end_epoch, loss_discrepancy.item(), loss_symmetry.item(), loss_orth.item(), model_time)
        print(output_data)

    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
