import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os

# Optimization-Inspired Dilated Deep Network for Compressive Sensing of Color Images

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ADConv(torch.nn.Module):
    def __init__(self,channels,dialtedTypes):
        super(ADConv, self).__init__()

        self.channels = channels

        # Convolutional kernel library 32*32*3*3
        self.conv = nn.Parameter(init.xavier_normal_(torch.Tensor(channels, channels, 3, 3)))

        # Three types of dialted convolution:1,2,3
        self.dilatedConvType = dialtedTypes
        
        # 32*3,Convolutional assignment parameters, dividing the 32 convolutions into three categories
        self.dilatedPara = nn.Parameter(init.xavier_normal_(torch.Tensor(channels, self.dilatedConvType)))
        self.softmax = nn.Softmax(dim=1)

        self.idxDilated = None
        self.idxOffset = []

    def exeTrain(self, data):
        # Output the assignment of the three types of convolution kernels by softmax
        soft = self.softmax(self.dilatedPara)

        _,indexs = torch.topk(soft,1)
        indexs = indexs.view(self.channels) 

        size = data.size() 
        result = torch.zeros([size[0], 0,size[2],size[3]], dtype=torch.float,device=device)

        # After convolving the three types of convolution kernels on the input, concat to output together
        for i in range(self.dilatedConvType):

            dilation = i + 1
            idxDilate = (indexs==i).nonzero(as_tuple=False).view(-1)          
            

            if idxDilate.size()[0] > 0:          
                conv = torch.index_select(self.conv,0,idxDilate)                   
                
                # To maintain the gradient on dilated paras, the result of the convolution is multiplied by the mean value of the corresponding value of dilated paras
                mul = torch.mean(torch.index_select(self.dilatedPara,0,idxDilate)[:,i])+1                                  
                result = torch.cat((result
                                ,mul * F.conv2d(data, conv, padding=dilation,dilation=dilation)),1)  

        return result 

    def reference(self, data):
        if self.idxDilated is None:

            self.idxDilated = []

            # Output the assignment of the three types of convolution kernels by softmax
            soft = self.softmax(self.dilatedPara)

            _,indexs = torch.topk(soft,1)
            indexs = indexs.view(self.channels) 

            # After convolving the three types of convolution kernels on the input, concat to output together
            for i in range(self.dilatedConvType):

                dilation = i + 1
                idxDilate = (indexs==i).nonzero(as_tuple=True)[0]

                self.idxDilated.append(idxDilate)
                                
        size = data.size() 
        result = torch.zeros([size[0], 0,size[2],size[3]], dtype=torch.float,device=device)

        # After convolving the three types of convolution kernels on the input, concat to output together
        for i in range(self.dilatedConvType):

            dilation = i + 1
            idxDilate = self.idxDilated[i]          
            

            if idxDilate.size()[0] > 0:          
                conv = torch.index_select(self.conv,0,idxDilate)                   
                
                # To maintain the gradient on dilated paras, the result of the convolution is multiplied by the mean value of the corresponding value of dilated paras
                mul = torch.mean(torch.index_select(self.dilatedPara,0,idxDilate)[:,i])+1                                  
                result = torch.cat((result
                                ,mul * F.conv2d(data, conv, padding=dilation,dilation=dilation)),1)  

        return result        

    def forward(self, data):              

        if self.training:      
            return self.exeTrain(data) 
        else:
            return self.reference(data)              


# Define Basic reconstruct Block
class BasicBlock(torch.nn.Module):
    def __init__(self,BLOCK_SIZE):
        super(BasicBlock, self).__init__()

        self.BLOCK_SIZE=BLOCK_SIZE

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))        
        self.t = nn.Parameter(torch.Tensor([1.0]))
        self.mergeScale = nn.Parameter(torch.Tensor([1.0]))
        self.mergeGScale = nn.Parameter(torch.Tensor([0.1]))
        

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 3, 3, 3)))        

        self.conv1_forward = ADConv(32,1)        
        self.conv2_forward = ADConv(32,3)        

        self.conv1_backward = ADConv(32,1)        
        self.conv2_backward = ADConv(32,3)

        self.conv1_G = ADConv(32,1)    
        self.conv2_G = ADConv(32,3)           
        self.conv3_G = nn.Parameter(init.xavier_normal_(torch.Tensor(3, 32, 3, 3)))

    def forward(self, xprev, x, PhiWeight, PhiTWeight, PhiTb):

        tplus = (1+torch.sqrt(1+4*self.t*self.t))/2
        xi = (self.t-1)/tplus
        deltax = x-xprev

        zeta = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight, PhiTWeight,self.BLOCK_SIZE)
        zeta = zeta - self.lambda_step * xi * PhiTPhi_fun(deltax, PhiWeight, PhiTWeight,self.BLOCK_SIZE)
        zeta = zeta + xi * deltax
        zeta = zeta + self.lambda_step * PhiTb

        x = zeta
        
        x_input = x

        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        
        x = self.conv1_forward(x_D)
        x = F.relu(x)        
        x_forward = self.conv2_forward(x)


        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
                
        x = self.conv1_forward(x)
        x = F.relu(x)                
        x_backward = self.conv2_backward(x)
        
        x = self.conv1_G(F.relu(x_backward))        
        x = self.conv2_G(F.relu(x))        

        x_G = F.conv2d(x, self.conv3_G, padding=1)

        x_pred = x_input + x_G*self.mergeScale
        
        x = self.conv1_backward(x_forward)
        x = F.relu(x)                
        x_D_est = self.conv2_backward(x)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


# Define OIDDN
class OIDDN(torch.nn.Module):

    def __init__(self, LayerNo, M, BLOCK_SIZE):
        
        super(OIDDN, self).__init__()

        N = BLOCK_SIZE * BLOCK_SIZE

        self.Phir = nn.Parameter(init.xavier_normal_(torch.Tensor(M, N)))
        self.Phig = nn.Parameter(init.xavier_normal_(torch.Tensor(M, N)))
        self.Phib = nn.Parameter(init.xavier_normal_(torch.Tensor(M, N)))
        self.Phi_scale = nn.Parameter(torch.Tensor([1.0]))
        

        onelayer = []
        self.LayerNo = LayerNo
        self.M = M
        self.N = N
        self.BLOCK_SIZE = BLOCK_SIZE

        for i in range(LayerNo):
            onelayer.append(BasicBlock(BLOCK_SIZE))

        self.fcs = nn.ModuleList(onelayer)
        self.shuffle = torch.nn.PixelShuffle(BLOCK_SIZE)

    def forward(self, x):

        origX = x

        # Sampling-subnet
        Phir = self.Phir * self.Phi_scale        
        Phig = self.Phig * self.Phi_scale        
        Phib = self.Phib * self.Phi_scale

        PhirWeight = Phir.contiguous().view(self.M, 1, self.BLOCK_SIZE, self.BLOCK_SIZE)
        PhigWeight = Phig.contiguous().view(self.M, 1, self.BLOCK_SIZE, self.BLOCK_SIZE)
        PhibWeight = Phib.contiguous().view(self.M, 1, self.BLOCK_SIZE, self.BLOCK_SIZE)

        Phixr = F.conv2d(x[:,0:1,:,:], PhirWeight, padding=0, stride=self.BLOCK_SIZE, bias=None)
        Phixg = F.conv2d(x[:,1:2,:,:], PhigWeight, padding=0, stride=self.BLOCK_SIZE, bias=None)
        Phixb = F.conv2d(x[:,2:3,:,:], PhibWeight, padding=0, stride=self.BLOCK_SIZE, bias=None)

        # Initialization-subnet
        PhiWeight = torch.cat((
            PhirWeight,
            PhigWeight,
            PhibWeight),dim=1)
            

        PhiTWeight = torch.cat((
            Phir.t().contiguous().view(self.N, self.M, 1, 1),
            Phig.t().contiguous().view(self.N, self.M, 1, 1),
            Phib.t().contiguous().view(self.N, self.M, 1, 1)),dim=0)
        

        PhiTb = torch.cat((
            self.shuffle(F.conv2d(Phixr, Phir.t().contiguous().view(self.N, self.M, 1, 1), padding=0, bias=None)),
            self.shuffle(F.conv2d(Phixg, Phig.t().contiguous().view(self.N, self.M, 1, 1), padding=0, bias=None)),
            self.shuffle(F.conv2d(Phixb, Phib.t().contiguous().view(self.N, self.M, 1, 1), padding=0, bias=None))),
            dim=1)            
        
        x = PhiTb    

        # Recovery-subnet
        layers_sym = []   # for computing symmetric loss        
        xprev = x
        size = x.size()
        for i in range(self.LayerNo):            
            
            [x1, layer_sym] = self.fcs[i](xprev, x, PhiWeight, PhiTWeight, PhiTb)            
            xprev = x
            x=x1

            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym, [Phir,Phig,Phib]]


def PhiTPhi_fun(x, PhiW, PhiTW,BLOCK_SIZE):

    N = BLOCK_SIZE * BLOCK_SIZE

    phir = F.conv2d(x[:,0:1,:,:], PhiW[:,0:1,:,:], padding=0,stride=BLOCK_SIZE, bias=None)
    phig = F.conv2d(x[:,1:2,:,:], PhiW[:,1:2,:,:], padding=0,stride=BLOCK_SIZE, bias=None)
    phib = F.conv2d(x[:,2:3,:,:], PhiW[:,2:3,:,:], padding=0,stride=BLOCK_SIZE, bias=None)

    xtempr = F.conv2d(phir, PhiTW[0:N,:,:,:], padding=0, bias=None)
    xtempg = F.conv2d(phig, PhiTW[N:N*2,:,:,:], padding=0, bias=None)
    xtempb = F.conv2d(phib, PhiTW[N*2:N*3,:,:,:], padding=0, bias=None)

    temp = torch.cat(
        (
            xtempr,xtempg,xtempb
        ),dim=1
    )

    return torch.nn.PixelShuffle(BLOCK_SIZE)(temp)