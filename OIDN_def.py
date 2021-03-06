import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define Basic reconstruct block
class BasicBlock(torch.nn.Module):
    def __init__(self,BLOCK_SIZE):
        super(BasicBlock, self).__init__()

        self.BLOCK_SIZE=BLOCK_SIZE

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))        
        self.t = nn.Parameter(torch.Tensor([1.0]))
        self.mergeScale = nn.Parameter(torch.Tensor([1.0]))
        

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 3, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv1_G = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_G = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))        
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

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x = F.conv2d(F.relu(x_backward), self.conv1_G, padding=1)
        x = F.conv2d(F.relu(x), self.conv2_G, padding=1)
        x_G = F.conv2d(x, self.conv3_G, padding=1)

        x_pred = x_input + x_G*self.mergeScale

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


# Define OIDN 
class OIDN(torch.nn.Module):

    def __init__(self, LayerNo, M, BLOCK_SIZE):
        
        super(OIDN, self).__init__()

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