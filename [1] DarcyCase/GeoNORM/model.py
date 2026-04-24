# -*- coding: utf-8 -*-
"""
@author: DingYe
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
# ----------------- Activation -----------------
ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1), 'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU
}

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) 
                                      for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

class SpectralF1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1,geo_dim):
        super(SpectralF1d, self).__init__()
   
     
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
  
    def compl_mul1d(self, input, weights):
      
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
      
        x_ft = torch.fft.rfft(x)
    
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
  
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
      
        x = torch.fft.irfft(out_ft, n=x.size(-1))

        return x
    
class SpectralM1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1,geo_dim):
        super(SpectralM1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
     

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):

        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, basis):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)


        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        out_r = torch.real(out_ft[:, :, :self.modes1])
        out_i = torch.imag(out_ft[:, :, :self.modes1])
        out=  out_r+ out_i

        x = out @ basis.T
        return x
    
from torch_geometric.nn import GCNConv
class MeshGCNEncoderTorch(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, Nk=6):
        super().__init__()

        # ---------- encoder (your original) ----------
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        self.out1 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.out2 = nn.Linear(2*hidden_dim, Nk) 



    def forward(self, meshPoints, adj_norm, LBO_INVERSE, LBO_MATRIX):

        x = self.input_proj(meshPoints)
        x = F.gelu(self.gcn1(x, adj_norm))
        x = self.gcn2(x, adj_norm)  
        
        x = self.out1(x) 
        x = self.out2(x) 
        x = LBO_INVERSE @ x
        x = F.softmax(x, dim=1)  

        return x
          


class GeoModSpectralOperator(nn.Module):

    def __init__(self,
                 C,
                 M,
                 geodim):
        super().__init__()
        self.C = C
        self.M = M
        self.Nk = 6
    
        self.scale = (1 / (C*C))
        self.weights = nn.Parameter(self.scale * torch.rand(self.Nk, C, C, M, dtype=torch.float))
        self.encoder = MeshGCNEncoderTorch(3,self.C,self.Nk)
        self.eps = 1e-6

    def forward(self,
                F,
                LBO_MATRIX,
                LBO_INVERSE,
                meshPoints,
                adj_norm):
                
        B, N, C = F.shape
        
        x = LBO_INVERSE @ F  
        C_f_i = x.permute(0, 2, 1) ## (B, C, M)

        geoT = self.encoder(meshPoints, adj_norm, LBO_INVERSE, LBO_MATRIX)
        weights = torch.einsum('nijm,mn->ijm', self.weights, geoT)


        C_g_i = torch.einsum("bix,iox->box", C_f_i, weights)
 
        G =  C_g_i @ LBO_MATRIX.T  # (B, C, N)  
        
        G = G.permute(0, 2, 1) # (B, N, C) 

        return G


from utils import SimpleGCN,SimpleGAT
    
class Convolution_block(nn.Module):
    def __init__(self, 
                 lbo_bases,
                 lbo_inver,
                 num_channels, 
                 num_modes = 64, 
                 dropout = 0.0): 
        super().__init__()
        
        self.norm1 = nn.LayerNorm(num_channels)
      
        self.SpectralConv = GeoModSpectralOperator(C=num_channels, M=num_modes, geodim = num_channels)


        self.SpatialConv = SimpleGCN(num_channels, num_channels*2, num_channels)
        
        self.lbo_bases = lbo_bases
        self.lbo_inver = lbo_inver
        
    def forward(self, x, meshPoints, sdfs, adj_norm, MATRIX_Phy, INVERSE_Phy):
        #  x : B N C
        x1 = self.SpectralConv(x, MATRIX_Phy, INVERSE_Phy, meshPoints, adj_norm) #B ,C, N
        x1 = self.norm1(x1) 
        x2 = self.SpatialConv(x, adj_norm) #B ,N, C
        x = x1 + x2
        
        return x
        
class NORMGCN_Layer(nn.Module):
    def __init__(
                    self,
                    lbo_bases,
                    lbo_inver,
                    num_channels,
                    num_modes,
                    dropout,
                    act='gelu',
                    mlp_ratio=2
                ):
        super(NORMGCN_Layer, self).__init__()
        
        self.ln_1 = nn.LayerNorm(num_channels)
        self.Conv = Convolution_block(
                                        lbo_bases = lbo_bases,
                                        lbo_inver = lbo_inver,
                                        num_channels = num_channels, 
                                        num_modes = num_modes, 
                                        dropout = dropout)
        
        self.ln_2 = nn.LayerNorm(num_channels)
        self.mlp  = MLP(num_channels, num_channels * mlp_ratio, num_channels, n_layers = 0, res=False, act=act)
        
    def forward(self, fx, meshPoints, sdfs, adj_norm, MATRIX_Phy, INVERSE_Phy):
        fx = self.Conv(self.ln_1(fx), meshPoints, sdfs, adj_norm, MATRIX_Phy, INVERSE_Phy) + fx  #B ,N, C
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx    
    
class NORM_net(nn.Module):
    def __init__(self, modes, width):
        super(NORM_net, self).__init__()

        self.modes1 = modes
        self.width = width
     
        LBO_MATRIX = None
        LBO_INVERSE = None
        self.fc0 = nn.Linear(4, self.width) 
        self.fc01 = nn.Linear(self.width, self.width)#self.width
    

        self.conv0 = SpectralF1d(self.width, self.width, self.modes1, self.modes1)
        self.conv1 = SpectralM1d(self.width, self.width, self.modes1, self.modes1)
        

        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        

        self.ln_1 = nn.LayerNorm(self.width)
        self.ln_2 = nn.LayerNorm(self.width)

 
        self.blocks_condnorm = nn.ModuleList([NORMGCN_Layer(            
                                                    lbo_bases = LBO_MATRIX,
                                                    lbo_inver = LBO_INVERSE,
                                                    num_channels = self.width,
                                                    num_modes    = self.modes1,
                                                    dropout      = 0.0,
                                                    act='gelu',
                                                    mlp_ratio=2,
                                                ) for _ in range(4)])                        

    def forward(self, x, MATRIX_Phy, INVERSE_Phy, meshPoints,adj_norm):
        
        sdfs = None

        self.mesh = meshPoints     
        mesh = self.mesh.unsqueeze(0) 
        batch_zise = x.shape[0]
        mesh = mesh.repeat(batch_zise, 1, 1)
        x = torch.cat((x, mesh), dim=-1)

        
        x = self.fc0(x)
        x = self.fc01(x)

        
        for block in self.blocks_condnorm:
            x = block(x, meshPoints, sdfs, adj_norm, MATRIX_Phy, INVERSE_Phy)
                
        
        x = self.fc1(self.ln_1(x))
        x = F.gelu(x)
        x = self.fc2(x)

        loss1 = 0
 
        return x ,loss1

  
    
