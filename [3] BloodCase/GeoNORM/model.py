# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 17:27:15 2025

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



#-----------------------------
# Full model
# -----------------------------
from utils import MeshNO
class GeoModSpectralOperator(nn.Module):

    def __init__(self,
                 C,
                 M,
                 geodim, 
                 geocond=True):
        super().__init__()
        self.C = C
        self.M = M
        self.geocond = geocond
    
        self.encoder = MeshNO( self.M, self.C, geodim)
        self.scale = (1 / (C*C))
        self.weights = nn.Parameter(self.scale * torch.rand(C, C, M, dtype=torch.float))

        self.eps = 1e-6

    def forward(self,
                F,
                LBO_MATRIX,
                LBO_INVERSE,
                meshPoints,
                sdfs):

        B, N, C = F.shape
        
        x = LBO_INVERSE @ F  
        C_f_i = x.permute(0, 2, 1) 

        if self.geocond == True:
           geoT = self.encoder(meshPoints,sdfs, LBO_MATRIX, LBO_INVERSE)
           weights = torch.einsum('ijm,m->ijm', self.weights, geoT)
        else:
           weights =  self.weights

        C_g_i = torch.einsum("bix,iox->box", C_f_i, weights)
        G =  C_g_i @ LBO_MATRIX.T  # (B, C, N)  
        G = G.permute(0, 2, 1) # (B, N, C) 

        return G


from utils import SimpleGCN,SimpleGAT
    
class Convolution_block(nn.Module):
    def __init__(self, 
                 model_type,
                 lbo_bases,
                 lbo_inver,
                 num_channels, 
                 num_modes = 64, 
                 dropout = 0.0): 
        super().__init__()
        
        self.norm1 = nn.LayerNorm(num_channels)
        if model_type == 'condnorm':
            self.SpectralConv = GeoModSpectralOperator(C=num_channels, M=num_modes, geodim = num_channels, geocond=True)
        elif model_type == 'norm':
            self.SpectralConv = GeoModSpectralOperator(C=num_channels, M=num_modes,geodim = num_channels, geocond=False)
        else:
            raise ValueError("Please check 'model_type' !")
        self.SpatialConv = SimpleGCN(num_channels, num_channels*2, num_channels)
        
        self.lbo_bases = lbo_bases
        self.lbo_inver = lbo_inver
        
    def forward(self, x, meshPoints, sdfs, adj_norm, MATRIX_Phy, INVERSE_Phy):
        #  x : B N C
        x1 = self.SpectralConv(x, MATRIX_Phy, INVERSE_Phy, meshPoints, sdfs) #B ,C, N
        x1 = self.norm1(x1) 
        x2 = self.SpatialConv(x, adj_norm) #B ,N, C
        x = x1 + x2
        
        return x
    
    
        
class NORMGCN_Layer(nn.Module):
    def __init__(
                    self,
                    model_type,
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
                                        model_type = model_type,
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
class BCNet(nn.Module):
    def __init__(self,width):
        super().__init__()
        c = 32
        self.blocks = nn.Sequential(
            nn.Conv1d(1, c, 9, padding=4), nn.BatchNorm1d(c), nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),                       # 260 -> 130

            nn.Conv1d(c, c*2, 9, padding=4), nn.BatchNorm1d(c*2), nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),                       # 130 -> 65

            nn.Conv1d(c*2, c*4, 9, padding=4), nn.BatchNorm1d(c*4), nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),                       # 65 -> 32

            nn.Conv1d(c*4, c*4, 9, padding=4), nn.BatchNorm1d(c*4), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)                # 任意 -> 1
        )
        self.fc = nn.Linear(c*4, width)
    def forward(self, x):
        # x: (B, 260, 1)
        x = x.permute(0, 2, 1)          # -> (B, 1, 260)
        y = self.blocks(x)              # -> (B, c*4, 1)
        y = y.view(y.size(0), -1)       # -> (B, c*4)
        return self.fc(y)               # -> (B, width)


class GeomBCProcessor(nn.Module):

    def __init__(self,width):
        super().__init__()
        self.bc_net = BCNet(width)

    def forward(self,N,x):

        B, _, _ = x.shape
        # N, _    = meshPoints.shape
        bc_vec = self.bc_net(x)              # (B, width)
        bc_vec = bc_vec.unsqueeze(1).expand(-1, N, -1) # (B, N, width)
        

        return bc_vec
class NORM_net(nn.Module):
    def __init__(self, modes, width):
        super(NORM_net, self).__init__()

        self.modes1 = modes
        self.width = width
        LBO_MATRIX = None
        LBO_INVERSE = None
     
        self.fc0 = nn.Linear(4, self.width) 
        self.fc01 = nn.Linear(self.width, self.width//2)#self.width
    


        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        

        self.ln_1 = nn.LayerNorm(self.width)
        self.ln_2 = nn.LayerNorm(self.width)

        self.blocks_norm = nn.ModuleList([NORMGCN_Layer(
                                                    model_type = 'norm',
                                                    lbo_bases = LBO_MATRIX,
                                                    lbo_inver = LBO_INVERSE,
                                                    num_channels = self.width,
                                                    num_modes    = self.modes1,
                                                    dropout      = 0.0,
                                                    act='gelu',
                                                    mlp_ratio=2,
                                                ) for _ in range(2)])   
        self.blocks_condnorm = nn.ModuleList([NORMGCN_Layer(
                                                    model_type = 'condnorm', 
                                                    lbo_bases = LBO_MATRIX,
                                                    lbo_inver = LBO_INVERSE,
                                                    num_channels = self.width,
                                                    num_modes    = self.modes1,
                                                    dropout      = 0.0,
                                                    act='gelu',
                                                    mlp_ratio=2,
                                                ) for _ in range(2)]) 
                       
        self.proc = GeomBCProcessor(self.width//2)
        
    def forward(self, x, MATRIX_Phy, INVERSE_Phy, meshPoints,adj_norm ,ilbo,sdfs):
        

        BBT, NCC, CNN = x.shape
        NCC,M = MATRIX_Phy.shape
                      
       
        bc_vec = self.proc(NCC, x)  

        
        pts_single = meshPoints.unsqueeze(0)          
        pts_batch = pts_single.expand(BBT, -1, -1)       

        sdfs_single = sdfs.unsqueeze(0)          
        sdfs_batch = sdfs_single.expand(BBT, -1, -1)
        
        
        geoinp = torch.cat([pts_batch,sdfs_batch], dim=-1) 

        x = self.fc0(geoinp)
        x  = F.gelu(x)
        x = self.fc01(x)
        
        
        x = torch.cat([x,bc_vec], dim=-1)  

        
        for block in self.blocks_norm:
            x = block(x, meshPoints, sdfs, adj_norm, MATRIX_Phy, INVERSE_Phy)
        for block in self.blocks_condnorm:
            x = block(x, meshPoints, sdfs, adj_norm, MATRIX_Phy, INVERSE_Phy)
                
        
        x = self.fc1(self.ln_1(x))
        x = F.gelu(x)
        x = self.fc2(x)

        loss1 = 0
 
        return x ,loss1


    
