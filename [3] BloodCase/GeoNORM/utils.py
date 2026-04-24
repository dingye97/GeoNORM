# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 17:27:41 2025

@author: DingYe
"""
import torch
import operator
from functools import reduce
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1), 'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU
}
class SamplewiseNormalizer:
    def __init__(self):
        self.input_norm_info = []   # [(mean, std), ...]
        self.output_norm_info = []  # [(mean, std), ...]

    def normalize_dataset(self, inputdata, outputdata):
        inputdata_norm, outputdata_norm = [], []

        for u, v in zip(inputdata, outputdata):
            
            u_mean = u.mean()
            u_std = u.std() + 1e-8
            u_norm = (u - u_mean) / u_std
            inputdata_norm.append(u_norm)
            self.input_norm_info.append((u_mean, u_std))

            v_mean = v.mean()
            v_std = v.std() + 1e-8
            v_norm = (v - v_mean) / v_std
            outputdata_norm.append(v_norm)
            self.output_norm_info.append((v_mean, v_std))

        return inputdata_norm, outputdata_norm

    def denormalize_single_output(self, v_norm, idx):
        """
        逐个样本反归一化输出
        :param v_norm: [1, N, 1] 归一化后的输出张量
        :param idx: 当前样本索引
        :return: 反归一化后的输出张量
        """
        v_mean, v_std = self.output_norm_info[idx]
        return v_norm * v_std + v_mean


class Approximation_block(nn.Module):
    
    def __init__ (self,in_channels, out_channels, modes, geo_dim):
        super(Approximation_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes1 = modes
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.float))
        # self.kernel_generator = AttentionConditionedKernel1(in_channels, out_channels, modes, geo_dim)



    def forward(self, x, LBO_MATRIX, LBO_INVERSE):
            
        ################################################################
        # Encode[10, 64, 184]
        ################################################################
        #(batch_size, sequence_length, channels)->(batch_size, channels, sequence_length)
        x = x.permute(0,2,1)
        x = LBO_INVERSE @ x  
        x = x.permute(0, 2, 1)#(batch_size, channels, sequence_length)->(batch_size, sequence_length128, channels64)
        ################################################################
        # Approximator
        ################################################################
        # kernel_batch = self.kernel_generator(geo_code) 
        
        # weights = kernel_batch[0]
        
        x = torch.einsum("bix,iox->box", x[:, :], self.weights1)
        ################################################################
        # Decode
        ################################################################
        x =  x @ LBO_MATRIX.T

        return x
class MLPdd(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLPdd, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
       # print('n_input',n_input)
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x     
class MeshNO(nn.Module):
    def __init__(self, modes, width, geodim):
        super(MeshNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        
        self.input_proj = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, width)
        )
        
        
        self.fc0 = nn.Linear(3, self.width)  # input channel is 2: (a(x), x)

        self.conv1 = Approximation_block(self.width, self.width, self.modes1 , self.modes1)
        self.conv2 = Approximation_block(self.width, self.width, self.modes1 , self.modes1)
        self.conv3 = Approximation_block(self.width, self.width, self.modes1 , self.modes1)
        # self.conv4 = Approximation_block(self.width, self.width, self.modes1, self.modes1)   
      
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        # self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.out1 = nn.Linear(self.width, geodim)
        self.out2 = nn.Linear(geodim, 1)
        
        
        
        self.ln_1 = nn.LayerNorm(self.width)
        self.ln_2 = nn.LayerNorm(self.width)
        self.ln_3 = nn.LayerNorm(self.width)
        self.ln_4 = nn.LayerNorm(self.width)
        
        self.mlp1 = MLPdd(self.width, self.width * 2, self.width, n_layers=0, res=False, act='gelu')
        self.mlp2 = MLPdd(self.width, self.width * 2, self.width, n_layers=0, res=False, act='gelu')
        # self.mlp3 = MLPdd(self.width, self.width * 2, self.width, n_layers=0, res=False, act='gelu')

    def forward(self, meshPoints,sdf, LBO_MATRIX, LBO_INVERSE):
        # grid = self.get_grid(x.shape, x.device)
        # print('grid',grid.shape)
        
        
        # x_in = torch.cat([meshPoints, sdf], dim=-1)  # [N, input_dim]
        # x = self.input_proj(x_in) #[N, input_dim]
        # x = x.unsqueeze(0)#1,N,input_dim
        # print('x',x.shape)
        # x = torch.cat((x, grid), dim=-1)
        # x = x.permute(0,2,1)  #1,input_dim,N 
        
   
        
        # x = self.conv1(self.ln_1(x), LBO_MATRIX, LBO_INVERSE) + x
        # x = self.mlp1(self.ln_2(x)) + x
        # # x  = x1  + x2 
        # # x = F.gelu(x)
        # #print("Output Shape4:", x.shape) #[10, 64, 7199]
        
        # # x1 = self.conv2(x, self.LBO_MATRIX, self.LBO_INVERSE)
        # # x2 = self.w1(x)
        # # x  = x1  + x2
        # # x = F.gelu(x)
        # #print("Output Shape5:", x.shape) #[10, 64, 7199]
        
        # x = self.conv3(self.ln_3(x), LBO_MATRIX, LBO_INVERSE)
        # x = self.mlp2(self.ln_4(x)) + x
        # # x2 = self.w2(x)
        # # x  = x1  + x2
        # # ————————————————————————————————————————————————————————————————
        # ### output layer
        # # x = x.permute(0, 2, 1)
        # x = self.fc(x) #B,N,geodim
        # # x = x.permute(0, 2, 1)
        
        # # x = x.permute(0, 2, 1)#(batch_size, sequence_length, channels)->(batch_size, channels, sequence_length)
        # x = LBO_INVERSE @ x # B,m,geo
        
        # x = torch.squeeze(x)
        
        
        x_in = torch.cat([meshPoints, sdf], dim=-1)  # [N, input_dim]
        x = self.input_proj(x_in) #[N, input_dim]
        x = x.unsqueeze(0)#1,N,input_dim

        x = x.permute(0,2,1)  #1,input_dim,N
        x1 = self.conv1(x, LBO_MATRIX, LBO_INVERSE)
        x2 = self.w0(x)
        x  = x1  + x2 
        x = F.gelu(x)
        
        x1 = self.conv3(x, LBO_MATRIX, LBO_INVERSE)
        x2 = self.w2(x)
        x  = x1  + x2
        # ————————————————————————————————————————————————————————————————
        ### output layer
        x = x.permute(0, 2, 1)
        x = self.out1(x) #B,N,geodim
        x = self.out2(x) #B,N,geodim
        x = LBO_INVERSE @ x # B,m,geo
        x = torch.squeeze(x)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
from torch_geometric.nn import GCNConv

class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # 两层 GCN
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        """
        x: Tensor of shape (B, C, N)
        edge_index: Tensor of shape (2, E), shared across batch
        """
        B, N, C = x.shape

        # (B, N, C) -> (B*N, C)
        x = x.contiguous().view(B * N, C)

        # # 构造 batched edge_index
        edge_index_batch = []
        for i in range(B):
            offset = i * N
            edge_index_batch.append(edge_index + offset)
        edge_index_batch = torch.cat(edge_index_batch, dim=1)  # shape (2, B*E)

        # GCNConv
        x = self.gcn1(x, edge_index_batch)
        x = F.gelu(x)
        x = self.gcn2(x, edge_index_batch)

        # reshape 回 (B, out_dim, N)
        x = x.view(B, N, -1).contiguous()
        return x  # shape: (B, N, out_dim)

from torch_geometric.nn import GATConv    
class SimpleGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=2):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        """
        x: Tensor of shape (B, C, N)
        edge_index: Tensor of shape (2, E), shared across batch
        """
        B, N, C = x.shape
    
        # (B, C, N) -> (B*N, C)
        x = x.contiguous().view(B * N, C)

        # 构造 batched edge_index
        edge_index_batch = []
        for i in range(B):
            offset = i * N
            edge_index_batch.append(edge_index + offset)
        edge_index_batch = torch.cat(edge_index_batch, dim=1)  # shape (2, B*E)

        # GATConv
        x = self.gat1(x, edge_index_batch)
        x = F.gelu(x)
        x = self.gat2(x, edge_index_batch)

        # reshape 回 (B, out_dim, N)
        x = x.view(B, N, -1).contiguous()
        return x  # shape: (B, out_dim, N)
    
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, 0)
        #print(self.mean.shape)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps 
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps 
                mean = self.mean[:,sample_idx]

        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x
  
    
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
   

    def abs(self, x, y):
        
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        print('abs')
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        eps = 1e-8
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        if(y_norms==0):
            print("出现0")
        eps = 1e-8
        print('rel')
        return diff_norms/(y_norms + eps)

    def __call__(self, x, y):
    
        return self.rel(x, y)

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c