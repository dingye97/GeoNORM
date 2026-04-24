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
    

def LBOProcess(MATRIX_Output):
    mat = MATRIX_Output 
    
    for j in range(mat.size(1)):
        col = mat[:, j]
        nonzero_indices = (col != 0).nonzero(as_tuple=True)[0]
        if nonzero_indices.numel() > 0:
            first_idx = nonzero_indices[0].item()
            first_val = col[first_idx]
        if first_val < 0:
            col = -col  # 反向这列
        norm = torch.norm(col, p=2)
        if norm > 0:
            col = col / norm
        mat[:, j] = col
    MATRIX_Output = mat
    return MATRIX_Output

class SamplewiseNormalizer:
    def __init__(self):
        self.input_norm_info = []   
        self.output_norm_info = []  

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
 
        v_mean, v_std = self.output_norm_info[idx]
        return v_norm * v_std + v_mean
  
    
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

