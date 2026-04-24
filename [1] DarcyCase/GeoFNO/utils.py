# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 17:27:41 2025

@author: DingYe
"""
import torch
import operator
from functools import reduce

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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