# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 17:26:37 2025

@author: DingYe
"""

import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import time
import pandas as pd
from utils import count_params,LpLoss
from model import NORM_net#,Geo_net
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def save_data(filename,data):
    with open(filename, 'w') as f:
        np.savetxt(f, data, delimiter=',')
        
def build_edge_index_from_elements(elements):
    edge_set = set()
    for tet in elements.tolist():  # 每个四面体
        i, j, k, l = tet
        edges = [
            (i, j), (j, i),
            (i, k), (k, i),
            (i, l), (l, i),
            (j, k), (k, j),
            (j, l), (l, j),
            (k, l), (l, k)
        ]
        edge_set.update(edges)
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    return edge_index
 

class BatchNormalizer:
    def __init__(self, batch_size=100, eps=1e-8):
        self.batch_size = batch_size
        self.eps = eps
        self.means = []
        self.stds = []

    def encode(self, data_list):
        normalized_list = []
        for i in range(0, len(data_list), self.batch_size):
            batch = data_list[i:i+self.batch_size]
            batch_tensor = torch.stack(batch)  # [B, ...]
            mean = batch_tensor.mean()
            std = batch_tensor.std()
            self.means.append(mean)
            self.stds.append(std)
            batch_norm = [(x - mean) / (std + self.eps) for x in batch]
            normalized_list.extend(batch_norm)
        return normalized_list

    def decode(self, tensor, idx):
        batch_idx = idx // self.batch_size
        mean = self.means[batch_idx]
        std = self.stds[batch_idx]
        return tensor * (std + self.eps) + mean
    
 


def getdata(modes):
    
    save_folder1 = "../data"
    
    LBO_MATRIX_data = torch.load(f"{save_folder1}/LBO_MATRIX_data.pt")
    LBO_INVERSE_data = torch.load(f"{save_folder1}/LBO_INVERSE_data.pt")
    allPoints = torch.load(f"{save_folder1}/allPoints.pt")
    allElements = torch.load(f"{save_folder1}/allElements.pt")
    inputdata = torch.load(f"{save_folder1}/inputdata.pt")
    outputdata = torch.load(f"{save_folder1}/outputdata.pt")
    outputdata = [t.to(torch.float32) for t in outputdata]
 
    allSDFs = torch.load(f"{save_folder1}/allDistance.pt")
 
    return allPoints,allElements,allSDFs,inputdata,outputdata,LBO_MATRIX_data, LBO_INVERSE_data 

def lr_schedule(epoch):
    if epoch < 45*3:
        return 0.5 ** (epoch // 45)
    else:
        return 0.5 ** (3 + (epoch - 45*3) // 20)
    
    
def main(args):  
 


   
    device_id = 0  
    torch.cuda.set_device(device_id)  
    print("\n=============================")
    print("当前使用的显卡:", torch.cuda.current_device()) 
    print("设备名称:", torch.cuda.get_device_name(device_id))
    print("=============================\n")
    
    
    ntrain = args.num_train
    ntest = args.num_test
    batch_size = args.batch_size
    learning_rate = args.lr  
    epochs = args.epochs 
    modes = args.modes
    width = args.width   
    ntrain = args.num_train
    ntest  = args.num_test
    num_rep = args.num
    

    allPoints,allElements,allSDFs,inputdata,outputdata,LBO_MATRIX_data, LBO_INVERSE_data = getdata(modes)
    

    
    print('allPoints length:',len(allPoints))
    print('inputdata length:',len(inputdata))
    print('outputdata length:',len(outputdata))  
    
    
    input_normalizer = BatchNormalizer(batch_size=1)#*****************************************************************
    output_normalizer = BatchNormalizer(batch_size=1)
    inputdata = input_normalizer.encode(inputdata)
    outputdata = output_normalizer.encode(outputdata)


    input_batches = []
    output_batches = []
    
    for i in range(0, ntrain+ntest, batch_size):
        batch_in = inputdata[i:i+batch_size]
        batch_out = outputdata[i:i+batch_size]
    
        x = torch.cat(batch_in, dim=0)   
        y = torch.cat(batch_out, dim=0)
    
        input_batches.append(x)
        output_batches.append(y)
    
    
    
    adj_norm_list = []
    for i in range(len( allElements)):   
        elements = allElements[i]
        edge_index = build_edge_index_from_elements(elements)
        adj_norm_list.append(edge_index)
        
 
    ################################################################
    # training and evaluation
    ################################################################

    model = NORM_net(modes, width).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4,foreach=False)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    
    myloss = LpLoss(size_average=False)
    
    time_step = time.perf_counter()
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    
    
    
    
    for ep in range(epochs):
        model.train()
        train_l2 = 0
        for ib, i in enumerate(range(0, ntrain, batch_size)):
       
            x = input_batches[ib]
            y = output_batches[ib]

            ilbo = i // 100 

            x, y = x.cuda(), y.cuda()
            MATRIX_Phy, INVERSE_Phy = LBO_MATRIX_data[ilbo], LBO_INVERSE_data[ilbo]
            MATRIX_Phy, INVERSE_Phy = MATRIX_Phy.cuda(), INVERSE_Phy.cuda()
            meshPoints = allPoints[ilbo]
            meshPoints = meshPoints.cuda()
            adj_norm = adj_norm_list[ilbo]
            adj_norm = adj_norm.cuda()  
            sdfs = allSDFs[ilbo]
            sdfs = sdfs.cuda() 
            
                       
            optimizer.zero_grad()
                  
            out,loss2 = model(x, MATRIX_Phy, INVERSE_Phy, meshPoints,adj_norm,ilbo,sdfs)
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
     

            l2.backward()
            optimizer.step()
            
            
            out = output_normalizer.decode(out, idx=i)
            
            out_real = out.view(batch_size, -1).cpu()
            y = output_normalizer.decode(y, idx=i)
            
            y_real   = y.view(batch_size, -1).cpu()
            train_l2 += myloss(out_real, y_real).item()   
 
            del x, y, out,  out_real, y_real
            del MATRIX_Phy, INVERSE_Phy, meshPoints, adj_norm
       
 
        scheduler.step()
        model.eval()
        test_l2 = 0.0

  
        ntrain_batches = ntrain // batch_size
        ntest_batches  = ntest // batch_size
        with torch.no_grad():
      
            for ib, i in enumerate(range(ntrain, ntrain + ntest, batch_size)):
                
                ilbo = i // 100  
           
    
                x = input_batches[ntrain_batches + ib]
                y = output_batches[ntrain_batches + ib]


                x, y = x.cuda(), y.cuda()
                
                MATRIX_Phy, INVERSE_Phy = LBO_MATRIX_data[ilbo], LBO_INVERSE_data[ilbo]
                MATRIX_Phy, INVERSE_Phy = MATRIX_Phy.cuda(), INVERSE_Phy.cuda()
                meshPoints = allPoints[ilbo]
                meshPoints = meshPoints.cuda()
                adj_norm = adj_norm_list[ilbo]
                adj_norm = adj_norm.cuda()
                
                sdfs = allSDFs[ilbo]
                sdfs = sdfs.cuda() 
            
                out,loss2 = model(x, MATRIX_Phy, INVERSE_Phy, meshPoints,adj_norm,ilbo,sdfs)

          
                out = output_normalizer.decode(out, idx=i)
                out_real = out.view(batch_size, -1).cpu()
             
                y = output_normalizer.decode(y, idx=i)
                y_real   = y.view(batch_size, -1).cpu()

                test_l2 += myloss(out_real, y_real).item()     
                
                del x, y, out, out_real, y_real
                del MATRIX_Phy, INVERSE_Phy, meshPoints, adj_norm
      
        train_l2 /= (ntrain-0)
        test_l2  /= ntest
        train_error[ep] = train_l2
        test_error [ep] = test_l2
        
        time_step_end = time.perf_counter()
        T = time_step_end - time_step
        
        if ep % 1 == 0:
            print('Step: %d, Train L2: %.5f, Test L2: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, T))
        time_step = time.perf_counter()

        folder = './net_model/'
        torch.save(model.state_dict(), os.path.join(folder,f"net_model_{num_rep}.pkl"))
  
        save_data(f"./train_loss_{num_rep}.csv",train_error)
        save_data(f"./test_loss_{num_rep}.csv",test_error)

                
     

if __name__ == "__main__":
    
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
              
    for i in range(5):
        
        print('====================================')
        print('NO.'+str(i)+'repetition......')
        print('====================================')

        for args in [
                        { 'modes'   : 96,  
                          'width'   : 64,
                          'batch_size': 20, 
                          'epochs'    : 500,
                          'num_train' : 800*100,
                          'num_test'  : 80*100,
                          'lr'        : 0.001,
                          'step_size' : 45,
                          'num': i},
                    ]:
            
            args = objectview(args)
                
        main(args)

    