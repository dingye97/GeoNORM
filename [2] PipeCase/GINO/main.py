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
from utils import LpLoss,SamplewiseNormalizer
from GNOFNOGNO import GNOFNOGNO
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.stderr = open(os.devnull, 'w')


def loadMatData(path):
    all_data_list = []
    
    for i in range(1, 21):
        file_path = path + f"/pipe_flow{i}.pt"
        data = torch.load(file_path)  
        all_data_list.extend(data)
    
    print(f"总零件数: {len(all_data_list)}")
    input_list = []
    output_list = []
    
    for sample in all_data_list:
       
        D, T, twoN = sample.shape
        for d in range(0, 40, 2):  
            for t in range(3):  
           
                xy_input = sample[d, t:t+3, :] 
         
                vx = xy_input[:, 0::2]  
                vy = xy_input[:, 1::2]  
            
                speed_input = torch.sqrt(vx**2 + vy**2)  
             
                x = speed_input.T.unsqueeze(0)
                xy_target = sample[d, t+3, :] 
                vx_t = xy_target[0::2]  
                vy_t = xy_target[1::2] 
                speed_target = torch.sqrt(vx_t**2 + vy_t**2) 
                y = speed_target.unsqueeze(0).unsqueeze(-1)  

                input_list.append(x)
                output_list.append(y)

    return input_list, output_list


def loadMatLBOData(path):
    data = sio.loadmat(path)
    #data

    allPoints = []
    allElements = []
 
    
    LBO_Output = sio.loadmat(path)
    LBO_Output = data['Results'][0]
    for ida in LBO_Output:
        ida = ida[0]
        ida = ida[0]
        points1 = ida[8]
        face1 = ida[9]

        points = torch.tensor(points1, dtype=torch.float32)
        face1 = torch.tensor(face1, dtype=torch.float32)
        allPoints.append(points)
        allElements.append(face1)
            
    return allPoints,allElements

def save_data(filename,data):
    with open(filename, 'w') as f:
        np.savetxt(f, data, delimiter=',')
        
def main(args):  
 

    device_id = 0 
    torch.cuda.set_device(device_id)
    print("\n=============================")
    print("cuda:", torch.cuda.current_device())  
    print("cuda:", torch.cuda.get_device_name(device_id))
    print("=============================\n")
    
   
    

    ntrain = args.num_train
    ntest = args.num_test
    batch_size = args.batch_size
    learning_rate = args.lr
    num_rep = args.num
    epochs = args.epochs
    step_size = args.step_size
    gamma = 0.2
    

    ################################################################
    # reading data and normalization
    ################################################################   

    inputdata, outputdata = loadMatData("../data_pipe")

    
    normalizer5 = SamplewiseNormalizer()
    

    inputdata, outputdata = normalizer5.normalize_dataset(inputdata, outputdata) 
    
    print('len(inputdata):',len(inputdata))
    print('len(outputdata):',len(outputdata))

     
    ntrain = args.num_train
    ntest  = args.num_test

         
    ################################################################
    # reading LBO basis
    ################################################################
    PATHS = [
        '../data_pipe//LBO_pipe_1.mat',
        '../data_pipe/LBO_pipe_2.mat'
    ]
    

    allPoints = []
    for path in PATHS:
        iallPoints, iallElements = loadMatLBOData(path)
        allPoints.extend(iallPoints)

    normalized_allPoints = []
    for idx, points in enumerate(allPoints):
        points  = points[:, :2]
        x_col_max = points.max(dim=0, keepdim=True).values  
        x_col_min = points.min(dim=0, keepdim=True).values  
        normPoints = (points-x_col_min) / (x_col_max-x_col_min)
        normalized_allPoints.append(normPoints)

    
    
    lin = np.linspace(0, 1, 64) 
    X, Y = np.meshgrid(lin, lin, indexing='ij') 
    latent_grid = np.stack([X, Y], axis=-1)
    latent_grid = torch.from_numpy(latent_grid).float()  
    latent_grid = latent_grid.cuda() 


   
        
    ################################################################
    # training and evaluation
    ################################################################

          
    model = GNOFNOGNO(
            radius_in=0.015,
            radius_out=0.015,
            embed_dim=48,
            hidden_channels=(48, 48),
            in_channels=3,
            out_channels=1,
            fno_out_channels=48,
            modes = 32,
            width = 72      ).cuda()         
        



    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0) #1e-4
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    myloss = LpLoss(size_average=False)
    
    time_start = time.perf_counter()
    time_step = time.perf_counter()
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    test_MAE = np.zeros((epochs))

   
       
    for ep in range(epochs):
 
        model.train()
        train_mse = 0
        train_l2 = 0
        for i in range(0, ntrain, batch_size):
     
            x = torch.cat(inputdata[i:i+batch_size], dim=0)   
            y = torch.cat(outputdata[i:i+batch_size], dim=0) 
            x, y = x.cuda(), y.cuda()
        
     
            ilbo = i // 60
            meshPoints = normalized_allPoints[ilbo]
            meshPoints = meshPoints.cuda()

            
            optimizer.zero_grad()
        
            out = model(meshPoints, latent_grid, x)#
            
          

            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()
            
            out = normalizer5.denormalize_single_output(out, i)
            out_real = out.view(batch_size, -1).cpu()
            y = normalizer5.denormalize_single_output(y, i)
            y_real   = y.view(batch_size, -1).cpu()
            train_l2 += myloss(out_real, y_real).item()   
            
          
            optimizer.step()
 
        scheduler.step()
        model.eval()
        test_l2 = 0.0
        test_mae = 0.0
        with torch.no_grad():
            for i in range(250*60, 250*60+ntest, batch_size):
                x = torch.cat(inputdata[i:i+batch_size], dim=0)  
                y = torch.cat(outputdata[i:i+batch_size], dim=0) 
                x, y = x.cuda(), y.cuda()
                
                ilbo = i // 60          
                meshPoints = normalized_allPoints[ilbo]
                meshPoints = meshPoints.cuda()
        
                
                out = model(meshPoints, latent_grid, x)
    
                out = normalizer5.denormalize_single_output(out, i)
                out_real = out.view(batch_size, -1).cpu()
                y = normalizer5.denormalize_single_output(y, i)
                y_real   = y.view(batch_size, -1).cpu()

                test_l2 += myloss(out_real, y_real).item()     
                mae_sum = torch.mean(torch.abs(out_real - y_real), dim=1).sum().item() 
                test_mae += mae_sum
          
        train_l2 /= ntrain
        test_l2  /= ntest
        test_mae /= ntest
        train_error[ep] = train_l2
        test_error [ep] = test_l2
        test_MAE[ep] = test_mae
        
        time_step_end = time.perf_counter()
        T = time_step_end - time_step
        
        if ep % 1 == 0:
            print('Step: %d, Train L2: %.5f, Test L2: %.5f,  test_mae: %.10f, Time: %.3fs'%(ep, train_l2, test_l2,test_mae, T))
        time_step = time.perf_counter()
    

        folder = './net_model/'
        torch.save(model.state_dict(), os.path.join(folder,f"net_model_{num_rep}.pkl"))

        save_data(f"./train_loss_{num_rep}.csv",train_error)
        save_data(f"./test_loss_{num_rep}.csv",test_error)
        save_data(f"./test_MAE_{num_rep}.csv",test_MAE)

                
     

if __name__ == "__main__":
    
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
              
    for i in range(5):
        
        print('====================================')
        print('NO.'+str(i)+'repetition......')
        print('====================================')

        for args in [
                        { 
                          'batch_size': 15, 
                          'epochs'    : 210,
                          'num_train' : 250*60, 
                          'num_test'  : 50*60,
                          'lr'        : 0.0001,
                          'step_size' : 50,
                          'num': i},
                    ]:
            
            args = objectview(args)
                
        main(args)

    