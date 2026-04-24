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
from utils import LpLoss,LBOProcess,SamplewiseNormalizer
from model import NORM_net
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def loadMatData(path):
    data = sio.loadmat(path)
    #input_data
    inputdata = []
    indata  = data['input'][0]
    #output_data
    outputdata = []
    outdata  = data['output'][0]   
    
    for ida in indata:
        ida = torch.tensor(ida[0], dtype=torch.float32)
        ida = ida.unsqueeze(0).unsqueeze(-1)
        inputdata.append(ida)
    for oda in outdata:
        oda = torch.tensor(oda[0], dtype=torch.float32)
        oda = oda.unsqueeze(0).unsqueeze(-1)
        outputdata.append(oda)     

    return inputdata,outputdata

    
def loadMatLBOData(path):
    data = sio.loadmat(path)
  
    LBO_MATRIX_data = []
    LBO_INVERSE_data = []
    allPoints = []
    allElements = []
 
    
    LBO_Output = sio.loadmat(path)
    LBO_Output = data['Results'][0]
    for ida in LBO_Output:
        ida = ida[0]
        ida = ida[0]
        Eigenvalues = ida[6]
        LBOMATRIX = ida[7]
        points1 = ida[8]
        face1 = ida[9]

        Eigenvalues = torch.tensor(Eigenvalues.squeeze(), dtype=torch.float32)
        points = torch.tensor(points1, dtype=torch.float32)
        face1 = torch.tensor(face1, dtype=torch.float32)
        MATRIX_Output = torch.tensor(LBOMATRIX, dtype=torch.float32)
        MATRIX_Output = LBOProcess(MATRIX_Output)
        INVERSE_Output = (MATRIX_Output.T @ MATRIX_Output).inverse() @ MATRIX_Output.T

        LBO_MATRIX_data.append(MATRIX_Output)
        LBO_INVERSE_data.append(INVERSE_Output)
        allPoints.append(points)
        allElements.append(face1)

              
    return LBO_MATRIX_data,LBO_INVERSE_data,allPoints,allElements

def save_data(filename,data):
    with open(filename, 'w') as f:
        np.savetxt(f, data, delimiter=',')
        
def build_edge_index_from_elements(elements):
    edge_set = set()
    for tri in elements.tolist():
        i, j, k = tri
        for a, b in [(i, j), (j, i), (i, k), (k, i), (j, k), (k, j)]:
            edge_set.add((a, b))
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()  # shape [2, E]
    return edge_index

        
def main(args):  
 

    
    device_id = 0  
    torch.cuda.set_device(device_id)
    print("\n=============================")
    print("CUDA:", torch.cuda.current_device()) 
    print("CUDA:", torch.cuda.get_device_name(device_id))
    print("=============================\n")
    
  
    ntrain = args.num_train
    ntest = args.num_test
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    modes = args.modes
    width = args.width
    step_size = args.step_size
    gamma = args.gamma
    num_rep = args.num
    
    
    ################################################################
    # reading data and normalization
    ################################################################   
    PATH1  = '../data_darcy/Darcy_flow1.mat'
    inputdata1,outputdata1 = loadMatData(PATH1)
    PATH2  = '../data_darcy/Darcy_flow2.mat'
    inputdata2,outputdata2 = loadMatData(PATH2)
    PATH3  = '../data_darcy/Darcy_flow3.mat'
    inputdata3,outputdata3 = loadMatData(PATH3)
    inputdata = inputdata1 + inputdata2 + inputdata3 
    outputdata = outputdata1 + outputdata2 + outputdata3 
 
    normalizer5 = SamplewiseNormalizer() 
    inputdata, outputdata = normalizer5.normalize_dataset(inputdata, outputdata)
     
    ################################################################
    # reading LBO basis
    ################################################################
    PATH21 = '../data_darcy/LBO_pipe1.mat'  
    LBO_MATRIX_data1,LBO_INVERSE_data1,allPoints1,allElements1 = loadMatLBOData(PATH21)
    PATH22 = '../data_darcy/LBO_pipe2.mat'  
    LBO_MATRIX_data2,LBO_INVERSE_data2,allPoints2,allElements2 = loadMatLBOData(PATH22)
    LBO_MATRIX_data  = LBO_MATRIX_data1 + LBO_MATRIX_data2 
    LBO_INVERSE_data = LBO_INVERSE_data1 + LBO_INVERSE_data2
    allPoints        = allPoints1 + allPoints2  
    allElements      = allElements1 + allElements2
    
    adj_norm_list = []
    for points, elements in zip(allPoints, allElements):
        edge_index = build_edge_index_from_elements(elements)
        adj_norm_list.append(edge_index)
        

    ################################################################
    # training and evaluation
    ################################################################
    model = NORM_net(modes, width).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    
    myloss = LpLoss(size_average=False)
    
   
    time_step = time.perf_counter()
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    
   
    best_loss = 0.06

  
    for ep in range(epochs):
        model.train()
        train_l2 = 0
        for i in range(0, ntrain, batch_size):
       
            x = torch.cat(inputdata[i:i+batch_size], dim=0)  
            y = torch.cat(outputdata[i:i+batch_size], dim=0)  

            ilbo = i // 100
            
            x, y = x.cuda(), y.cuda()
            MATRIX_Phy, INVERSE_Phy = LBO_MATRIX_data[ilbo], LBO_INVERSE_data[ilbo]
            MATRIX_Phy, INVERSE_Phy = MATRIX_Phy.cuda(), INVERSE_Phy.cuda()
            
            meshPoints = allPoints[ilbo]
            meshPoints = meshPoints.cuda()
            adj_norm = adj_norm_list[ilbo]
            adj_norm = adj_norm.cuda() 
            
            optimizer.zero_grad()
            out,loss2 = model(x, MATRIX_Phy, INVERSE_Phy, meshPoints,adj_norm)
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
  
        with torch.no_grad():
            for i in range(ntrain, ntrain+ntest, batch_size):
                ilbo = i // 100
                x = torch.cat(inputdata[i:i+batch_size], dim=0)  
                y = torch.cat(outputdata[i:i+batch_size], dim=0)  
                x, y = x.cuda(), y.cuda()
                
                MATRIX_Phy, INVERSE_Phy = LBO_MATRIX_data[ilbo], LBO_INVERSE_data[ilbo]
                MATRIX_Phy, INVERSE_Phy = MATRIX_Phy.cuda(), INVERSE_Phy.cuda()
                meshPoints = allPoints[ilbo]
                meshPoints = meshPoints.cuda()
                adj_norm = adj_norm_list[ilbo]
                adj_norm = adj_norm.cuda()      
                out,loss2 = model(x, MATRIX_Phy, INVERSE_Phy, meshPoints,adj_norm)
   
                out = normalizer5.denormalize_single_output(out, i)
                out_real = out.view(batch_size, -1).cpu()
                y = normalizer5.denormalize_single_output(y, i)
                y_real   = y.view(batch_size, -1).cpu()
                test_l2 += myloss(out_real, y_real).item()                
          
        train_l2 /= ntrain
        test_l2  /= ntest
        train_error[ep] = train_l2
        test_error [ep] = test_l2
        time_step_end = time.perf_counter()
        T = time_step_end - time_step
        
        if ep % 1 == 0:
            print('Step: %d, Train L2: %.5f, Test L2: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, T))
        time_step = time.perf_counter()
    
        #save net param
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
                        { 'modes'     : 128,  
                          'width'     : 48,
                          'batch_size': 25, 
                          'epochs'    : 500,      
                          'num_train' : 40000, 
                          'num_test'  : 10000,
                          'lr'        : 0.001,
                          'step_size' : 40,
                          'gamma'     : 0.1,
                          'num': i},
                    ]:
            
            args = objectview(args)
                
        main(args)

    