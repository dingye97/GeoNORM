
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import time
import pandas as pd
from utils import LpLoss,SamplewiseNormalizer
from Transolver import Transolver
import argparse
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
        
def main(num):  
    
    num_rep = num
 
    device_id = 0 
    torch.cuda.set_device(device_id)
    print("\n=============================")
    print("cuda:", torch.cuda.current_device())  
    print("cuda:", torch.cuda.get_device_name(device_id))
    print("=============================\n")
    
   
    
    ################################################################
    # reading data and normalization
    # ################################################################   
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
    PATH21 = '../data_darcy/LBO_darcy1.mat'  
    allPoints1,allElements1 = loadMatLBOData(PATH21)
    PATH22 = '../data_darcy/LBO_darcy2.mat'  
    allPoints2,allElements2 = loadMatLBOData(PATH22)
    
    allPoints    = allPoints1 + allPoints2 

  
    ################################################################
    # training and evaluation
    ################################################################
    parser = argparse.ArgumentParser('Training Transolver')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
    parser.add_argument('--n-layers', type=int, default=12, help='layers')
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--mlp_ratio', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--unified_pos', type=int, default=0)
    parser.add_argument('--ref', type=int, default=8)
    parser.add_argument('--slice_num', type=int, default=64)
    
    parser.add_argument('--num_train', type=int, default=40000)
    parser.add_argument('--num_test', type=int, default=10000)
    
    args = parser.parse_args()   
    
    ntrain = args.num_train
    ntest = args.num_test
    batch_size = args.batch_size
    epochs = args.epochs
       

    model = Transolver(space_dim=3,
                      n_layers=args.n_layers,
                      n_hidden=args.n_hidden,
                      dropout=args.dropout,
                      n_head=args.n_heads,
                      mlp_ratio=args.mlp_ratio,
                      fun_dim=1,
                      out_dim=1,
                      slice_num=args.slice_num,
                      ref=args.ref,
                      unified_pos=args.unified_pos).cuda()

    steps_per_epoch = ntrain // batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=steps_per_epoch)

    
    myloss = LpLoss(size_average=False)
    
    time_step = time.perf_counter()
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    test_MAE = np.zeros((epochs))
    

    best_loss = 0.1

    for ep in range(epochs):
        model.train()
        train_l2 = 0
        for i in range(0, ntrain, batch_size):
            x = torch.cat(inputdata[i:i+batch_size], dim=0)  
            y = torch.cat(outputdata[i:i+batch_size], dim=0) 

            x, y = x.cuda(), y.cuda()


            ilbo = i // 100
            meshPoints = allPoints[ilbo]
            meshPoints = meshPoints.cuda()
 
            
            optimizer.zero_grad()
                  
            out = model(x,  meshPoints)


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
            for i in range(ntrain, ntrain+ntest, batch_size):
                x = torch.cat(inputdata[i:i+batch_size], dim=0)  
                y = torch.cat(outputdata[i:i+batch_size], dim=0) 
                x, y = x.cuda(), y.cuda()
                
                ilbo = i // 100
                meshPoints = allPoints[ilbo]
                meshPoints = meshPoints.cuda()

                out = model(x,  meshPoints)
      
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


                
        main(i)

    