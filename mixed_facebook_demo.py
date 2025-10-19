import torch.profiler
from torch.profiler import profile, ProfilerActivity
import time
from scipy.optimize import linear_sum_assignment
from numpy import *
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
from line_profiler import LineProfiler
import pickle
import networkx
import torch
from torch.cuda.amp import autocast


def GM_DSPFP_float(A1, A2, K=None, a=1, lamb=1, eps=0.1, 
                   percent=None, percent_in=None, 
                   maxit_in=150, theta=2, num_iter = 1) :
    # device = torch.cuda.device("cpu")
    a = torch.tensor(a, device=device, 
                     dtype=torch.float)
    # A1
    A2 = torch.tensor(A2).to(device)
    A1 = torch.tensor(A1).to(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    if K is not None:
        K = torch.tensor(K, device=device, dtype=torch.float)
        a_in = (torch.floor(torch.sqrt(torch.max(torch.max(K)))) * 10).to(device)
        k = a_in**2
        K_in = (K / k).to(torch.float32)  
        A1_in = (A1 / a_in).to(torch.float32)  
        A2_in = (A2 / a_in).to(torch.float32)  
    else:
        K_in = torch.tensor(0, device=device, 
                            dtype=torch.float)
        a1_in = torch.max(A1)
        a2_in = torch.max(A2)
        A1_in = (A1 / a1_in).to(torch.float32)  
        A2_in = (A2 / a2_in).to(torch.float32)  
    
    # start_time = time.perf_counter()
    n1 = A1_in.size(0)  
    n2 = A2_in.size(0)  

    n1 = torch.tensor([n1], device=A1_in.device, 
                             dtype=torch.int32)
    n2 = torch.tensor([n2], device=A2_in.device, 
                             dtype=torch.int32)
    nn = n1 * n2
    
    n = torch.max(n1, n2)  
    tol = n * 1.1
    theta = torch.tensor(theta, device=device, 
                         dtype=torch.float)

    for _ in range(5):
        nothing = torch.linalg.norm(A1_in, ord="fro")
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    
    for m in range(num_iter):
        start_event.record()
        # print(time.time())

        X = torch.ones((n1, n2), device=device, 
                       dtype=torch.float) / (nn)
        ep, ind = 100, 1
    
        Y = torch.zeros((n, n), device=device, 
                        dtype=torch.float)
        x = torch.ones((n1, n2), device=device, 
                       dtype=torch.float)
    
        ind_per = []
        while ep >= eps and ind <= 30:
           
            result = A1_in @ (X @ A2_in) + lamb * K_in 

            max_value = torch.max(result)

            Y[:n1, :n2] = torch.div(result, max_value)
            
            Y = (theta / 2) * Y
            
            Y, j = project_DS_per_float(Y, n1, n2, 
                                        percent_in, 
                                        eps, maxit_in, n,tol)            
            Y = Y[:n1, :n2]
    
            ind_per.append(j)

            X = (1-a)*X + (a*Y)
            
            ep = torch.linalg.norm(X-x, ord="fro") / torch.linalg.norm(X, ord="fro")

            x = X
            ind = ind + 1

    
    end_event.record()
    end_event.synchronize()  
    # torch.cuda.synchronize()  
    elapsed_time_ms = start_event.elapsed_time(end_event)

    P = hugarian(X)
    print(ind_per)
    print(f"Elapsed time: {elapsed_time_ms} ms")
    return P, elapsed_time_ms
    
# @profile
def project_DS_per_float(X=None,
                        n1=None,
                        n2=None,
                        percent_in=None,
                        eps=None,
                        maxit=None,
                        n=None,tol=None):
    
    nn1, nn2 = X.shape[0], X.shape[1]
    # n0 = n.to('cpu')
    i = torch.zeros(1, dtype=torch.int32, device=X.device)
    X_sum = torch.zeros(1, dtype=torch.bool, device=X.device)
    # one1 = torch.ones((1, nn1), dtype=torch.float32, device=X.device)
    one2 = torch.ones((nn2, 1), dtype=torch.float32, device=X.device).T
    X, i = while_loop_on_gpu(X, X_sum, n, one2, i,tol)
    return X, i


@torch.jit.script
def while_loop_on_gpu(X, X_sum, n, one2, i,tol):
    while (not X_sum):
        k1 = torch.sum(X.T, dim=0, keepdim=True) / n  
        k2 = torch.sum(X, dim=0, keepdim=True) / n  
        X = X + 1 / n + torch.sum(k1) / n
        X = X - k2 
        X = X - k1.T @ one2 
        X = torch.relu(X)
        X_sum = (torch.sum(X) < tol)
        i = i + 1
    return X, i


def avoid(X,n,one2):
    k1 = torch.sum(X.T, dim=0, keepdim=True) / n  
    k2 = torch.sum(X, dim=0, keepdim=True) / n   
    X = X + 1 / n + torch.sum(k1) / n
    X = X - k2 
    X = X - k1.T @ one2.T 
    X = torch.clamp(X, min=0)
    return X

def hugarian(matrix):
    matrix = matrix.cpu().numpy()
    n, m = matrix.shape
    P = np.mat(torch.zeros((n, m)))
    row_ind, col_ind = linear_sum_assignment(-matrix)
    P[row_ind, col_ind] = 1
    return P

# @profile
def Evaluation(P,A1,A2,F1,F2):
    torch.backends.cuda.matmul.allow_tf32 = False
    dissimilarEdges = A1 - P @ A2 @ P.T
    dissimilarNodes = F1 - P @ F2
    ERROR = torch.norm(dissimilarEdges, 'fro') / 2 + 1 * torch.norm(dissimilarNodes, 'fro')
    print(ERROR)
    return ERROR   

print(torch.cuda.is_available())
device = torch.device("cuda")  
print(while_loop_on_gpu.code)


profiler = LineProfiler()


profiler.add_function(GM_DSPFP_float)  
profiler.add_function(project_DS_per_float)  
profiler.add_function(while_loop_on_gpu)  


import networkx as nx



with open("G_face.gpickle", "rb") as f:
    source_graph = pickle.load(f)

with open("G_face_noise_5.gpickle", "rb") as f:
    noisy5_graph = pickle.load(f)

with open("G_face_noise_15.gpickle", "rb") as f:
    noisy15_graph = pickle.load(f)

with open("G_face_noise_25.gpickle", "rb") as f:
    noisy25_graph = pickle.load(f)

print("Graphs loaded successfully!")
adj_sorce = networkx.to_numpy_array(source_graph)
adj_noisy5 = networkx.to_numpy_array(noisy5_graph)
adj_noisy15 = networkx.to_numpy_array(noisy15_graph)
adj_noisy25 = networkx.to_numpy_array(noisy25_graph)


for step in range(1):
    P, rtime  = GM_DSPFP_float(adj_noisy25, adj_sorce, a=0.95, lamb=0, eps=0.1, theta = 10)


print(torch.backends.cuda.matmul.allow_tf32)
print("Run time: "+str(rtime))
print("Node acc: "+str(sum(np.diag(P))/4039)) 
print("Edge acc: "+str(np.sum(np.multiply(adj_noisy25,P*adj_sorce*P.T))/adj_sorce.sum()))


profiler.runcall(GM_DSPFP_float, adj_noisy25, adj_sorce, a=0.95, lamb=0, eps=0.1, theta=10)

profiler.print_stats()