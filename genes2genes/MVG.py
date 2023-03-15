import torch
import seaborn as sb
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import gpytorch
import matplotlib.pyplot as plt
import torch.distributions as td
import scipy
import warnings
warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)

def generate_random_MVG_dataset(d,N,DIST_SEED=1,use_zero_mean=False,MEAN_SEED=1,):
    #d = n_dimensions
    #N = n_data_points
    input_dims = []
    for i in range(d):
        input_dims.append(i)
    X = torch.tensor(input_dims) # input points on x axis (dims) as in GP
    kernel = gpytorch.kernels.RBFKernel()
    C = kernel(X).evaluate()
    μ = torch.zeros(d) # zero mean case for dimensions
    if(not use_zero_mean):
        # difference mean for all cases
        torch.manual_seed(MEAN_SEED)
        for i in range(d):
            μ[i] = torch.distributions.Uniform(5,10).rsample() 
    D = torch.empty(N,d) # Data matrix
    torch.manual_seed(DIST_SEED)
    for i in range(N):
        D[i] = torch.distributions.MultivariateNormal(μ, C).rsample().detach() 
    return μ,C,D

# As p (n free dimensions) increases, the lower and upper bounds converge [Ref: Wallace book]
#def conway_constant_upper_bound(p):
#    return ((scipy.special.gamma( (p/2)+1  )**(2/p))*scipy.special.gamma( (2/p)+1))/(np.pi*p)
# Test case: p = 100 ----- 2**log2_conway_constant_upper_bound(p) #0.0613252739213439
def log_factorial(x):
    #return scipy.special.gammaln((x+1))/np.log(2)
    return scipy.special.gammaln((x+1))
def log_conway_constant_upper_bound(p):
    #return ((2/p)*log2_factorial(p/2)) +  log2_factorial(2/p) -np.log2(np.pi) -np.log2(p)
    return ((2/p)*log_factorial(p/2)) +  log_factorial(2/p) -np.log(np.pi) -np.log(p)

def negative_log_likelihood(μ,C,N,data, d, det_C):
    #print('det_C -- ', det_C)
    term1 = ((N*d)/2.0)*np.log(2*np.pi) 
    #term2 = (N/2.0)*np.log(det_C)
    term2 = 0.0 # bcz det_C =1 due to C=I
    term3 = 0.0
    #inverse_C = torch.linalg.inv(C)
    inverse_C = C # inverse of the I is itself (since we use Identity matrix)
    
    for i in range(N):
        temp = np.matrix(data[i] - μ)
        x_i = torch.tensor(temp)
        x_it =torch.tensor(temp.transpose())
        #term3 = term3 + torch.matmul(torch.matmul(x_i , inverse_C ), x_it).flatten()[0]
        term3 = term3 + torch.matmul(x_i, x_it).flatten()[0] # because C=I
    term3 = term3 * 0.5
    #print('NEG LOG:2 ', term1,term2, term3)
    return (term1 + term2 + term3).detach().item() 

def I_first_part(p,d,N,det_C):
    return (0.5*p*log_conway_constant_upper_bound(p))  + ((p/2)*np.log(N)) - (d/2) - (0.5*np.log(det_C))

def compute_mml_estimates(data,d,N):
    μ_mml = torch.mean(data,axis=0)
    term = 0.0
    for i in range(N):
        temp = data[i] - μ_mml   
        temp = np.matrix(temp) 
        temp_C = np.matmul(temp.transpose(),temp)
        term = term + temp_C
    C_mml = torch.tensor(term/(N-1)) 
    #print(np.linalg.det(temp_C))
    
    if(torch.linalg.det(C_mml)<=0): # (adding a small perturbation) regularisation to avoid numerical instability --- then it will have only positive eigenvalues and it will have the exact same eigenvectors
        C_mml = C_mml + (0.001*torch.eye(len(C_mml)))
    
    return  μ_mml,C_mml

def compute_MML_msg_len(data):
    
    d = data.shape[1]; N = len(data)
    μ,C = compute_mml_estimates(data,d,N)
    d = len(C)
    p = d*(d+3)/2
    det_C = torch.linalg.det(C).detach().numpy()  # determinant of the covariance matrix 

    I_model = I_first_part(p,d,N,det_C)
    I_data_g_model = negative_log_likelihood(μ,C,N,data,d, det_C) + p/2 
    #print('NEG LOG: ', I_data_g_model)
    return I_model, I_data_g_model, C

def run_dist_compute_v3(data_to_model, μ_base, C_base):
    data = data_to_model
    d = data.shape[1]; N = len(data)
    μ = torch.tensor(μ_base); C = torch.tensor(C_base) 
    d = len(C)
    p = d*(d+3)/2
    #det_C = torch.linalg.det(C).detach().numpy()  # determinant of the covariance matrix 
    det_C = 1.0 #(if we are using C=Identity matrix)
    
   # I_model = I_first_part(p,d,N,det_C)
    I_model = 0.0 # (because we consider same C for both)
    I_data_g_model = negative_log_likelihood(μ,C,N,data,d, det_C) + p/2 
    #print('NEG LOG:2 ', p/2)
    
    #print('msg len entropy: ', (I_model + I_data_g_model)/len(data_to_model)  )
    return I_model, I_data_g_model







































































