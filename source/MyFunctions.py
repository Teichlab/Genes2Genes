import torch
import seaborn as sb
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import gpytorch
import matplotlib.pyplot as plt
import torch.distributions as td
torch.set_default_dtype(torch.float64)


def negative_log_likelihood(μ,σ,N,data):
    data = torch.tensor(data)
    #opt_mode = True
    #if(opt_mode):
    sum_term = torch.sum(((data - μ)/σ)**2.0)/2.0
    return ((N/2.0)* torch.log(2*torch.tensor(np.pi))) + (N*torch.log(σ)) + sum_term 

   # print('arr sum: ',torch.sum(((data - μ)/σ)**2.0))
   # print('arr grad sum: ', torch.neg(torch.sum((data - μ)/(σ**2)))   )
    reimplemented_mode = True
    # Reimplementation =============================================================
    if(reimplemented_mode):
        ts = time.time()
        sum_term = 0.0
        #grad1_term = 0.0 
        for n in range(N):
            sum_term = sum_term + (((data[n] - μ)/σ)**2.0)
            #grad1_term = grad1_term - ((data[n] - μ)/(σ**2))
        sum_term = sum_term/2.0
        
        te = time.time()
      #  print('TIME: ', te-ts)
        return ((N/2.0)* torch.log(2*torch.tensor(np.pi))) + (N*torch.log(σ)) + sum_term 
    # ================================================================================
    else:
        Gaussian_dist = torch.distributions.Normal(μ,σ)
        sum_term = 0.0
        for n in range(N):
            sum_term = sum_term - Gaussian_dist.log_prob(torch.tensor(data[n])) 
        return sum_term

def compute_expected_Fisher_matrix(μ,σ,N):
    return torch.tensor([[N/(σ**2),0],[0,(2*(N**2)/(σ**4))]]) # depends on σ
    
#def compute_observed_Fisher_matrix(μ,σ):
#    return torch.autograd.functional.hessian(negative_log_likelihood ,(μ,σ))

def I_prior(μ,σ): 
    R_μ = torch.tensor(15.0) # uniform prior for mean over region R_μ 
    R_σ = torch.tensor(3.0) # log σ has a uniform prior
    return torch.log(σ) + torch.log(R_μ * R_σ)  # depends on σ

def I_conway_const(d):
    #if(d==2): # check withdrawn for optimisation (we know this is n=2 for Gaussian!)
    c_2 = torch.tensor(5/(36 * np.sqrt(3)))
    return torch.log(c_2)  # a constant

def run_dist_compute_v3(data_to_model,μ_base, σ_base, print_stat=False):

    #global data
    #global N 
    if(len(data_to_model)==0):
        return 
    μ_base = torch.tensor(μ_base); σ_base=torch.tensor(σ_base) 
    data = data_to_model
    N = torch.tensor(len(data_to_model), requires_grad=False)
    
    # MODEL1 - using base model to encode data
    ts = time.time()
    expected_Fisher = compute_expected_Fisher_matrix(μ_base,σ_base,N)
    te = time.time()
    #  print('elapsed exp fisher - ', te-ts)
    ts = time.time()
    L_θ = negative_log_likelihood(μ_base,σ_base,N,data) - (N*np.log(0.001)) 
    te = time.time()
    #  print('elapsed neg log - ', te-ts)
    # compute the I(base_model) 
    I_base_model = (I_conway_const(d=2) + I_prior(μ_base,σ_base) + (0.5*torch.log(torch.det(expected_Fisher)))) 
    # compute the I(data|base_model)
    I_data_g_base_model = L_θ  + torch.tensor(1.0)
    
    return I_base_model, I_data_g_base_model

    
# random gaussian distributed data generation
def generate_random_dataset(N_datapoints, mean, variance):  
    
    μ = torch.tensor(mean); σ = torch.tensor(variance) 
    if(variance<0):
        μ = torch.distributions.Uniform(0,10.0).rsample() # random μ sampling
        σ = torch.distributions.Uniform(0.8,3.0).rsample() # random σ sampling
        #σ = torch.distributions.HalfCauchy(1).rsample() # random σ sampling   
    
    μ.requires_grad = True
    σ.requires_grad = True
    NormalDist = torch.distributions.Normal(μ,σ)
    D = []
    for n in range(N_datapoints):
        D.append(float(NormalDist.rsample().detach().numpy()))
    #print('True params: [ μ=',μ.data.numpy(), ' , σ=', σ.data.numpy(),']' )
    return D,μ,σ
    

def sample_state(x):
    x = np.cumsum(x)
    rand_num = np.random.rand(1)
   # print(rand_num)
    if(rand_num<=x[0]):
        return 0#'M'
    elif(rand_num>x[0] and rand_num<=x[1]):
        return 1#'W'
    elif(rand_num>x[1] and rand_num<=x[2]):
        return 2#'V'
    elif(rand_num>x[2] and rand_num<=x[3]):
        return 3#'D'
    elif(rand_num>x[3] and rand_num<=x[4]):
        return 4#'I'
    
    
    
    
    
    
    