import torch
import numpy as np

torch.set_default_dtype(torch.float64)

"""
This script defines all methods required for computing mml distance between two gene expression distributions as Gaussian
"""

def negative_log_likelihood(μ,σ,N,data):
    data = torch.tensor(data)
    #opt_mode = True
    #if(opt_mode):
    sum_term = torch.sum(((data - μ)/σ)**2.0)/2.0
    return ((N/2.0)* torch.log(2*torch.tensor(np.pi))) + (N*torch.log(σ)) + sum_term 

def compute_expected_Fisher_matrix(μ,σ,N):
    return torch.tensor([[N/(σ**2),0],[0,(2*N)/(σ**2)]]) # depends on σ
    ####  ---- expected_Fisher = compute_expected_Fisher_matrix(μ_base,σ_base,N) # compute the closed form of matrix determinant instead 

def I_prior(μ,σ): 
    R_μ = torch.tensor(15.0) # uniform prior for mean over region R_μ 
    R_σ = torch.tensor(3.0) # log σ has a uniform prior
    return torch.log(σ) + torch.log(R_μ * R_σ)  # depends on σ

def I_conway_const(d):
    #if(d==2): # check withdrawn for optimisation (we know this is n=2 for Gaussian!)
    c_2 = torch.tensor(5/(36 * np.sqrt(3)))
    return torch.log(c_2)  # a constant

def run_dist_compute_v3(data_to_model,μ_base, σ_base, print_stat=False):

    if(len(data_to_model)==0):
        return 
    μ_base = torch.tensor(μ_base); σ_base=torch.tensor(σ_base) 
    data = data_to_model
    N = torch.tensor(len(data_to_model), requires_grad=False)
    
    # MODEL1 - using base model to encode data

    determinant_of_the_expected_fisher = (2*(N**2))/(σ_base**4)   #torch.det(expected_Fisher) CLOSED FORM

    L_θ = negative_log_likelihood(μ_base,σ_base,N,data) - (N*np.log(0.001)) # Accuracy of Measurement epsilon = 0.001
    
    #I_base_model = (I_conway_const(d=2) + I_prior(μ_base,σ_base) + (0.5*torch.log(torch.det(expected_Fisher)))) 
    I_base_model = (I_conway_const(d=2) + I_prior(μ_base,σ_base) + (0.5*torch.log(determinant_of_the_expected_fisher))) 
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

    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
