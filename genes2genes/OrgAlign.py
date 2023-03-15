import torch
import time
import regex
import numpy as np
import pandas as pd
import gpytorch
import seaborn as sb
import matplotlib.pyplot as plt
import torch.distributions as td
from tqdm import tqdm 
from scipy.spatial import distance
from scipy.special import softmax
from scipy.special import kl_div

from . import MyFunctions
from . import TimeSeriesPreprocessor
from . import MVG

torch.set_default_dtype(torch.float64)

# M,I,D as usual
# Additional states: W(Wd) , V(Wi) (representing insert direction warps and delete direction warps)

class FiveStateMachine:
    
    def __init__(self, P_mm, P_ii, P_mi, PROHIBIT_CASE):
        
        # ====== M STATE
        #self.P_mm = P_mm/3.0
        #self.P_wm = self.P_mm
        #self.P_vm = self.P_mm
        #self.P_im = (1.0 - self.P_mm - self.P_wm - self.P_vm)/2.0
        #self.P_dm = self.P_im
        #print(self.P_mm + self.P_wm + self.P_vm + self.P_im + self.P_dm )
        #assert(self.P_mm + self.P_wm + self.P_vm + self.P_im + self.P_dm == 1.0) 
        
        self.P_mm = P_mm
        k = (1.0 - self.P_mm)/4.0
        self.P_wm = k
        self.P_vm = k
        self.P_im = k
        self.P_dm = k
        
        # ====== W and V STATES as equivalent to M STATE        
        self.P_ww = self.P_mm
        self.P_mw = self.P_wm
        self.P_vw = self.P_vm
        self.P_iw = self.P_im
        self.P_dw = self.P_dm
        #print(self.P_ww + self.P_mw + self.P_vw + self.P_iw + self.P_dw )        
        #assert(self.P_ww + self.P_mw + self.P_vw + self.P_iw + self.P_dw == 1.0) 
              
        self.P_vv = self.P_mm
        self.P_mv = self.P_vm
        self.P_wv = self.P_wm
        self.P_iv = self.P_im
        self.P_dv = self.P_dm
        #print(self.P_vv + self.P_mv + self.P_wv + self.P_iv + self.P_dv)
        #assert(self.P_vv + self.P_mv + self.P_wv + self.P_iv + self.P_dv == 1.0)
        
        # ====== I STATE
        # prohibit any transition from I or D to a warp state
       # self.P_ii = P_ii/2.0      # USE P_II for prohibitive case
       # self.P_mi = P_mi
       # self.P_wi = 0.0
       # self.P_vi = self.P_ii # USE 0 for prohibitive case
        
        if(PROHIBIT_CASE):
            self.P_ii = P_ii   
            self.P_mi = P_mi
            self.P_wi = 0.0
            self.P_vi = 0.0 
        else:
            self.P_ii = P_ii/2.0      # USE P_II for prohibitive case
            self.P_mi = P_mi
            self.P_wi = 0.0
            self.P_vi = self.P_ii # USE 0 for prohibitive case
        
        self.P_di = 1.0 - self.P_ii - self.P_mi - self.P_wi - self.P_vi
        #print(self.P_ii + self.P_mi + self.P_wi + self.P_vi + self.P_di)
        #assert(self.P_ii + self.P_mi + self.P_wi + self.P_di == 1.0)
        
        # ====== D STATE as equivalent to I STATE
        self.P_md = self.P_mi; 
        self.P_dd = self.P_ii; 
        self.P_id = self.P_di
        self.P_wd = self.P_vi #self.P_wi; 
        self.P_vd = 0.0 #self.P_vi
        #print(self.P_dd + self.P_md + self.P_wd + self.P_vd + self.P_id)
        #assert(self.P_dd + self.P_md + self.P_wd + self.P_vd + self.P_id == 1.0)
        
        # =====================================================
        # encoding length terms 
        self.I_mm = -np.log(self.P_mm)
        self.I_wm = -np.log(self.P_wm)
        self.I_vm = -np.log(self.P_vm)
        self.I_im = -np.log(self.P_im)
        self.I_dm = -np.log(self.P_dm)
        
        self.I_ww = self.I_mm; 
        self.I_mw = self.I_wm; 
        self.I_vw = self.I_vm; 
        self.I_dw = self.I_dm;  
        self.I_iw = self.I_im;
        
        self.I_vv = self.I_mm; 
        self.I_mv = self.I_vm; 
        self.I_wv = self.I_wm; 
        self.I_dv = self.I_dm;  
        self.I_iv = self.I_im;
        
        self.I_ii = -np.log(self.P_ii)
        self.I_mi = -np.log(self.P_mi)
        self.I_wi = -np.log(self.P_wi)
        self.I_vi = -np.log(self.P_vi)
        self.I_di = -np.log(self.P_di)
        
        self.I_dd = -np.log(self.P_dd)
        self.I_md = -np.log(self.P_md)
        self.I_wd = -np.log(self.P_wd) 
        self.I_vd = -np.log(self.P_vd)
        self.I_id = -np.log(self.P_id); 

    def print(self):
        print('M state outgoing: ')
        print('Pmm = ', self.I_mm, self.P_mm)
        print('Pwm = ', self.I_wm, self.P_wm)
        print('Pvm = ', self.I_vm, self.P_vm)
        print('Pim = ', self.I_im, self.P_im)
        print('Pdm = ', self.I_dm, self.P_dm)
        print(self.P_mm + self.P_wm + self.P_vm + self.P_im + self.P_dm)
        
        print('====================')
        print('W state outgoing: ')
        print('Pww = ', self.I_ww, self.P_ww)
        print('Pmw = ', self.I_mw, self.P_mw)
        print('Pvw = ', self.I_vw, self.P_vw)
        print('Piw = ', self.I_iw, self.P_iw)
        print('Pdw = ', self.I_dw, self.P_dw) 
        print(self.P_ww + self.P_mw + self.P_vw + self.P_iw + self.P_dw )
        
        print('====================')
        print('V state outgoing: ')
        print('Pvv = ', self.I_vv, self.P_vv)
        print('Pmv = ', self.I_mv, self.P_mv)
        print('Pwv = ', self.I_wv, self.P_wv)
        print('Piv = ', self.I_iv, self.P_iv)
        print('Pdv = ', self.I_dv, self.P_dv) 
        print(self.P_vv + self.P_mv + self.P_wv + self.P_iv + self.P_dv )
        
        print('====================')
        print('I state outgoing: ')
        print('Pii = ', self.I_ii, self.P_ii)
        print('Pmi = ', self.I_mi, self.P_mi)
        print('Pwi = ', self.I_wi, self.P_wi)   
        print('Pvi = ', self.I_vi, self.P_vi)    
        print('Pdi = ', self.I_di, self.P_di)
        print(self.P_ii + self.P_mi + self.P_wi + self.P_vi + self.P_di)
            
        print('====================')
        print('D state outgoing: ')
        print('Pdd = ', self.I_dd, self.P_dd)
        print('Pmd = ', self.I_md, self.P_md)
        print('Pwd = ', self.I_wd, self.P_wd)
        print('Pvd = ', self.I_vd, self.P_vd)
        print('Pid = ', self.I_id, self.P_id) 
        print(self.P_dd + self.P_md + self.P_wd + self.P_vd + self.P_id )
        
    def _set_all_zero_costs(self):
            self.I_mm = 0.0
            self.I_wm = 0.0
            self.I_vm = 0.0
            self.I_im = 0.0
            self.I_dm = 0.0

            self.I_ww = 0.0 
            self.I_mw = 0.0 
            self.I_vw = 0.0 
            self.I_dw = 0.0
            self.I_iw = 0.0

            self.I_vv = 0.0
            self.I_mv = 0.0
            self.I_wv = 0.0
            self.I_dv = 0.0
            self.I_iv = 0.0

            self.I_ii = 0.0
            self.I_mi = 0.0
            self.I_wi = 0.0
            self.I_vi = 0.0
            self.I_di = 0.0

            self.I_dd = 0.0
            self.I_md = 0.0
            self.I_wd = 0.0 
            self.I_vd = 0.0
            self.I_id = 0.0
            
    def reverse(self):
        P_mm = self.P_mm; P_mi = self.P_im; P_md = self.P_dm; P_mw = self.P_wm; P_mv = self.P_vm
        P_im = self.P_mi; P_ii = self.P_ii; P_id = self.P_di; P_iw = self.P_wi; P_iv = self.P_vi 
        P_dd = self.P_dd; P_dm = self.P_md; P_di = self.P_id; P_dw = self.P_wd; P_dv = self.P_vd
        P_ww = self.P_ww; P_wm = self.P_mw; P_wi = self.P_iw; P_wd = self.P_dw; P_wv = self.P_vw 
        P_vm = self.P_mv; P_vv = self.P_vv; P_vd = self.P_dv; P_vw = self.P_wv; P_vi = self.P_iv 

        self.P_mm = P_mm; self.P_mi = P_mi; self.P_md = P_md; self.P_mw = P_mw; self.P_mv = P_mv
        self.P_im = P_im; self.P_ii = P_ii; self.P_id = P_id; self.P_iw = P_iw; self.P_iv = P_iv
        self.P_dd = P_dd; self.P_dm = P_dm; self.P_di = P_di; self.P_dw = P_dw; self.P_dv = P_dv
        self.P_ww = P_ww; self.P_wm = P_wm; self.P_wi = P_wi; self.P_wd = P_wd; self.P_wv = P_wv 
        self.P_vm = P_vm; self.P_vv = P_vv; self.P_vd = P_vd; self.P_vw = P_vw; self.P_vi = P_vi 
        
        # encoding length terms 
        self.I_mm =  -np.log(self.P_mm); self.I_im =  -np.log(self.P_im); self.I_dm =  -np.log(self.P_dm);
        self.I_wm =  -np.log(self.P_wm); self.I_vm =  -np.log(self.P_vm);

        self.I_ii =  -np.log(self.P_ii); self.I_mi =  -np.log(self.P_mi); self.I_di =  -np.log(self.P_di)
        self.I_wi =  -np.log(self.P_wi); self.I_vi =  -np.log(self.P_vi);
        
        self.I_md = self.I_mi; self.I_dd = self.I_ii; self.I_id = self.I_di
        self.I_wd = self.I_wi; self.I_vd = self.I_vi
        
        self.I_vv =  -np.log(self.P_vv); self.I_mv =  -np.log(self.P_mv); self.I_dv =  -np.log(self.P_dv)
        self.I_wv =  -np.log(self.P_wv); self.I_iv =  -np.log(self.P_iv);
        
        self.I_ww = self.I_vv; self.I_iw = self.I_iv; self.I_dw = self.I_dv; 
        self.I_vw = self.I_wv; self.I_mw = self.I_mv; 

        

class DP5:

    def __init__(self, S,T, backward_run, free_params, zero_transition_costs = False, prohibit_case=True, MVG_MODE_KL=True):#, mean_batch_effect=0.0):
        self.S = S
        self.T = T
        self.S_len = len(S.data_bins)
        self.T_len = len(T.data_bins) 
        #self.mean_batch_effect = mean_batch_effect
        self.MVG_MODE_KL = MVG_MODE_KL
        #print('*** ', self.S_len, self.T_len)
        self.FSA = FiveStateMachine(free_params[0], free_params[1], free_params[2], PROHIBIT_CASE= prohibit_case)
        self.backward_run = backward_run
        if(backward_run):
            self.FSA.reverse() 
        if(zero_transition_costs):
            self.FSA._set_all_zero_costs() 
        self.init_DP_matrices()  
        self.init_backtrackers() 
        self.alignment_str = "" 
        self.init() 
        
    def init_DP_matrices(self):    
        
        self.DP_M_matrix = []
        self.DP_I_matrix = []
        self.DP_D_matrix = []
        self.DP_W_matrix = []
        self.DP_V_matrix = []
        self.DP_util_matrix = [] # to store match and null lengths that make the amount of compression 
        
        for i in range(self.T_len+1):
            self.DP_M_matrix.append(np.repeat(0.0,self.S_len+1))
            self.DP_I_matrix.append(np.repeat(0.0,self.S_len+1))
            self.DP_D_matrix.append(np.repeat(0.0,self.S_len+1))
            self.DP_W_matrix.append(np.repeat(0.0,self.S_len+1))
            self.DP_V_matrix.append(np.repeat(0.0,self.S_len+1))
            
            self.DP_util_matrix.append(np.repeat(None,self.S_len+1))
            
        self.DP_M_matrix = np.matrix(self.DP_M_matrix) 
        self.DP_I_matrix = np.matrix(self.DP_I_matrix)
        self.DP_D_matrix = np.matrix(self.DP_D_matrix) 
        self.DP_W_matrix = np.matrix(self.DP_W_matrix) 
        self.DP_V_matrix = np.matrix(self.DP_V_matrix) 
        
        self.DP_util_matrix = np.matrix(self.DP_util_matrix) 
        
    def init_backtrackers(self):
        self.backtrackers_M = []
        self.backtrackers_I = []
        self.backtrackers_D = []
        self.backtrackers_W = []
        self.backtrackers_V = []
        
        for i in range(self.T_len+1):
            row = []
            for j in range(self.S_len+1):
                row.append([0,0,0]) # record <i,j,state> backtracker pointer info
            self.backtrackers_M.append(row.copy())
            self.backtrackers_I.append(row.copy())
            self.backtrackers_D.append(row.copy())
            self.backtrackers_W.append(row.copy())
            self.backtrackers_V.append(row.copy())
            
    def init(self): 
        
        # DP_M --- first row and first col --- np.inf
        for j in range(1,self.S_len+1):
            self.DP_M_matrix[0,j] = np.inf
            self.backtrackers_M[0][j] = [np.inf,np.inf,np.inf]
            self.DP_W_matrix[0,j] = np.inf
            self.backtrackers_W[0][j] = [np.inf,np.inf,np.inf]
            self.DP_V_matrix[0,j] = np.inf
            self.backtrackers_V[0][j] = [np.inf,np.inf,np.inf]
            
        for i in range(1,self.T_len+1):
            self.DP_M_matrix[i,0] = np.inf
            self.backtrackers_M[i][0] = [np.inf,np.inf,np.inf]
            self.DP_W_matrix[i,0] = np.inf
            self.backtrackers_W[i][0] = [np.inf,np.inf,np.inf]
            self.DP_V_matrix[i,0] = np.inf
            self.backtrackers_V[i][0] = [np.inf,np.inf,np.inf]
        
        # DP_I --- first row np.inf
        for j in range(1,self.S_len+1):
            self.DP_I_matrix[0,j] = np.inf
            self.backtrackers_I[0][j] = [np.inf,np.inf,np.inf]
            
        for i in range(1,self.T_len+1):
            if(isinstance(self.S, TimeSeriesPreprocessor.SummaryTimeSeries)):
                cost_D, cost_I = self.compute_cell(i-1,0, only_non_match=True)
            elif(isinstance(self.S, TimeSeriesPreprocessor.SummaryTimeSeriesMVG)):
                cost_D, cost_I = self.compute_cell_as_MVG(i-1,0, only_non_match=True)
                
            #self.DP_I_matrix[i,0] = self.DP_I_matrix[i-1,0] + cost_I + self.FSA.I_ii  
            
            #  [START] NEW TEST 21122022 ====
            if(i==1):
                self.DP_I_matrix[i,0] = self.DP_I_matrix[i-1,0] + cost_I -np.log(1/3) 
            else:
                self.DP_I_matrix[i,0] = self.DP_I_matrix[i-1,0] + cost_I + self.FSA.I_ii 
            # [END] NEW TEST 21122022 ====
            
            #self.backtrackers_I[i][0] = [i-1,0,4] 
            if(not self.backward_run):
                self.backtrackers_I[i][0] = [i-1,0,4] 
            else:
                self.backtrackers_I[i][0] = [i-1,0,0] 

        # DP_D --- first col np.inf
        for i in range(1,self.T_len+1):
            self.DP_D_matrix[i,0] = np.inf
            self.backtrackers_D[i][0] = [np.inf,np.inf,np.inf]
            
        for j in range(1,self.S_len+1):
            if(isinstance(self.S, TimeSeriesPreprocessor.SummaryTimeSeries)):
                cost_D, cost_I =self.compute_cell(0,j-1, only_non_match=True)
            elif(isinstance(self.S, TimeSeriesPreprocessor.SummaryTimeSeriesMVG)):
                cost_D, cost_I =self.compute_cell_as_MVG(0,j-1, only_non_match=True)
                
            #self.DP_D_matrix[0,j] = self.DP_D_matrix[0,j-1] + cost_D + self.FSA.I_dd
            
            #  [START] NEW TEST 21122022 ====
            if(j==1):
                self.DP_D_matrix[0,j] = self.DP_D_matrix[0,j-1] + cost_D -np.log(1/3) 
            else:
                self.DP_D_matrix[0,j] = self.DP_D_matrix[0,j-1] + cost_D + self.FSA.I_dd
            # [END] NEW TEST 21122022 ====
            
            #self.backtrackers_D[0][j] = [0,j-1,3] 
            if(not self.backward_run):
                self.backtrackers_D[0][j] = [0,j-1,3]  
            else:
                self.backtrackers_D[0][j] = [0,j-1,1]  
        
    def run_optimal_alignment(self):
        
        #for i in tqdm(range(1,self.T_len+1)):
        for i in range(1,self.T_len+1):
            for j in range(1,self.S_len+1):
                if(isinstance(self.S, TimeSeriesPreprocessor.SummaryTimeSeries)):
                    match_len,non_match_len_D,non_match_len_I = self.compute_cell(i-1,j-1) # here we use i-1 and j-1 to correctly call the time bin to use 
                elif(isinstance(self.S, TimeSeriesPreprocessor.SummaryTimeSeriesMVG)):
                    match_len,non_match_len_D,non_match_len_I = self.compute_cell_as_MVG(i-1,j-1)
                   
                if(not self.backward_run):
                    # filling M matrix
                   # temp_m = [  self.DP_M_matrix[i-1,j-1] + match_len + self.FSA.I_mm, # 0
                   #             self.DP_W_matrix[i-1,j-1] + match_len  + self.FSA.I_mw, # 1
                   #             self.DP_V_matrix[i-1,j-1] + match_len + self.FSA.I_mv, # 2
                   #             self.DP_D_matrix[i-1,j-1]  + match_len + self.FSA.I_md, # 3
                   #             self.DP_I_matrix[i-1,j-1] + match_len  + self.FSA.I_mi  # 4
                   #          ] 
                    #  [START] NEW TEST 21122022 ====
                    if(i==1 and j==1):
                        temp_m = [  self.DP_M_matrix[i-1,j-1] + match_len -np.log(1/3), #self.FSA.I_mm, # 0
                               np.inf,# self.DP_W_matrix[i-1,j-1] + match_len + np.inf, # -np.log(1/5), #+ self.FSA.I_mw, # 1
                               np.inf,# self.DP_V_matrix[i-1,j-1] + match_len + np.inf, # -np.log(1/5),  #+ self.FSA.I_mv, # 2
                               np.inf,# self.DP_D_matrix[i-1,j-1]  + match_len -np.log(1/5), #+ self.FSA.I_md, # 3
                               np.inf# self.DP_I_matrix[i-1,j-1] + match_len  -np.log(1/5) #+ self.FSA.I_mi  # 4
                             ]  
                    else:      
                        temp_m = [  self.DP_M_matrix[i-1,j-1] + match_len + self.FSA.I_mm, # 0
                                self.DP_W_matrix[i-1,j-1] + match_len  + self.FSA.I_mw, # 1
                                self.DP_V_matrix[i-1,j-1] + match_len + self.FSA.I_mv, # 2
                                self.DP_D_matrix[i-1,j-1]  + match_len + self.FSA.I_md, # 3
                                self.DP_I_matrix[i-1,j-1] + match_len  + self.FSA.I_mi  # 4
                             ] 
                    
                    # [END] NEW TEST 21122022 ====
                    
                    # filling W matrix 
                    temp_w = [  self.DP_M_matrix[i,j-1] + match_len + self.FSA.I_wm, 
                                self.DP_W_matrix[i,j-1]  + match_len  + self.FSA.I_ww, 
                                self.DP_V_matrix[i,j-1]  + match_len  + self.FSA.I_wv, 
                                self.DP_D_matrix[i,j-1]  + match_len + self.FSA.I_wd, 
                                self.DP_I_matrix[i,j-1]  + match_len +  self.FSA.I_wi
                             ]
                    # filling V matrix  
                    temp_v = [ self.DP_M_matrix[i-1,j]  + match_len+ self.FSA.I_vm, 
                                self.DP_W_matrix[i-1,j]   + match_len + self.FSA.I_vw, 
                                self.DP_V_matrix[i-1,j]   + match_len + self.FSA.I_vv, 
                                self.DP_D_matrix[i-1,j]  + match_len+ self.FSA.I_vd, 
                                self.DP_I_matrix[i-1,j]  + match_len+ self.FSA.I_vi
                             ]

                    # filling D matrix
                    temp_d = [  self.DP_M_matrix[i,j-1]  + non_match_len_D + self.FSA.I_dm, 
                                self.DP_W_matrix[i,j-1]  + non_match_len_D + self.FSA.I_dw,
                                self.DP_V_matrix[i,j-1]  + non_match_len_D + self.FSA.I_dv, 
                                self.DP_D_matrix[i,j-1]  + non_match_len_D + self.FSA.I_dd, 
                                self.DP_I_matrix[i,j-1]  + non_match_len_D + self.FSA.I_di
                             ]
                    # filling I matrix
                    temp_i = [  self.DP_M_matrix[i-1,j]  + non_match_len_I + self.FSA.I_im, 
                                self.DP_W_matrix[i-1,j]  + non_match_len_I + self.FSA.I_iw, 
                                self.DP_V_matrix[i-1,j]  + non_match_len_I + self.FSA.I_iv, 
                                self.DP_D_matrix[i-1,j]  + non_match_len_I + self.FSA.I_id, 
                                self.DP_I_matrix[i-1,j]  + non_match_len_I + self.FSA.I_ii
                             ]
                else:
                    # filling M matrix
                    temp_m = [  self.DP_I_matrix[i-1,j-1] + match_len + self.FSA.I_mi, # 0
                                self.DP_D_matrix[i-1,j-1] + match_len + self.FSA.I_md, # 1
                                self.DP_V_matrix[i-1,j-1] + match_len + self.FSA.I_mv, # 2
                                self.DP_W_matrix[i-1,j-1] + match_len + self.FSA.I_mw, # 3
                                self.DP_M_matrix[i-1,j-1] + match_len + self.FSA.I_mm  # 4
                             ] 
                    # filling W matrix 
                    temp_w = [  self.DP_I_matrix[i,j-1] + match_len +  self.FSA.I_wi,
                                self.DP_D_matrix[i,j-1] + match_len+ self.FSA.I_wd, 
                                self.DP_V_matrix[i,j-1]  + match_len + self.FSA.I_wv, 
                                self.DP_W_matrix[i,j-1]  + match_len + self.FSA.I_ww,
                                self.DP_M_matrix[i,j-1] + match_len+ self.FSA.I_wm
                             ]
                    # filling V matrix  
                    temp_v = [  self.DP_I_matrix[i-1,j] + match_len + self.FSA.I_vi,
                                self.DP_D_matrix[i-1,j] + match_len + self.FSA.I_vd,
                                self.DP_V_matrix[i-1,j] + match_len + self.FSA.I_vv,
                                self.DP_W_matrix[i-1,j] + match_len + self.FSA.I_vw, 
                                self.DP_M_matrix[i-1,j] + match_len + self.FSA.I_vm
                             ]

                    # filling D matrix
                    temp_d = [  self.DP_I_matrix[i,j-1] + non_match_len_D + self.FSA.I_di,
                                self.DP_D_matrix[i,j-1] + non_match_len_D + self.FSA.I_dd,
                                self.DP_V_matrix[i,j-1] + non_match_len_D + self.FSA.I_dv,
                                self.DP_W_matrix[i,j-1] + non_match_len_D + self.FSA.I_dw,
                                self.DP_M_matrix[i,j-1] + non_match_len_D + self.FSA.I_dm
                             ]
                    # filling I matrix
                    temp_i = [  self.DP_I_matrix[i-1,j] + non_match_len_I + self.FSA.I_ii,
                                self.DP_D_matrix[i-1,j] + non_match_len_I + self.FSA.I_id, 
                                self.DP_V_matrix[i-1,j] + non_match_len_I + self.FSA.I_iv,
                                self.DP_W_matrix[i-1,j] + non_match_len_I + self.FSA.I_iw,
                                self.DP_M_matrix[i-1,j] + non_match_len_I + self.FSA.I_im
                             ]

                tot_m = min(temp_m) 
                min_idx_m = temp_m.index(tot_m)
                tot_d = min(temp_d) 
                min_idx_d = temp_d.index(tot_d)
                tot_i = min(temp_i) 
                min_idx_i = temp_i.index(tot_i)
                tot_w = min(temp_w)
                min_idx_w = temp_w.index(tot_w)
                tot_v = min(temp_v)
                min_idx_v = temp_v.index(tot_v)
            
                self.DP_M_matrix[i,j] =  tot_m #+ match_len
                self.DP_I_matrix[i,j] =  tot_i #+ non_match_len_I
                self.DP_D_matrix[i,j] =  tot_d #+ non_match_len_D
                self.DP_W_matrix[i,j] =  tot_w #+ match_len
                self.DP_V_matrix[i,j] =  tot_v #+ match_len
                
               # print(i,j,match_len,non_match_len_I,non_match_len_D )
               # print('** ', self.DP_M_matrix[i,j],self.DP_I_matrix[i,j],self.DP_D_matrix[i,j]   )
               # print('temp ', temp_m, temp_i,temp_d  )
                     
                # save backtracker info
                self.backtrackers_M[i][j] = [i-1,j-1,min_idx_m]
                self.backtrackers_W[i][j] = [i,j-1,min_idx_w]
                self.backtrackers_V[i][j] = [i-1,j,min_idx_v]
                self.backtrackers_D[i][j] = [i,j-1,min_idx_d]
                self.backtrackers_I[i][j] = [i-1,j,min_idx_i]
                             
        
        # RETURN TOT MESSAGE LENGTH OF THE OPTIMAL ALIGNMENT    
        if(not self.backward_run):
            self.opt_cost = min([self.DP_M_matrix[self.T_len,self.S_len], 
                               self.DP_W_matrix[self.T_len,self.S_len],
                               self.DP_V_matrix[self.T_len,self.S_len], 
                               self.DP_D_matrix[self.T_len,self.S_len], 
                               self.DP_I_matrix[self.T_len,self.S_len]])
            return 
        else:
            self.opt_cost = min([self.DP_I_matrix[self.T_len,self.S_len],
                        self.DP_D_matrix[self.T_len,self.S_len], 
                        self.DP_V_matrix[self.T_len,self.S_len],
                        self.DP_W_matrix[self.T_len,self.S_len],
                        self.DP_M_matrix[self.T_len,self.S_len]
                    ])
            return 
    
        

    #def criterion2_compute_cell(self,i,j,only_non_match=False):
    def compute_cell(self,i,j,only_non_match=False):
        # Maximising the COMPRESSION ============
        if(only_non_match):
            return 0.0,0.0
        
        μ_S = self.S.mean_trend[j]; σ_S = self.S.std_trend[j]; 
        ref_data = self.S.data_bins[j]
        μ_T = self.T.mean_trend[i]; σ_T = self.T.std_trend[i]; 
        query_data = self.T.data_bins[i]

        I_ref_model, I_refdata_g_ref_model = MyFunctions.run_dist_compute_v3(ref_data, μ_S, σ_S) 
        I_query_model, I_querydata_g_query_model = MyFunctions.run_dist_compute_v3(query_data, μ_T, σ_T) 
        I_ref_model, I_querydata_g_ref_model = MyFunctions.run_dist_compute_v3(query_data, μ_S, σ_S) 
        I_query_model, I_refdata_g_query_model = MyFunctions.run_dist_compute_v3(ref_data, μ_T, σ_T) 
        
        match_encoding_len1 = I_ref_model + I_querydata_g_ref_model + I_refdata_g_ref_model
        match_encoding_len1 = match_encoding_len1/(len(query_data)+len(ref_data))
        match_encoding_len2 = I_query_model + I_refdata_g_query_model + I_querydata_g_query_model
        match_encoding_len2 = match_encoding_len2/(len(query_data)+len(ref_data))
        match_encoding_len = (match_encoding_len1 + match_encoding_len2 )/2.0 
        #match_encoding_len = torch.min(torch.tensor([match_encoding_len, match_encoding_len2]))
        
        null = (I_ref_model + I_refdata_g_ref_model + I_query_model + I_querydata_g_query_model)/(len(query_data)+len(ref_data))
        match_compression =   match_encoding_len - null 
        
        #match_compression = match_compression - self.mean_batch_effect # [POSSIBLE METHOD]constant adjustment for accounting for batch effect
        
        non_match_encoding_len_D = 0.0
        non_match_encoding_len_I = 0.0 
        self.DP_util_matrix[i+1,j+1] = [null.numpy(),match_encoding_len.numpy(),match_compression.numpy()]

        return match_compression.numpy(), non_match_encoding_len_D, non_match_encoding_len_I


    def compute_cell_as_MVG(self,i,j,only_non_match=False):
        
        if(only_non_match):
            return 0.0,0.0
        
        ref_data = torch.tensor(np.asarray(self.S.data_bins[j]) )
        query_data = torch.tensor(np.asarray(self.T.data_bins[i]) )

       # μ_S, C_S = MVG.compute_mml_estimates(ref_data, ref_data.shape[1], ref_data.shape[0]) 
       # C_S = torch.eye(ref_data.shape[1])
       # μ_T, C_T = MVG.compute_mml_estimates(query_data, query_data.shape[1], query_data.shape[0]) 
       # C_T = torch.eye(query_data.shape[1])
        
        μ_S = self.S.mean_trends[j]
        μ_T = self.T.mean_trends[i]
        
        if(self.MVG_MODE_KL):
            return Utils().compute_KLDivBasedDist(μ_S,μ_T), 0.0,0.0 
        
        #return distance.euclidean(μ_S,μ_T), 0.0,0.0 
        
        C_S = torch.eye(ref_data.shape[1])
        C_T = torch.eye(query_data.shape[1])

        I_ref_model, I_refdata_g_ref_model  = MVG.run_dist_compute_v3(ref_data, μ_S, C_S)
        I_query_model, I_querydata_g_query_model  = MVG.run_dist_compute_v3(query_data,μ_T, C_T)
        I_ref_model, I_querydata_g_ref_model = MVG.run_dist_compute_v3(query_data, μ_S, C_S) 
        I_query_model, I_refdata_g_query_model = MVG.run_dist_compute_v3(ref_data, μ_T, C_T) 
        
        match_encoding_len1 = I_ref_model + I_querydata_g_ref_model + I_refdata_g_ref_model
        match_encoding_len1 = match_encoding_len1/(len(query_data)+len(ref_data))
        match_encoding_len2 = I_query_model + I_refdata_g_query_model + I_querydata_g_query_model
        match_encoding_len2 = match_encoding_len2/(len(query_data)+len(ref_data))
        match_encoding_len = (match_encoding_len1 + match_encoding_len2 )/2.0 
        
        null = (I_ref_model + I_refdata_g_ref_model + I_query_model + I_querydata_g_query_model)/(len(query_data)+len(ref_data))
        
        #n_dimensions = self.S.data_bins[0].shape[0]
        match_compression =   (match_encoding_len - null)
        non_match_encoding_len_D = 0.0
        non_match_encoding_len_I = 0.0 
        self.DP_util_matrix[i+1,j+1] = [null,match_encoding_len,match_compression]
        
        assert(null>=0.0)
        assert(match_encoding_len>=0.0)
        
        return match_compression,non_match_encoding_len_D,non_match_encoding_len_I
    

    def _backtrack_util(self,backtracker_pointer):
        
        prev_i = backtracker_pointer[0]
        prev_j = backtracker_pointer[1]
        prev_state = '-'

        if(not self.backward_run):
            if(backtracker_pointer[2]==0):
                prev_state = 'M'
            elif(backtracker_pointer[2]==1): 
                prev_state = 'W'
            elif(backtracker_pointer[2]==2):
                prev_state = 'V'
            elif(backtracker_pointer[2]==3):
                prev_state = 'D'
            elif(backtracker_pointer[2]==4):
                prev_state = 'I'

        else:
            if(backtracker_pointer[2]==0):
                prev_state = 'I'
            elif(backtracker_pointer[2]==1):
                prev_state = 'D'
            elif(backtracker_pointer[2]==2):
                prev_state = 'V'
            elif(backtracker_pointer[2]==3): 
                prev_state = 'W'
            elif(backtracker_pointer[2]==4):
                prev_state = 'M'

        return prev_i, prev_j, prev_state
    
    def backtrack(self):
        self.alignment_str = ""
        j = self.S_len ; i = self.T_len
        self.S_str = "" ; self.T_str = ""
        tracked_path = [] 
        # seek backtrack starting point 
        if(not self.backward_run):
            last_cell_costs = [self.DP_M_matrix[i,j], 
                               self.DP_W_matrix[i,j],
                               self.DP_V_matrix[i,j], 
                               self.DP_D_matrix[i,j], 
                               self.DP_I_matrix[i,j]]
            min_idx = last_cell_costs.index(min(last_cell_costs))  

            #print('tot_msg_len_of_alignment = ', min(last_cell_costs)) 
            if(min_idx==0): # match
                state = 'M'
            elif(min_idx==1):
                state = 'W'
            elif(min_idx==2):
                state = 'V'
            elif(min_idx==3): # delete
                state = 'D'
            elif(min_idx==4): # insert
                state = 'I'
        else:
            last_cell_costs = [self.DP_I_matrix[i,j], 
                               self.DP_D_matrix[i,j],
                               self.DP_V_matrix[i,j], 
                               self.DP_W_matrix[i,j], 
                               self.DP_M_matrix[i,j]]
            min_idx = last_cell_costs.index(min(last_cell_costs))  

            #print('tot_msg_len_of_alignment = ', min(last_cell_costs)) 
            if(min_idx==0): # match
                state = 'I'
            elif(min_idx==1):
                state = 'D'
            elif(min_idx==2):
                state = 'V'
            elif(min_idx==3): # delete
                state = 'W'
            elif(min_idx==4): # insert
                state = 'M'
            
        #  self.alignment_str = state + self.alignment_str
        #  self._align_str_util(state)
        while(True): 
            if(i==0 and j==0):
                break
            #print(i,j,state)
            if(state=='M'): # match
                prev_i, prev_j, prev_state = self._backtrack_util(self.backtrackers_M[i][j])
            elif(state=='D'): # delete
                prev_i, prev_j, prev_state = self._backtrack_util(self.backtrackers_D[i][j])
            elif(state=='I'): 
                prev_i, prev_j, prev_state = self._backtrack_util(self.backtrackers_I[i][j]) 
            elif(state=='W'):
                prev_i, prev_j, prev_state = self._backtrack_util(self.backtrackers_W[i][j]) 
            elif(state=='V'):
                prev_i, prev_j, prev_state = self._backtrack_util(self.backtrackers_V[i][j]) 
            #self._align_str_util(state)
            self.alignment_str = state + self.alignment_str
            #print(i,j, state)
            tracked_path.append([i,j])
            #print('goto --> ', prev_i, prev_j, prev_state )
            i = prev_i
            j = prev_j
            state = prev_state
            #self.alignment_str = state + self.alignment_str

        return tracked_path
                
            
     
    def get_matched_regions(self):
        D_regions = [(m.start(0), m.end(0)) for m in regex.finditer("D+", self.alignment_str)]
        I_regions = [(m.start(0), m.end(0)) for m in regex.finditer("I+", self.alignment_str)]
        M_regions = [(m.start(0), m.end(0)) for m in regex.finditer("M+", self.alignment_str)] 
        W_regions = [(m.start(0), m.end(0)) for m in regex.finditer("W+", self.alignment_str)]
        V_regions = [(m.start(0), m.end(0)) for m in regex.finditer("V+", self.alignment_str)]
        def resolve(regions):
            for i in range(len(regions)):
                x = list(regions[i]); x[1] = x[1]-1; regions[i] = x
            return regions
        M_regions = resolve(M_regions); D_regions = resolve(D_regions); I_regions = resolve(I_regions)
        i = 0; j = 0; m_id = 0; i_id = 0; d_id = 0; c = 0
        S_match_regions = []; T_match_regions = []
        S_non_match_regions = []; T_non_match_regions = []
        a1 = ""; a2 = ""

        while(c<len(self.alignment_str)):
            if(self.alignment_str[c]=='M'):
                step = (M_regions[m_id][1] - M_regions[m_id][0] + 1)
                S_match_regions.append([j,j+step-1]); T_match_regions.append([i,i+step-1])
                i = i + step; j = j + step; m_id = m_id + 1
                a1 = a1 + "*"*(step); a2 = a2 + "*"*(step)
                # process W,V separately 
            if(self.alignment_str[c]=='I'):
                step = (I_regions[i_id][1] - I_regions[i_id][0] + 1)
                T_non_match_regions.append([i,i+step-1])
                i = i + step; i_id = i_id + 1
                a1 = a1 + "-"*(step); a2 = a2 + "*"*(step)
            if(self.alignment_str[c]=='D'):
                step = (D_regions[d_id][1] - D_regions[d_id][0] + 1)
                S_non_match_regions.append([j,j+step-1])
                j = j + step; d_id = d_id + 1
                a1 = a1 + "*"*(step); a2 = a2 + "-"*(step)
            c = c + step 

        return [S_match_regions, T_match_regions, S_non_match_regions, T_non_match_regions]
    
    
class AlignmentLandscape:
    
    def __init__(self, fwd_DP, bwd_DP, S_len, T_len, alignment_path, the_5_state_machine = False):
        self.the_5_state_machine = the_5_state_machine
        self.fwd_DP_M = fwd_DP.DP_M_matrix
        self.fwd_DP_I = fwd_DP.DP_I_matrix
        self.fwd_DP_D = fwd_DP.DP_D_matrix 
        if(the_5_state_machine):
            self.fwd_DP_W = fwd_DP.DP_W_matrix
            self.fwd_DP_V = fwd_DP.DP_V_matrix
        if(bwd_DP!=None): 
            self.bwd_DP_M = bwd_DP.DP_M_matrix
            self.bwd_DP_I = bwd_DP.DP_I_matrix
            self.bwd_DP_D = bwd_DP.DP_D_matrix
            if(the_5_state_machine):
                self.bwd_DP_W = bwd_DP.DP_W_matrix
                self.bwd_DP_V = bwd_DP.DP_V_matrix
        
        self.FSA = fwd_DP.FSA
        self.S_len = S_len
        self.T_len = T_len
        
        self.L_matrix = []
        for i in range(T_len+1):
            self.L_matrix.append(np.repeat(0.0,S_len+1))
        self.L_matrix = np.matrix(self.L_matrix) 
        
        self.L_matrix_states = []
        for i in range(T_len+1):
            self.L_matrix_states.append(np.repeat('',S_len+1))
        self.L_matrix_states = np.matrix(self.L_matrix_states) 
        self.alignment_path = alignment_path
        
    def collate(self):
        
        #for i in tqdm(range(0,self.T_len+1)):
        for i in range(0,self.T_len+1):
            for j in range(0,self.S_len+1):
                # considering all possible state transitions (cartesian product of {m,i,d})
                _i = self.T_len-i
                _j = self.S_len-j
                
                if(not self.the_5_state_machine):
                    temp = [ self.fwd_DP_M[i,j] + self.bwd_DP_M[_i, _j] + self.FSA.I_mm, 
                             self.fwd_DP_M[i,j] + self.bwd_DP_I[_i, _j] + self.FSA.I_im,
                             self.fwd_DP_M[i,j] + self.bwd_DP_D[_i, _j] + self.FSA.I_dm,
                             self.fwd_DP_I[i,j] + self.bwd_DP_M[_i, _j] + self.FSA.I_mi, 
                             self.fwd_DP_I[i,j] + self.bwd_DP_I[_i, _j] + self.FSA.I_ii, 
                             self.fwd_DP_I[i,j] + self.bwd_DP_D[_i, _j] + self.FSA.I_di, 
                             self.fwd_DP_D[i,j] + self.bwd_DP_M[_i, _j] + self.FSA.I_md, 
                             self.fwd_DP_D[i,j] + self.bwd_DP_I[_i, _j] + self.FSA.I_id, 
                             self.fwd_DP_D[i,j] + self.bwd_DP_D[_i, _j] + self.FSA.I_dd ] 
                else:
                    temp = [ self.fwd_DP_M[i,j] + self.bwd_DP_M[_i, _j] + self.FSA.I_mm, 
                         self.fwd_DP_M[i,j] + self.bwd_DP_W[_i, _j] + self.FSA.I_wm,
                         self.fwd_DP_M[i,j] + self.bwd_DP_V[_i, _j] + self.FSA.I_vm,
                         self.fwd_DP_M[i,j] + self.bwd_DP_D[_i, _j] + self.FSA.I_dm,
                         self.fwd_DP_M[i,j] + self.bwd_DP_I[_i, _j] + self.FSA.I_im,
                         self.fwd_DP_W[i,j] + self.bwd_DP_M[_i, _j] + self.FSA.I_mw,
                         self.fwd_DP_W[i,j] + self.bwd_DP_W[_i, _j] + self.FSA.I_ww,
                         self.fwd_DP_W[i,j] + self.bwd_DP_V[_i, _j] + self.FSA.I_vw,
                         self.fwd_DP_W[i,j] + self.bwd_DP_D[_i, _j] + self.FSA.I_dw,
                         self.fwd_DP_W[i,j] + self.bwd_DP_I[_i, _j] + self.FSA.I_iw,
                         self.fwd_DP_V[i,j] + self.bwd_DP_M[_i, _j] + self.FSA.I_mv, 
                         self.fwd_DP_V[i,j] + self.bwd_DP_W[_i, _j] + self.FSA.I_wv,
                         self.fwd_DP_V[i,j] + self.bwd_DP_V[_i, _j] + self.FSA.I_vv,
                         self.fwd_DP_V[i,j] + self.bwd_DP_D[_i, _j] + self.FSA.I_dv,
                         self.fwd_DP_V[i,j] + self.bwd_DP_I[_i, _j] + self.FSA.I_iv,
                         self.fwd_DP_D[i,j] + self.bwd_DP_M[_i, _j] + self.FSA.I_md, 
                         self.fwd_DP_D[i,j] + self.bwd_DP_W[_i, _j] + self.FSA.I_wd, 
                         self.fwd_DP_D[i,j] + self.bwd_DP_V[_i, _j] + self.FSA.I_vd,
                         self.fwd_DP_D[i,j] + self.bwd_DP_D[_i, _j] + self.FSA.I_dd,
                         self.fwd_DP_D[i,j] + self.bwd_DP_I[_i, _j] + self.FSA.I_id, 
                         self.fwd_DP_I[i,j] + self.bwd_DP_M[_i, _j] + self.FSA.I_mi, 
                         self.fwd_DP_I[i,j] + self.bwd_DP_W[_i, _j] + self.FSA.I_wi, 
                         self.fwd_DP_I[i,j] + self.bwd_DP_V[_i, _j] + self.FSA.I_vi, 
                         self.fwd_DP_I[i,j] + self.bwd_DP_D[_i, _j] + self.FSA.I_di, 
                         self.fwd_DP_I[i,j] + self.bwd_DP_I[_i, _j] + self.FSA.I_ii
                       ]
                    #print(min(temp))
                self.L_matrix[i,j] = min(temp)
        
        
    def collate_fwd(self):
        
        #for i in tqdm(range(0,self.T_len+1)):
        for i in range(0,self.T_len+1):
            for j in range(0,self.S_len+1):
                _i = self.T_len-i
                _j = self.S_len-j
                
                if(not self.the_5_state_machine):
                    temp = [ self.fwd_DP_M[i,j], self.fwd_DP_D[i,j], self.fwd_DP_I[i,j]]
                else:
                    temp = [ self.fwd_DP_M[i,j],self.fwd_DP_W[i,j] ,self.fwd_DP_V[i,j], 
                           self.fwd_DP_D[i,j], self.fwd_DP_I[i,j]]
                    #print(min(temp))
                self.L_matrix[i,j] = min(temp)
                self.L_matrix_states[i,j] = np.argmin(temp) 
                
        
    def plot_alignment_landscape(self): # pass alignment path coordinates
        fig, ax = plt.subplots(figsize=(5, 5))
        ax = sb.heatmap(self.L_matrix, square=True,  cmap="jet", ax=ax)
        path_x = [p[0]+0.5 for p in self.alignment_path]
        path_y = [p[1]+0.5 for p in self.alignment_path]
        ax.plot(path_y, path_x, color='black', linewidth=3, alpha=0.5, linestyle='dashed') # path plot
        plt.xlabel("S",fontweight='bold')
        plt.ylabel("T",fontweight='bold')            
        
        
class Utils:

    def compute_alignment_area_diff_distance(self, A1, A2, S_len, T_len):

       # print(A1)
       # print(A2)
        pi = np.arange(1, S_len+T_len+1) # skew diagonal indices 
        A1_ = ""
        for c in A1:
            A1_ = A1_ + c
            if(c=='M'):
                A1_ = A1_ + 'X'
        A2_ = ""
        for c in A2:
            A2_ = A2_ + c 
            if(c=='M'):
                A2_ = A2_ + 'X'

        pi_1_k = 0
        pi_2_k = 0
        #print(0, pi_1_k , pi_2_k )
        A1_al_index = 0
        A2_al_index = 0
        absolute_dist_sum = 0.0
        for k in pi:
            #print('k=',k, A1_al_index, A2_al_index)
            A1_state = A1_[A1_al_index]
            A2_state = A2_[A2_al_index]
            if(A1_state=='I' or A1_state=='V'):
                pi_1_k = pi_1_k - 1
            elif(A1_state=='D' or  A1_state=='W'):
                pi_1_k = pi_1_k + 1 
            if(A2_state=='I' or A2_state=='V'):
                pi_2_k = pi_2_k - 1   
            elif(A2_state=='D' or  A2_state=='W'):
                pi_2_k = pi_2_k + 1   
            #print(k, pi_1_k, pi_2_k)
            #print(A1_state, A2_state)
            absolute_dist_sum = absolute_dist_sum + np.abs(pi_1_k - pi_2_k)
            #print('-----')
            A1_al_index = A1_al_index + 1
            A2_al_index = A2_al_index + 1 

        return absolute_dist_sum

    def compute_chattergi_coefficient(self, y1,y2):
        df = pd.DataFrame({'S':y1, 'T':y2})
        df['rankS'] = df['S'].rank() 
        df['rankT'] = df['T'].rank() 
        # sort df by the S variable first
        df = df.sort_values(by='rankS')
        return 1 - ((3.0 * df['rankT'].diff().abs().sum())/((len(df)**2)-1)) 
    
    
    def plot_different_alignments(self, paths, S_len, T_len, ax, mat=[]): # pass alignment path coordinates
        mat=[]
      #  if(len(mat)==0):
        for i in range(T_len+1):
            mat.append(np.repeat(0,S_len+1)) 
        sb.heatmap(mat, square=True,  cmap='viridis', ax=ax, vmin=0, vmax=0, cbar=False,xticklabels=False,yticklabels=False)
        path_color = "orange"
     #   else:
     #       sb.heatmap(mat, square=True,  cmap='viridis', cbar=False,xticklabels=False,yticklabels=False, vmax=1)
     #       path_color = "black"
        
        for path in paths: 
            path_x = [p[0]+0.5 for p in path]
            path_y = [p[1]+0.5 for p in path]
            ax.plot(path_y, path_x, color=path_color, linewidth=3, alpha=0.5, linestyle='dashed') # path plot
        plt.xlabel("S",fontweight='bold')
        plt.ylabel("T",fontweight='bold')
        
    
    def check_alignment_clusters(self, n_clusters , cluster_ids, alignments, n_cols = 5, figsize= (10,6)): 
        
        clusters = [] 
        S_len = alignments[0].fwd_DP.S_len
        T_len = alignments[0].fwd_DP.T_len
        
        unique_cluster_ids = np.unique(cluster_ids)
        n_rows = int(np.ceil(n_clusters/n_cols)) 
        

        fig, axs = plt.subplots(n_rows,n_cols, figsize = (20,n_rows*3)) # custom -- only for 20 clusters -- TODO change later
        axs = axs.flatten() 
        i = 0
        k=1
        for cluster_id in range(n_clusters): 
            paths = []
            cluster_genes = [] 
            cluster_alignments = np.asarray(alignments)[cluster_ids == unique_cluster_ids[cluster_id]]
            for a in cluster_alignments:
                    paths.append(a.fwd_DP.alignment_path)
                    #print(a.gene)
                    cluster_genes.append(a.gene);# cluster_genes.append(a.gene)
            clusters.append(list(np.unique(cluster_genes)) ) 
    #        self.plot_different_alignments(paths, S_len, T_len, ax=axs[cluster_id])
            
            ####
   #         mat = []
   #         for i in range(T_len+1):
   #             mat.append(np.repeat(0,S_len+1)) 
   #         sb.heatmap(mat, square=True,  cmap='viridis', ax=axs[cluster_id], vmin=0, vmax=0, cbar=False,xticklabels=False,yticklabels=False)
   #         path_color = "orange"
   #         for path in paths: 
   #             path_x = [p[0]+0.5 for p in path]
   #             path_y = [p[1]+0.5 for p in path]
    #            axs[cluster_id].plot(path_y, path_x, color=path_color, linewidth=3, alpha=0.5, linestyle='dashed') # path plot
    #        plt.xlabel("S",fontweight='bold')
   #         plt.ylabel("T",fontweight='bold')
            
            self.plot_different_alignments(paths, S_len, T_len, axs[cluster_id])
            axs[cluster_id].set_title('Cluster-'+str(i) + ' | '+str(len(cluster_alignments)))

            i=i+1
            k=k+1
        
        fig.tight_layout()
        n = n_cols * n_rows
        i = 1
        while(k<=n):
            axs.flat[-1*i].set_visible(False) 
            k = k+1
            i=i+1        
            
        return clusters
    

    # input: log1p gene expression vectors
    def compute_KLDivBasedDist(self,x,y):

        # convert to probabilities
      #  x = softmax(x)
      #  y = softmax(y)
        x = x.numpy()
        y = y.numpy()
        # convering backto counts+1
        x = np.exp(x)
        y = np.exp(y)
        x = x/np.sum(x)
        y = y/np.sum(y)

        sum_term = 0.0
        for i in range(len(x)):
            sum_term += x[i]*(np.log(x[i]) - np.log(y[i]))
      #  print(x,y,' ---------- ',sum_term)
        return sum_term



    
    
    
