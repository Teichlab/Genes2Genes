import numpy as np
from scipy.sparse import csr_matrix
from . import MyFunctions

# UTIL FUNCTIONS
def csr_mat_col_densify(csr_matrix, j): 
        start_ptr = csr_matrix.indptr[j]
        end_ptr = csr_matrix.indptr[j + 1]
        data = csr_matrix.data[start_ptr:end_ptr]
        dense_column = np.zeros(csr_matrix.shape[1])
        dense_column[csr_matrix.indices[start_ptr:end_ptr]] = data
        return dense_column

    
def minmax_normalise(arr):
        
        norm_arr = []
        arr = np.asarray(arr)
        arr_max = np.max(arr)
        arr_min = np.min(arr)
        for i in range(len(arr)):
            norm_arr.append((arr[i] - arr_min )/(arr_max  - arr_min )) 
        return norm_arr
    

# computes distributional distance under the MML framework
def compute_mml_dist(ref_adata_subset,query_adata_subset, gene):

        ref_data = np.asarray(ref_adata_subset[:,gene].X.todense()).flatten()
        query_data = np.asarray(query_adata_subset[:,gene].X.todense()).flatten()
        μ_S = np.mean(ref_data)
        μ_T = np.mean(query_data)
        σ_S =np.std(ref_data)
        σ_T =np.std(query_data)
        #print(μ_S,μ_T)
        if(not np.any(ref_data)):
            σ_S = 0.1
        if(not np.any(query_data)):
            σ_T = 0.1    

        I_ref_model, I_refdata_g_ref_model = MyFunctions.run_dist_compute_v3(ref_data, μ_S, σ_S) 
        I_query_model, I_querydata_g_query_model = MyFunctions.run_dist_compute_v3(query_data, μ_T, σ_T) 
        I_ref_model, I_querydata_g_ref_model = MyFunctions.run_dist_compute_v3(query_data, μ_S, σ_S) 
        I_query_model, I_refdata_g_query_model = MyFunctions.run_dist_compute_v3(ref_data, μ_T, σ_T) 

        match_encoding_len1 = I_ref_model + I_querydata_g_ref_model + I_refdata_g_ref_model
        match_encoding_len1 = match_encoding_len1/(len(query_data)+len(ref_data))
        match_encoding_len2 = I_query_model + I_refdata_g_query_model + I_querydata_g_query_model
        match_encoding_len2 = match_encoding_len2/(len(query_data)+len(ref_data))
        match_encoding_len = (match_encoding_len1 + match_encoding_len2 )/2.0 

        null = (I_ref_model + I_refdata_g_ref_model + I_query_model + I_querydata_g_query_model)/(len(query_data)+len(ref_data))
        match_compression =   match_encoding_len - null 

        return match_compression
    
    
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
    

def compute_alignment_area_diff_distance(A1, A2, S_len, T_len):

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
  
            absolute_dist_sum = absolute_dist_sum + np.abs(pi_1_k - pi_2_k)
            #print('-----')
            A1_al_index = A1_al_index + 1
            A2_al_index = A2_al_index + 1 

        return absolute_dist_sum

def compute_chattergi_coefficient(y1,y2):
        df = pd.DataFrame({'S':y1, 'T':y2})
        df['rankS'] = df['S'].rank() 
        df['rankT'] = df['T'].rank() 
        # sort df by the S variable first
        df = df.sort_values(by='rankS')
        return 1 - ((3.0 * df['rankT'].diff().abs().sum())/((len(df)**2)-1)) 
    
    
def plot_different_alignments(paths, S_len, T_len, ax, mat=[]): # pass alignment path coordinates
        mat=[]
      #  if(len(mat)==0):
        for i in range(T_len+1):
            mat.append(np.repeat(0,S_len+1)) 
        sb.heatmap(mat, square=True,  cmap='viridis', ax=ax, vmin=0, vmax=0, cbar=False,xticklabels=False,yticklabels=False)
        path_color = "orange"
        
        for path in paths: 
            path_x = [p[0]+0.5 for p in path]
            path_y = [p[1]+0.5 for p in path]
            ax.plot(path_y, path_x, color=path_color, linewidth=3, alpha=0.5, linestyle='dashed') # path plot
        plt.xlabel("S",fontweight='bold')
        plt.ylabel("T",fontweight='bold')
        
    
def check_alignment_clusters(n_clusters , cluster_ids, alignments, n_cols = 5, figsize= (10,6)): 
        
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

            plot_different_alignments(paths, S_len, T_len, axs[cluster_id])
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
def compute_KLDivBasedDist(x,y):

        # convert to probabilities
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
    
        return sum_term

    