from tqdm.notebook import tqdm
import numpy as np
from . import MyFunctions
import scipy
import scipy.sparse
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

class BatchAnalyser:
    
    def __init__(self):
        # HOUSEKEEPING GENES LIST FROM  https://www.genomics-online.com/resources/16/5049/housekeeping-genes/
        self.housekeeping_genes = ['RRN18S','ACTB','GAPDH','PGK1','PPIA','RPL13A','RPLP0','ARBP','B2M','YWHAZ','SDHA','TFRC','GUSB','HMBS','HPRT1','TBP']
        
        hg1 = pd.read_csv('/home/jovyan/OrgAlign/source/HousekeepingGeneLists/MostStable.csv',sep=';')
        hg1 = list(hg1['Gene name'])
        self.housekeeping_genes = np.unique(hg1 + self.housekeeping_genes) 
        
    def eval_between_system_batch_effect(self, adata_ref, adata_query, plot=False):
        return self.eval_batch_effect(adata_ref, adata_query, plot=plot)

    def eval_batch_effect(self, adata_ref, adata_query,plot):
        
        housekeeping_genes  = np.intersect1d(self.housekeeping_genes,adata_ref.var_names)
        mean_batch_dist = 0.0
        ncols = 5
        nrows = int(np.ceil(len(housekeeping_genes)/ncols))
        if(plot):
            plt.subplots(nrows,ncols,figsize=(20,nrows*2.5))
            k=1
        for gene in housekeeping_genes:
            if(plot):
                plt.subplot(nrows,ncols,k)
                k+=1
            delta = self.compute_batch_effect(adata_ref, adata_query,gene, plot=plot)
            mean_batch_dist = mean_batch_dist + delta
        mean_batch_dist = mean_batch_dist/len(housekeeping_genes)
        return mean_batch_dist
        
    def eval_within_system_batch_effect(self, adata, batch_key):

        batch_adata = [] 
        batches = []
        for batch in np.unique(adata.obs[batch_key]):
            batch_adata.append(adata[adata.obs[batch_key]==batch])  #if(batch_key =='donor' and (batch.startswith('T') or batch.startswith('P'))):
            batches.append(batch)
        
        housekeeping_genes  = np.intersect1d(self.housekeeping_genes,adata.var_names)
        mean_batch_dists = []
        for i in tqdm(range(len(batch_adata))):
            for j in range(i, len(batch_adata)):
                if(i==j):
                    continue
                mean_batch_dist = self.eval_batch_effect(batch_adata[i],batch_adata[j],plot=False)
                mean_batch_dists.append(mean_batch_dist) 
                #print('mean batch dist = ', mean_batch_dist, ' | ', batches[i], batches[j], '|', i,j)
               # plt.show() 
        print('List of batches: ', batches)
        return np.mean(mean_batch_dists)
        
    def compute_batch_effect(self, adata_ref, adata_query, housekeeping_gene, plot=False):

            if(isinstance(adata_ref.X, scipy.sparse.csr.csr_matrix)):
                ref_data = np.asarray(adata_ref[:,housekeeping_gene].X.todense().flatten())[0] 
            else:
                ref_data = np.asarray(adata_ref[:,housekeeping_gene].X.flatten()) 
            if(isinstance(adata_query.X, scipy.sparse.csr.csr_matrix)):
                query_data = np.asarray(adata_query[:,housekeeping_gene].X.todense().flatten())[0] 
            else:
                query_data = np.asarray(adata_query[:,housekeeping_gene].X.flatten())    
            
            μ_S = np.mean(ref_data); σ_S = np.std(ref_data); 
            μ_T = np.mean(query_data); σ_T = np.std(query_data); 

            dist_ref = MyFunctions.generate_random_dataset(100, μ_S, σ_S)
            dist_query = MyFunctions.generate_random_dataset( 100, μ_T, σ_T)
            if(plot):
                sb.kdeplot(dist_ref[0])
                sb.kdeplot(dist_query[0])
                plt.title(housekeeping_gene)

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
            return match_compression.numpy()

        

        

        

        
