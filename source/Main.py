import multiprocessing
from multiprocessing import Pool
from tqdm.notebook import tqdm_notebook
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import textdistance 
from Levenshtein import distance
import time
import copy
from sklearn.cluster import AgglomerativeClustering
import scipy
import scipy.sparse
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
from scipy.stats import zscore
import regex 

import OrgAlign as orgalign
import MyFunctions 
import TimeSeriesPreprocessor
import AlignmentDistMan

class AligmentObj:
    
    def __init__(self,gene, S,T, fwd_DP_obj,bwd_DP_obj, landscape_obj):
        
        self.gene = gene
        self.S = S
        self.T = T
        self.alignment_str = fwd_DP_obj.alignment_str
        self.landscape_obj = landscape_obj
        self.fwd_DP = fwd_DP_obj
        self.bwd_DP = bwd_DP_obj
        
        out = self.fwd_DP.get_matched_regions()
        self.match_regions_S = out[0]
        self.match_regions_T = out[1]
        self.non_match_regions_S = out[2]
        self.non_match_regions_T = out[3]
        
        self.compute_series_match_percentage()
        #if(isinstance(S,TimeSeriesPreprocessor.SummaryTimeSeries)):
        try:
            self.run_DEAnalyser() 
        except Exception as e:
            print(str(e),gene)

    def print(self):
        print('Fwd opt cost', self.fwd_DP.opt_cost)
        print(self.fwd_DP.alignment_str) 
        #print('Bwd opt cost: ', self.bwd_DP_obj.opt_cost)
        #print(bwd_DP.alignment_str[::-1])
        print(self.match_regions_S)
        print(self.match_regions_T)
        print(self.non_match_regions_S)
        print(self.non_match_regions_T)
        self.landscape_obj.plot_alignment_landscape() 
        
    def plotTimeSeries(self, refQueryAlignerObj, plot_cells = False, plot_mean_trend= False):
        sb.scatterplot(self.S.X, self.S.Y, color = 'forestgreen' ,alpha=0.05)#, label='Ref') 
        sb.scatterplot(self.T.X, self.T.Y, color = 'midnightblue' ,alpha=0.05)#, label ='Query')
      #  plt.legend(loc='upper left')
        if(plot_cells):
            sb.scatterplot(refQueryAlignerObj.ref_time, np.asarray(refQueryAlignerObj.ref_mat[self.gene]), color = 'forestgreen' ) 
            sb.scatterplot(refQueryAlignerObj.query_time, np.asarray(refQueryAlignerObj.query_mat[self.gene]), color = 'midnightblue' )
        plt.title(self.gene)
        plt.xlabel('pseudotime')
        plt.ylabel('log1p expression')
        
        if(plot_mean_trend):
            self.plot_mean_trends() 
            
    def plotTimeSeriesAlignment(self):  
        sb.scatterplot(self.S.X, self.S.Y, color = 'forestgreen' ,alpha=0.05)#, label='Ref') 
        sb.scatterplot(self.T.X, self.T.Y, color = 'midnightblue' ,alpha=0.05)#, label ='Query') 
      #  plt.legend(loc='upper left')
        self.plot_mean_trends() 
        plt.title(self.gene)
        plt.xlabel('pseudotime')
        plt.ylabel('log1p expression')
        
        for i in range(self.matched_region_DE_info.shape[0]):
            S_timebin = int(self.matched_region_DE_info.iloc[i]['ref_bin'])
            T_timebin = int(self.matched_region_DE_info.iloc[i]['query_bin']) 
            x_vals = [self.matched_region_DE_info.iloc[i]['ref_pseudotime'],self.matched_region_DE_info.iloc[i]['query_pseudotime']] 
            y_vals = [self.S.mean_trend[S_timebin ], self.T.mean_trend[T_timebin]] 
            plt.plot(x_vals, y_vals, color='black', linestyle='dashed', linewidth=0.6)

        
    def get_ref_timeseries_obj(self):
        return self.fwd_DP.S
    
    def get_query_timeseries_obj(self):
        return self.fwd_DP.T
    
    
    def compute_series_match_percentage(self):
        x = np.unique(list(self.alignment_str), return_counts=True) 
        x = dict(zip(x[0],x[1]))
        p1 = 0
        p2 = 0 
        p3 = 0
        for state in ['M','V','W']:
            if(state in x.keys()):
                if(state=='M'):
                    p1 = p1 + x[state] 
                    p2 = p2 + x[state] 
                    p3 = p3 +  x[state] 
                if(state == 'V'):
                    p1 = p1 + x[state] 
                    p2 = p2 +  x[state] 
                if(state == 'W'):
                    p1 = p1 + x[state] 
                    p3 = p3 +  x[state] 
                
        self.match_percentage = round((p1/len(self.alignment_str))*100,2)
        self.match_percentage_T = round((p2/len(self.T.data_bins))*100,2)
        self.match_percentage_S = round((p3/len(self.S.data_bins))*100,2)
        
    def get_series_match_percentage(self):
        return self.match_percentage , self.match_percentage_S, self.match_percentage_T  
    
    def run_DEAnalyser(self):
        #print('DEAnalyser: started')
        DE_analyser = DEAnalyser(self)
        #print('DEAnalyser: get matched regions')
        DE_analyser.get_matched_regions()
        #print('DEAnalyser: get matched timepoints')
        DE_analyser.get_matched_time_points()

        if(isinstance(self.S,TimeSeriesPreprocessor.SummaryTimeSeries)):
            #print('DEAnalyser: get DE info')
            DE_analyser.get_DE_info_for_matched_regions() 
        
    def plot_matched_region_dists(self):
        
        n_cols = 10
        n_rows = int(np.ceil(len(self.match_points_S)/n_cols))
        fig, axs = plt.subplots(n_rows,n_cols,figsize=(20,n_rows*3))

        for i in range(len(self.match_points_S)):
            plt.subplot(n_rows,n_cols,i+1)
            S_bin = self.S.data_bins[self.match_points_S[i]]  
            T_bin = self.T.data_bins[self.match_points_T[i]] 
            sb.kdeplot(S_bin, fill=True)
            sb.kdeplot(T_bin, fill=True)  
            
        fig.tight_layout()
        n = n_cols * n_rows
        k = i
        i = 1
        while(k<=n):
            axs.flat[-1*i].set_visible(False) 
            k = k+1
            i=i+1
            
    def print_alignment(self):

       # print('S matched : ', self.S_match_regions)
       # print('T matched : ', self.T_match_regions)
       # print('S not matched : ', self.S_non_match_regions)
       # print('T not matched : ',  self.T_non_match_regions)
       # print('')
        print(self.al_visual)
        print('Matched percentages: ')
        p1,p2,p3 = self.get_series_match_percentage()
        print('w.r.t alignment: ', p1,'%')
        print('w.r.t ref: ', p2,'%')
        print('w.r.t query: ', p3,'%')
        
    def plot_mean_trends(self):
        # mean trend plot
        self.S.plot_mean_trend(color='forestgreen')
        self.T.plot_mean_trend(color='midnightblue')
        
    def get_opt_alignment_cost(self):
        return self.fwd_DP.opt_cost
    
            
class RefQueryAligner:
    
    def __init__(self, *args):
        if(len(args) ==4 ):
            self.run_init1(args[0], args[1], args[2], args[3])
        elif(len(args) == 6 ): 
            self.run_init2(args[0], args[1], args[2], args[3], args[4], args[5])
        else:
            print('pls pass the required number of args')
    
    def set_n_threads(self,n):
        self.n_threads = n
    
    # converts ref and query anndata objects to pd.DataFrames 
    def run_init1(self, adata_ref, adata_query, gene_list, n_artificial_time_points):
       
        #if(not hasattr(self, 'mean_batch_effect' )):
            #self.mean_batch_effect =  BatchAnalyser.BatchAnalyser().eval_between_system_batch_effect(adata_ref, adata_query)
        
        if(isinstance(adata_ref.X, scipy.sparse.csr.csr_matrix)):
            ref_mat = pd.DataFrame(adata_ref.X.todense()) 
        else:
            ref_mat = pd.DataFrame(adata_ref.X) 
        if(isinstance(adata_query.X, scipy.sparse.csr.csr_matrix)):
            query_mat = pd.DataFrame(adata_query.X.todense()) 
        else:
            query_mat = pd.DataFrame(adata_query.X)     
            
        ref_mat.columns = adata_ref.var_names
        ref_mat = ref_mat.set_index(adata_ref.obs_names)
        ref_time = np.asarray(adata_ref.obs['time']) 
        query_mat.columns = adata_query.var_names
        query_mat = query_mat.set_index(adata_query.obs_names)
        query_time = np.asarray(adata_query.obs['time']) 
        
        self.run_init2(ref_mat, ref_time, query_mat, query_time, gene_list, n_artificial_time_points)
    
    def run_init2(self, ref_mat, ref_time, query_mat, query_time, gene_list, n_artificial_time_points, CONST_STD=False):
        self.ref_mat = ref_mat
        self.query_mat = query_mat
        self.ref_time = ref_time
        self.query_time = query_time
        self.gene_list = gene_list
        self.pairs = {}
        self.n_threads = multiprocessing.cpu_count()
        self.CONST_STD = CONST_STD
        
        # to preserve the number of time points ratio
        time_lens = [len(self.ref_time), len(self.query_time)]  
        self.n_artificial_time_points = n_artificial_time_points
        #self.n_q_points = int(n_artificial_time_points * time_lens[0]/time_lens[1])
        self.n_q_points = n_artificial_time_points
        
       # self.ref_processor = TimeSeriesPreprocessor.Prepocessor(self.ref_mat, self.ref_time, 50)
       # self.query_processor =  TimeSeriesPreprocessor.Prepocessor(self.query_mat, self.query_time, n_q_points)
        
    def run_interpolation(self, gene):
        ref_processor = TimeSeriesPreprocessor.Prepocessor(self.ref_mat, self.ref_time, self.n_artificial_time_points)
        query_processor =  TimeSeriesPreprocessor.Prepocessor(self.query_mat, self.query_time, self.n_q_points)
        
        S = ref_processor.prepare_interpolated_gene_expression_series(gene, WEIGHT_BY_CELL_DENSITY = self.WEIGHT_BY_CELL_DENSITY)
        T = query_processor.prepare_interpolated_gene_expression_series(gene, WEIGHT_BY_CELL_DENSITY = self.WEIGHT_BY_CELL_DENSITY)
        return [S,T]
    
    def interpolate_genes(self):
        with Pool(self.n_threads) as p:
            results = list(tqdm_notebook(p.imap(self.run_interpolation, self.gene_list), total=len(self.gene_list)))
        #print(len(results))
        p.close()
        p.join()
        for i in range(len(self.gene_list)):
            self.pairs[self.gene_list[i]] = results[i]
            
    def align_single_pair(self, gene, state_params = [0.99,0.5,0.4],zero_transition_costs=False, prohibit_case = False): #[0.9,0.5,0.4]):
        
       # if(prohibit_case):
       #     state_params = [0.95,0.3,0.6]
       # else:
       #     state_params=[0.95,0.5,0.4]
 
        if(not(gene in self.pairs.keys())):
           # print('interpolating sequences')
            self.pairs[gene] = self.run_interpolation(gene)
            
        S = self.pairs[gene][0] 
        T = self.pairs[gene][1] 
        
        fwd_DP = orgalign.DP5(S,T,  free_params = state_params, backward_run=False,  zero_transition_costs= zero_transition_costs, prohibit_case = prohibit_case) #,mean_batch_effect=self.mean_batch_effect) 
        fwd_opt_cost = fwd_DP.run_optimal_alignment() 
        alignment_path = fwd_DP.backtrack() 
        fwd_DP.alignment_path = alignment_path

        landscapeObj = orgalign.AlignmentLandscape(fwd_DP, None, len(S.mean_trend), len(T.mean_trend), alignment_path, the_5_state_machine = True)
        landscapeObj.collate_fwd() #landscapeObj.plot_alignment_landscape()
        
        return AligmentObj(gene, S,T,fwd_DP,None, landscapeObj)
        
        #return AligmentObj(gene, S,T,fwd_DP,None, None)
        # Backward alignment =========== 
        S_rev = copy.deepcopy(S)
        T_rev = copy.deepcopy(T)
       # artificial_time_points1 = S.time_points
       # artificial_time_points2 = T.time_points
       # X1 = S.X; Y1 = S.Y; 
       # X2 = T.X; Y2 = T.Y; 
       # proc1 = TimeSeriesPreprocessor.Prepocessor(); proc1.artificial_time_points = artificial_time_points1
       # proc2 = TimeSeriesPreprocessor.Prepocessor(); proc2.artificial_time_points = artificial_time_points2
       # S = proc1.create_summary_trends(X1[::-1], Y1[::-1])
       # T = proc2.create_summary_trends(X2[::-1], Y2[::-1])
        bwd_DP = orgalign.DP5(S_rev,T_rev, free_params= state_params, backward_run=True, zero_transition_costs=zero_transition_costs)#,mean_batch_effect=self.mean_batch_effect) 
        bwd_opt_cost = bwd_DP.run_optimal_alignment() 
        temp_path = bwd_DP.backtrack() 
        landscapeObj = orgalign.AlignmentLandscape(fwd_DP, bwd_DP, len(S.mean_trend), len(T.mean_trend), alignment_path, the_5_state_machine = True)
        landscapeObj.collate() #landscapeObj.plot_alignment_landscape()
        return AligmentObj(gene, S,T,fwd_DP, bwd_DP, landscapeObj)
            
    def align_all_pairs(self):
        
        with Pool(self.n_threads) as p:
            results = list(tqdm_notebook(p.imap(self.align_single_pair, self.gene_list), total=len(self.gene_list)))
        self.results = results 
        
        self.results_map = {}
        for a in self.results:
            self.results_map[a.gene] = a
            

    def align_all_pairs_no_thread_version(self):
        
        self.results = [] 
        for gene in self.gene_list:
            self.results.append(self.align_single_pair(gene))
        
        self.results_map = {}
        for a in self.results:
            self.results_map[a.gene] = a  
   
        
    def cluster_all_alignments(self, n_clusters=None, possible_dist_threshold=None, linkage_method='complete', scheme=0):
            
        # compute the pairwise alignment distance matrix 
        if(not hasattr(self, 'DistMat' )):
            print('computing the Distance matrix')
            DistMat = AlignmentDistMan.AlignmentDist(self).compute_alignment_ensemble_distance_matrix(scheme=scheme)
            #c = sb.clustermap(DistMat,figsize=(10,30)) 
            self.DistMat = DistMat
        if(n_clusters!=None):
            gene_clusters, cluster_ids = self.cluster_alignments_v1(n_clusters=n_clusters, linkage_method= linkage_method)
        else:
            gene_clusters, cluster_ids = self.cluster_alignments_v2(linkage_method, possible_dist_threshold=possible_dist_threshold)
        self.gene_clusters = gene_clusters
        self.cluster_ids = cluster_ids
        
    
    def cluster_alignments_v1(self, n_clusters, linkage_method):
        
        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage=linkage_method) 
        x = cluster.fit_predict(self.DistMat)
        gene_clusters = orgalign.Utils().check_alignment_clusters(n_clusters, x,
                                                                self.results, n_cols=5, figsize=(10,10)) 
        return gene_clusters, x   
    
    def cluster_alignments_v2(self, linkage_method, possible_dist_threshold = None):

        X = squareform(self.DistMat)
        #print(X)
        Z = linkage(X, linkage_method)
        if(possible_dist_threshold==None):
            possible_dist_threshold = np.quantile(squareform(self.DistMat),0.25)
        x = fcluster(Z, possible_dist_threshold , criterion='distance') # cluster ids
        n_clusters = len(np.unique(x))
        gene_clusters = orgalign.Utils().check_alignment_clusters(n_clusters, x,
                                                                self.results, n_cols=5, figsize=(10,10)) 
        x = x-1 # to make cluster ids 0-indexed
        return gene_clusters, x   
    
 
    
    def show_cluster(self, cluster_id):

        for i in range(len(self.cluster_ids)):
            if(self.cluster_ids[i]==cluster_id):
                print('Gene: ', self.results[i].gene)
                print(self.results[i].al_visual)
                self.results[i].plotTimeSeries(self, plot_cells=True) 
                plt.show() 
                print('----------------------------------------------')
                
      
    def show_cluster_alignment_strings(self,cluster_id):
        for i in range(len(self.cluster_ids)):
            if(self.cluster_ids[i]==cluster_id):
                print(self.results[i].alignment_str)
                self.results[i].cluster_id = cluster_id
                
    def get_cluster_alignment_objects(self, cluster_id):
        cluster_al_objects = []
        for i in range(len(self.cluster_ids)):
            if(self.cluster_ids[i]==cluster_id):
                #print(self.results[i].alignment_str)
                self.results[i].cluster_id = cluster_id
                cluster_al_objects.append(self.results[i])
        return cluster_al_objects

    def show_cluster_plots(self, cluster_id, show_alignment = False):

        temp = np.unique(self.cluster_ids == cluster_id, return_counts=True)[1][1]
        n_cols = 4
        n_rows = int(np.ceil(temp/n_cols))
        fig,axs = plt.subplots(n_rows,n_cols,figsize=(20,n_rows*3))

        k = 1
        for i in range(len(self.cluster_ids)):
            if(self.cluster_ids[i]==cluster_id):
                plt.subplot(n_rows, n_cols, k )
                if(show_alignment):
                    self.results[i].plotTimeSeriesAlignment()
                else:
                    self.results[i].plotTimeSeries(self, plot_cells=True, plot_mean_trend=True) 
                plt.title(self.results[i].gene)
                k = k+1
        fig.tight_layout()
        n = n_cols * n_rows
        i = 1
        while(k<=n):
            axs.flat[-1*i].set_visible(False) 
            k = k+1
            i=i+1         

    def show_ordered_alignments(self):
        
        return AlignmentDistMan.AlignmentDist(self).order_genes_by_alignments()
                
                
    def show_pairwise_distance_matrix(self, al_obj): # pairwise log compression matrix 
        
        # check compression of each matched pair
        temp_mat = al_obj.fwd_DP.DP_util_matrix
        compression_dist_mat = [] 
        for i in range(1,temp_mat.shape[0]):
            row = []
            for j in range(1,temp_mat.shape[1]):
                x = np.abs(temp_mat[i,j][2])
                row.append(float(x))
            compression_dist_mat.append(row)   

        x = pd.DataFrame(np.log10(np.asarray(compression_dist_mat) ))
        min_x =  np.nanmin(np.asarray(x).flatten())
        x = x.fillna(min_x) 
        sb.heatmap(x, cmap='jet')  
        
        
    # separately get correlation coefficient of ref and query mean trends along the trajectory by 
    # first doing distributional interpolation with number of time bins and then take sliding window to compute 
    # pearson correlation coefficient 
    def get_correlation_coefficient_trend(self, gene, SLIDING_WINDOW = 10, n_bins = 50):
        
        # correlation coefficient trend over sliding window of 10 bins
        rp = TimeSeriesPreprocessor.Prepocessor(self.ref_mat, self.ref_time, n_bins)
        qp =  TimeSeriesPreprocessor.Prepocessor(self.query_mat, self.query_time, n_bins)
        S = rp.prepare_interpolated_gene_expression_series(gene,WEIGHT_BY_CELL_DENSITY = self.WEIGHT_BY_CELL_DENSITY)
        T = qp.prepare_interpolated_gene_expression_series(gene,WEIGHT_BY_CELL_DENSITY = self.WEIGHT_BY_CELL_DENSITY)
        Y1 = S.Y; Y2 = T.Y
        X1 = S.X; X2 = T.X
        bin_times = np.unique(X1) 
        correlation_coefficients = []
        for i in range(len(bin_times)):
            if(i+SLIDING_WINDOW>=len(bin_times)):
                break
            s = []
            t = []
            for k in range(SLIDING_WINDOW):
                s.append(S.mean_trend[i+k])
                t.append(T.mean_trend[i+k])
            cc = stats.pearsonr(s,t)[0]
            #print('Pearson correlation: ', stats.pearsonr(s,t)[0])
            correlation_coefficients.append(cc)
        return correlation_coefficients
    
    def get_correlation_coefficient_trend_for_all_genes(self):

        cc = []
        for gene in tqdm_notebook(self.gene_list):
            pcc = self.get_correlation_coefficient_trend(gene, SLIDING_WINDOW=10)
            cc.append(pcc)
        df = pd.DataFrame(cc) 
        df.index = self.gene_list
        return df
    
    
    def get_match_stat_for_all_genes(self):
        m_p = []
        m_ps = []
        m_pt = []
        for a in self.results:
            m_p.append(a.get_series_match_percentage()[0])
            m_ps.append(a.get_series_match_percentage()[1])
            m_pt.append(a.get_series_match_percentage()[2])
            
        df = pd.DataFrame([m_p,m_ps,m_pt,self.cluster_ids]).transpose() 
        df.columns = ['match %', 'match % S', 'match % T', 'cluster_id']
        return df 
    
    
    #interpolated gene expression heat matrix 
    def __prepare_intpl_df(self,intpl_df, intpl_time):
        intpl_df = pd.DataFrame(intpl_df)
        intpl_df = intpl_df.transpose()
        intpl_df['time'] = intpl_time
        intpl_df = intpl_df.sort_values(by='time')
        intpl_df = intpl_df.iloc[:,intpl_df.columns!='time']
        intpl_df.columns = self.gene_list
        df_zscore = intpl_df.apply(zscore)
        return df_zscore

    def __get_zscore_expr_mat(self,expr_mat, expr_time):
        expr_mat['time'] = expr_time
        expr_mat = expr_mat.sort_values(by='time')
        df = expr_mat
        df  = df.iloc[:,df.columns!='time']
        df_zscore = df.apply(zscore)
        #df_zscore  = df_zscore.iloc[:,df_zscore.columns!='time']
        return df_zscore

    # z normalised for plotting purposes
    def __save_interpolated_and_noninterpolated_mats(self):
        ref_intpl_df = [] 
        query_intpl_df = [] 
        for gene in self.gene_list:
            ref_intpl_df.append(self.pairs[gene][0].Y)
            query_intpl_df.append(self.pairs[gene][1].Y) 
        self.ref_intpl_df = self.__prepare_intpl_df(ref_intpl_df, self.pairs[gene][0].X)
        self.query_intpl_df = self.__prepare_intpl_df(query_intpl_df, self.pairs[gene][1].X)
        self.ref_expr_df = self.__get_zscore_expr_mat(self.ref_mat[self.gene_list], self.ref_time )
        self.query_expr_df = self.__get_zscore_expr_mat(self.query_mat[self.gene_list], self.query_time )
        #return ref_intpl_df, query_intpl_df, ref_expr_df, query_expr_df 

    def __plot_comparative_heatmap(self, ref_df, query_df, cluster_id = None):
        
        if(cluster_id!=None):
            ref_df = ref_df[self.gene_clusters[cluster_id]]
            query_df = query_df[self.gene_clusters[cluster_id]]
            fig, axs = plt.subplots(1,2, figsize=(10,ref_df.shape[1]*0.5))
        else:
            fig, axs = plt.subplots(1,2, figsize=(10,10))

       # plt.subplot(1,2,1)
        sb.clustermap(ref_df.transpose(), xticklabels=False, vmin=-2, vmax=2, cbar=False,cmap = 'YlGnBu', col_cluster=False)#, ax=axs[0])
       # plt.subplot(1,2,2)
        sb.clustermap(query_df.transpose(), xticklabels=False, vmin=-2, vmax=2,cmap = 'YlGnBu',col_cluster=False)#,ax=axs[1])
        fig.tight_layout()
    
    def prepare_interpolated_non_interpolated_mats(self):
        self.__save_interpolated_and_noninterpolated_mats() 
    
    def plot_comparative_heatmap_intpl(self, cluster_id = None):
        if(not hasattr(self, 'ref_intpl_df' )):
            self.__save_interpolated_and_noninterpolated_mats() 
        self.__plot_comparative_heatmap(self.ref_intpl_df, self.query_intpl_df,cluster_id=cluster_id)
    
    def plot_comparative_heatmap_expr(self, cluster_id = None):
        if(not hasattr(self, 'ref_expr_df' )):
            self.__save_interpolated_and_noninterpolated_mats() 
        self.__plot_comparative_heatmap(self.ref_expr_df, self.query_expr_df,cluster_id=cluster_id)
    
    
    def run_MVG_alignment(self, mvg_genes,MVG_MODE_KL=True):
        
        D_ref = []
        D_query = [] 
        i = 0
        for gene in mvg_genes:
            S = self.pairs[gene][0]
            T = self.pairs[gene][1]

            if(i==0):
                for bin_id in range(len(S.data_bins)):
                    D_ref.append(pd.DataFrame(S.data_bins[bin_id]))
            else:
                for bin_id in range(len(S.data_bins)):
                    D_ref[bin_id] = pd.concat([D_ref[bin_id], pd.Series(S.data_bins[bin_id]) ], axis=1) 

            if(i==0):
                for bin_id in range(len(T.data_bins)):
                    D_query.append(pd.DataFrame(T.data_bins[bin_id]))
            else:
                for bin_id in range(len(T.data_bins)):
                    D_query[bin_id] = pd.concat([D_query[bin_id], pd.Series(T.data_bins[bin_id]) ], axis=1) 

            if(i==0):
                S_time = S.X  # no need to do this at every iteration since it is the same artificial time points for all genes
                T_time = T.X

            i=i+1

        S = TimeSeriesPreprocessor.SummaryTimeSeriesMVG(S_time, D_ref)
        T = TimeSeriesPreprocessor.SummaryTimeSeriesMVG(T_time, D_query)
        state_params=[0.99,0.5,0.4] #[0.95,0.5,0.4]
        fwd_DP = orgalign.DP5(S,T,  free_params = state_params, backward_run=False,  zero_transition_costs= False, prohibit_case = False, MVG_MODE_KL = MVG_MODE_KL)#,mean_batch_effect=self.mean_batch_effect) 
        fwd_opt_cost = fwd_DP.run_optimal_alignment() 
        alignment_path = fwd_DP.backtrack() 
        fwd_DP.alignment_path = alignment_path
        landscapeObj = orgalign.AlignmentLandscape(fwd_DP, None,len(S.data_bins), len(T.data_bins), alignment_path, the_5_state_machine = True)
        landscapeObj.collate_fwd() #landscapeObj.plot_alignment_landscape()
        #return fwd_DP.alignment_str, fwdlandscapeObj
        return AligmentObj(str(mvg_genes), S,T,fwd_DP,None, landscapeObj)
        
    
    
    def compute_cluster_MVG_alignments(self,MVG_MODE_KL=True, RECOMPUTE=False):
        
        if((not hasattr(self, 'mvg_cluster_average_alignments')) or RECOMPUTE):
            print('run MVG alignment')
            self.mvg_cluster_average_alignments = []

            for cluster_id in tqdm_notebook(range(len(self.gene_clusters))):
                group = self.gene_clusters[cluster_id]
                if(len(group)>1):
                    al_obj = self.run_MVG_alignment(group,MVG_MODE_KL=MVG_MODE_KL)
                    self.mvg_cluster_average_alignments.append(al_obj)
                else: # don't run MVG because there is only one gene in this cluster
                    self.mvg_cluster_average_alignments.append(self.get_cluster_alignment_objects(cluster_id)[0])
                         
        return             

        n_col = 5; n_row = int(np.ceil(len(self.mvg_cluster_average_alignments)/n_col)) 
        fig,axs =plt.subplots(n_row,n_col,figsize=(20,n_row*3))
        i=1
        for a in self.mvg_cluster_average_alignments:
           # plt.subplot(4,5,i)
           # plot_alignment_landscape(a.landscape_obj,i)
            plt.subplot(n_row,n_col,i)
            ax = sb.heatmap(a.landscape_obj.L_matrix, square=True,  cmap="jet")
            path_x = [p[0]+0.5 for p in a.landscape_obj.alignment_path]
            path_y = [p[1]+0.5 for p in a.landscape_obj.alignment_path]
            ax.plot(path_y, path_x, color='black', linewidth=3, alpha=0.5, linestyle='dashed') # path plot
            plt.xlabel("S",fontweight='bold')
            plt.ylabel("T",fontweight='bold')    
            i=i+1
            
            
    def plot_mvg_alignment(self, cluster_id):
            
        mvg_path = None
        if(len(self.gene_clusters[cluster_id])>1):
            mvg_path = self.mvg_cluster_average_alignments[cluster_id].landscape_obj.alignment_path
            avg_alignment, path = self.get_cluster_average_alignments(cluster_id)
        else: 
            path = self.get_cluster_alignment_objects(cluster_id)[0].landscape_obj.alignment_path
            mvg_path = path
            avg_alignment = self.get_cluster_alignment_objects(cluster_id)[0].alignment_str
        self.__plot_avg_alignment_landscape_in_cluster(cluster_id, path, mvg_path)


    def __plot_avg_alignment_landscape_in_cluster(self,cluster_id, path, mvg_path=None): 

            avg_DP_M_matrix = None 
            avg_DP_W_matrix = None 
            avg_DP_V_matrix = None 
            avg_DP_D_matrix = None 
            avg_DP_I_matrix = None 

            cluster_al_objects = self.get_cluster_alignment_objects(cluster_id) 
            for a in cluster_al_objects:
                if(avg_DP_M_matrix is None):
                    avg_DP_M_matrix = a.fwd_DP.DP_M_matrix
                    avg_DP_W_matrix = a.fwd_DP.DP_W_matrix
                    avg_DP_V_matrix = a.fwd_DP.DP_V_matrix
                    avg_DP_D_matrix = a.fwd_DP.DP_D_matrix
                    avg_DP_I_matrix = a.fwd_DP.DP_I_matrix
                else:
                    avg_DP_M_matrix = avg_DP_M_matrix + a.fwd_DP.DP_M_matrix
                    avg_DP_W_matrix = avg_DP_W_matrix + a.fwd_DP.DP_W_matrix
                    avg_DP_V_matrix = avg_DP_V_matrix + a.fwd_DP.DP_V_matrix
                    avg_DP_D_matrix = avg_DP_D_matrix + a.fwd_DP.DP_D_matrix
                    avg_DP_I_matrix = avg_DP_I_matrix + a.fwd_DP.DP_I_matrix

            avg_DP_M_matrix = avg_DP_M_matrix/len(cluster_al_objects)
            avg_DP_W_matrix = avg_DP_W_matrix/len(cluster_al_objects)
            avg_DP_V_matrix = avg_DP_V_matrix/len(cluster_al_objects)
            avg_DP_D_matrix = avg_DP_D_matrix/len(cluster_al_objects)
            avg_DP_I_matrix = avg_DP_I_matrix/len(cluster_al_objects)

            L_matrix = []
            T_len = self.results[0].fwd_DP.T_len
            S_len = self.results[0].fwd_DP.S_len
            for i in range(T_len+1):
                L_matrix.append(np.repeat(0.0,S_len+1))
            L_matrix = np.matrix(L_matrix) 

            if(mvg_path != None):
                paths = [path, mvg_path]
            else:
                paths = [path]
            for a in cluster_al_objects:
                paths.append(a.landscape_obj.alignment_path)

            for i in range(0,T_len+1):
                    for j in range(0,S_len+1):
                        _i = T_len-i
                        _j = S_len-j
                        temp = [ avg_DP_M_matrix[i,j],avg_DP_W_matrix[i,j] ,avg_DP_V_matrix[i,j], avg_DP_D_matrix[i,j], avg_DP_I_matrix[i,j]]
                        L_matrix[i,j] = min(temp)

            mat = L_matrix
            fig, ax = plt.subplots(1,1, figsize=(5,5))
            sb.heatmap(mat, square=True,  cmap='jet', ax=ax, cbar=False,xticklabels=False,yticklabels=False)
            path_color = "black"
            alpha = 2.0; linewidth = 4
            i=0
            for path in paths: 
                path_x = [p[0]+0.5 for p in path]
                path_y = [p[1]+0.5 for p in path]
                ax.plot(path_y, path_x, color=path_color, linewidth=linewidth, alpha=alpha, linestyle='dashed') # path plot
                if((i>=1)):
                    alpha = 0.5
                    linewidth = 1
                    path_color = 'black'
                else:
                    path_color = 'brown'
                i=i+1

            plt.xlabel("S",fontweight='bold')
            plt.ylabel("T",fontweight='bold')
            
    def get_cluster_average_alignments(self, cluster_id, deterministic=True):
        
            cluster_alobjs = self.get_cluster_alignment_objects(cluster_id)
            i = self.results[0].fwd_DP.T_len
            j = self.results[0].fwd_DP.S_len
            
            avg_alignment = ''
            tracked_path = []
            tracked_path.append([i,j])
            
            while(True):
                if(i==0 and j==0):
                    break
                backtrack_states_probs = {}
                backtrack_states_probs['M'] = 0 
                backtrack_states_probs['W'] = 0 
                backtrack_states_probs['V'] = 0 
                backtrack_states_probs['D'] = 0 
                backtrack_states_probs['I'] = 0 
                for a in cluster_alobjs:
                    backtract_state = a.landscape_obj.L_matrix_states[i,j]
                    if(backtract_state=='0'):
                        backtrack_states_probs['M']+=1 
                    elif(backtract_state=='1'):
                        backtrack_states_probs['W']+=1 
                    elif(backtract_state=='2'):
                        backtrack_states_probs['V']+=1 
                    elif(backtract_state=='3'):
                        backtrack_states_probs['D']+=1 
                    elif(backtract_state=='4'):
                        backtrack_states_probs['I']+=1 
                for state in backtrack_states_probs.keys(): 
                    backtrack_states_probs[state] = backtrack_states_probs[state]/len(cluster_alobjs) 

                if(deterministic):
                    cs = np.argmax(np.asarray(list(backtrack_states_probs.values())) )
                else:
                    cs = MyFunctions.sample_state(np.asarray(list(backtrack_states_probs.values()) ) )
                if(cs==0):
                    i = i-1
                    j = j-1
                    avg_alignment = 'M' + avg_alignment 
                elif(cs==1 or cs==3):
                    j= j-1
                    if(cs==1):
                        avg_alignment = 'W' + avg_alignment
                    else:
                        avg_alignment = 'D' + avg_alignment
                elif(cs==2 or cs==4):
                    i=i-1
                    if(cs==2):
                        avg_alignment = 'V' + avg_alignment
                    else:
                        avg_alignment = 'I' + avg_alignment
            
                tracked_path.append([i,j])
            
            return avg_alignment, tracked_path
        
    
class DEAnalyser:
    
    def __init__(self, al_obj):
        self.al_obj = al_obj
        self.alignment_str = al_obj.alignment_str
        self.al_visual = None

    def _util_1(self, a):
        ind = ""
        i = 0
        for c in range(len(a)):
            if(a[c]=='-'):
                ind = ind + ' '
            elif(a[c]=='^'):
                ind = ind + str(i-1)
            elif(a[c] == '*'):
                if(i<10):
                    ind = ind + str(i)
                    i=i+1
                else:
                    i=0
                    ind = ind + str(i)
                    i=i+1
        return ind

    def resolve(self, regions):
        for i in range(len(regions)):
            x = list(regions[i]); x[1] = x[1]-1; regions[i] = x
        return regions

    def get_matched_regions(self):
        
        if(self.al_visual == None):
           # print('retrieving matched regions')
            D_regions = [(m.start(0), m.end(0)) for m in regex.finditer("D+", self.alignment_str)]
            I_regions = [(m.start(0), m.end(0)) for m in regex.finditer("I+", self.alignment_str)]
            M_regions = [(m.start(0), m.end(0)) for m in regex.finditer("M+", self.alignment_str)] 
            W_regions = [(m.start(0), m.end(0)) for m in regex.finditer("W+", self.alignment_str)]
            V_regions = [(m.start(0), m.end(0)) for m in regex.finditer("V+", self.alignment_str)]

            M_regions = self.resolve(M_regions); D_regions = self.resolve(D_regions); 
            I_regions = self.resolve(I_regions)
            W_regions = self.resolve(W_regions); V_regions =self. resolve(V_regions)
            i = 0; j = 0; m_id = 0; i_id = 0; d_id = 0; v_id = 0; w_id = 0; c = 0
            S_match_regions = []; T_match_regions = []
            S_non_match_regions = []; T_non_match_regions = []
            a1 = ""; a2 = ""

            abstract_regions = []
            colored_string=''
            
            while(c<len(self.alignment_str)):
                #print(self.alignment_str[c])
                if(self.alignment_str[c]=='M'):
                    #print('match region')
                    step = (M_regions[m_id][1] - M_regions[m_id][0] + 1)
                    S_match_regions.append([j,j+step-1]); T_match_regions.append([i,i+step-1])
                    i = i + step; j = j + step; m_id = m_id + 1
                    a1 = a1 + hcolors.MATCH; a2 = a2 + hcolors.MATCH;
                    a1 = a1 + "*"*(step); a2 = a2 + "*"*(step)
                    a1 = a1 + hcolors.STOP; a2 = a2 + hcolors.STOP
                    abstract_regions.append('M')
                    colored_string += (hcolors.MATCH + "M"*(step) + hcolors.STOP)
                    
                    # process W,V separately 
                if(self.alignment_str[c]=='V'):
                    #print('warp V region')
                    step = (V_regions[v_id][1] - V_regions[v_id][0] + 1)
                    T_match_regions.append([i,i+step-1])
                    i = i + step; v_id = v_id + 1
                    a2 = a2 + hcolors.MATCH;
                    a1 = a1 + "^"*(step); a2 = a2 + "*"*(step)
                    a2 = a2 + hcolors.STOP
                    abstract_regions.append('V')
                    
                    colored_string += (hcolors.MATCH + "V"*(step) + hcolors.STOP)
                    
                if(self.alignment_str[c]=='W'):
                    #print('warp W region')
                    step = (W_regions[w_id][1] - W_regions[w_id][0] + 1)
                    S_match_regions.append([j,j+step-1])
                    j = j + step; w_id = w_id + 1
                    a1 = a1 + hcolors.MATCH;
                    a1 = a1 + "*"*(step); a2 = a2 + "^"*(step)
                    a1 = a1 + hcolors.STOP;
                    abstract_regions.append('W')
                    
                    colored_string += (hcolors.MATCH + "W"*(step) + hcolors.STOP)
                    
                if(self.alignment_str[c]=='I'):
                    #print('insert region')
                    step = (I_regions[i_id][1] - I_regions[i_id][0] + 1)
                    T_non_match_regions.append([i,i+step-1])
                    i = i + step; i_id = i_id + 1
                    a2 = a2+ hcolors.INSERT
                    a1 = a1 + "-"*(step); a2 = a2 + "*"*(step)
                    a2 = a2 + hcolors.STOP
                    
                    colored_string += (hcolors.INSERT + "I"*(step) + hcolors.STOP)
                    #abstract_regions.append('I')
                if(self.alignment_str[c]=='D'):
                    #print('delete region')
                    step = (D_regions[d_id][1] - D_regions[d_id][0] + 1)
                    S_non_match_regions.append([j,j+step-1])
                    j = j + step; d_id = d_id + 1
                    a1 = a1 + hcolors.DELETE
                    a1 = a1 + "*"*(step); a2 = a2 + "-"*(step)
                    a1 = a1 + hcolors.STOP;
                    
                    colored_string += (hcolors.DELETE + "D"*(step) + hcolors.STOP)
                    #abstract_regions.append('D')
                c = c + step 
                #print(step)
            #print(abstract_regions)
            # ----- to get index line along the alignment -----
            i=0; index_line = ""
            k = 0
            while(k<len(self.alignment_str)):
                if(i<10):
                    index_line = index_line + str(i)
                    i=i+1
                else:
                    i=0
                    index_line = index_line + str(i)
                    i=i+1
                k=k+1
            # ----- to get index line along the alignment -----

            index_line_S =  self._util_1(a1) + ' Reference index'
            index_line_T =  self._util_1(a2) + ' Query index'
            
            self.al_visual = index_line_S + '\n' + a1 + '\n' + a2 + '\n' + index_line_T
            self.S_match_regions = S_match_regions
            self.T_match_regions = T_match_regions
            self.S_non_match_regions = S_non_match_regions
            self.T_non_match_regions = T_non_match_regions 
            self.abstract_regions = abstract_regions 
            self.index_line = index_line
            self.al_obj.colored_alignment_str = colored_string
            
        self.al_obj.al_visual = self.index_line + ' Alignment index \n'  + self.al_visual  + '\n'+ self.alignment_str + ' 5-state string '
            
  
        # returns each 1-1 matching of time bins matched through M,W,V
    def get_matched_time_points(self):
        #print('alignment string: ', self.alignment_str)
        j = 0
        i = 0
        FLAG = False
        matched_points_S = [] 
        matched_points_T = [] 
        prev_c = ''
        for c in self.alignment_str:
            if(c=='M'):
                #print(c,i,j)
                if(prev_c=='W'):
                    i=i+1
                if(prev_c=='V'):
                    j=j+1
           #     if(FLAG):
           #         if(prev_c=='I'):
           #             j=j+1
           #         if(prev_c=='D'):
           #             i=i+1
           #         FLAG=False
                matched_points_T.append(i)
                matched_points_S.append(j)
                i=i+1
                j=j+1
            elif(c=='W'):
                #print(prev_c,i,j)
                if(prev_c not in ['W','V']):
                    i=i-1
                if(prev_c=='V'):
                    i=i-1
                    j=j+1
                if(prev_c=='D' and not FLAG):
                    FLAG = True
                matched_points_T.append(i)
                matched_points_S.append(j)
                j=j+1
            elif(c=='V'):
                if(prev_c not in ['W','V']):
                    j=j-1
                if(prev_c=='W'):
                    j=j-1
                    i=i+1
                if(prev_c=='I' and not FLAG):
                    FLAG = True
                matched_points_T.append(i)
                matched_points_S.append(j)
                i=i+1
            elif(c=='I'):
                if(prev_c=='W'):
                    i=i+1
                if(prev_c=='V'):
                    j=j+1
                i=i+1
            elif(c=='D'):
                if(prev_c=='W'):
                    i=i+1
                if(prev_c=='V'):
                    j=j+1
                j=j+1
            prev_c = c
        assert(len(matched_points_S) == len(matched_points_T))  
 
        self.match_points_S = np.array(matched_points_S)
        self.match_points_T = np.array(matched_points_T)        
        self.l2fold_changes = [] 
        self.l2fold_changes_in_matches = []

        for i in range(len(self.match_points_S)):
            
            S_bin = self.al_obj.S.data_bins[self.match_points_S[i]] 
            T_bin = self.al_obj.T.data_bins[self.match_points_T[i]] 
            S_bin_mean = np.mean(S_bin)
            T_bin_mean = np.mean(T_bin )
            self.l2fold_changes.append([np.log2(S_bin_mean/T_bin_mean),self.match_points_S[i],self.match_points_T[i]])
             # converting time bin ids to interpolated pseudotime values
            self.l2fold_changes_in_matches.append([np.log2(S_bin_mean/T_bin_mean), self.al_obj.fwd_DP.S.time_points[self.match_points_S[i]],
                                                 self.al_obj.fwd_DP.T.time_points[self.match_points_T[i]] ])
            
            #print('**** ', self.match_points_S[i], self.match_points_T[i], np.log2(S_bin_mean/T_bin_mean))
               
    
    # Sanity checker for non-significant DE in matched regions
    def get_DE_info_for_matched_regions(self):
        
        s = self.match_points_S
        t = self.match_points_T

        matched_S_time = []
        matched_T_time = []
        compression_statistic = []
        l2fc = []
        wilcox_p = []
        ks2_p = []
        ttest_p = []

        for i in range(len(s)):
            l2fc.append(self.l2fold_changes_in_matches[i][0])
            matched_S_time.append(self.l2fold_changes_in_matches[i][1]) 
            matched_T_time.append(self.l2fold_changes_in_matches[i][2]) 
            S_bin = self.al_obj.S.data_bins[s[i]] 
            T_bin = self.al_obj.T.data_bins[t[i]] 
            
            if(not np.any(self.al_obj.S.data_bins[s[i]] - self.al_obj.S.data_bins[t[i]] )):
                wilcox_p.append(0.0)
                ks2_p.append(0.0) 
                ttest_p.append(0.0)
            else:
                wilcox = stats.wilcoxon(self.al_obj.S.data_bins[s[i]], self.al_obj.T.data_bins[t[i]])
                ks2 =  stats.ks_2samp(self.al_obj.S.data_bins[s[i]], self.al_obj.T.data_bins[t[i]])
                ttest = stats.ttest_ind(self.al_obj.S.data_bins[s[i]], self.al_obj.T.data_bins[t[i]])

                wilcox_p.append(np.round(wilcox[1],6))
                ks2_p.append(np.round(ks2[1],6)) 
                ttest_p.append(np.round(ttest[1],6))

            delta = np.round(self.al_obj.fwd_DP.DP_util_matrix[t[i]+1,s[i]+1][2],7)
            compression_statistic.append(delta)

        df = pd.DataFrame([s,t,matched_S_time, matched_T_time, compression_statistic, l2fc, wilcox_p, ks2_p,ttest_p]).transpose()
        df.columns = [ 'ref_bin','query_bin','ref_pseudotime','query_pseudotime','Delta','l2fc', 'wilcox','ks2','ttest'] 
                  
        self.al_obj.matched_region_DE_info = df 
        self.al_obj.match_points_S = s
        self.al_obj.match_points_T = t

    
    
class hcolors:
    MATCH = '\033[92m'
    INSERT = '\033[91m'
    DELETE = '\033[91m'
    STOP = '\033[0m'    
    
    #### test cases for 1-1 match point retrieval:
       # al_str = 'MMMVVVVVVWWWWWW'
       # al_str = 'MMMWWWWWWVVVVVV'
       # al_str = 'DDDIIIVVVIIIMMM'
       # al_str = 'IIIDDDWWWIIIMMM'
       # al_str  = 'MMMIIIDDWWDDDIIVVDDMM'
    


    
    
    
    
    
    
    
        