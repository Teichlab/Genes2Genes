import regex 
import copy
import scipy
import anndata
import scipy.sparse
import pandas as pd
import numpy as np
import seaborn as sb
from tqdm import tqdm
import multiprocessing
import scipy.stats as stats
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm.notebook import tqdm_notebook

from . import OrgAlign as orgalign
from . import MyFunctions 
from . import TimeSeriesPreprocessor
from . import AlignmentDistMan
from . import VisualUtils
from . import ClusterUtils
from . import Utils

__version__ = 'v0.2.0'

class hcolors:
    MATCH = '\033[92m'
    INSERT = '\033[91m'
    DELETE = '\033[91m'
    STOP = '\033[0m'    

    
class AligmentObj:
    
    """
    This class defines an aligned object of a reference and query gene expression time series,
    to carry all related results. 
    """
    
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
        try:
            self.run_DEAnalyser() 
        except Exception as e:
            print(str(e),gene)

    # Printing details about the optimal alignment object
    def print(self):
        print('Fwd opt cost', self.fwd_DP.opt_cost)
        print('5-state alignment string: ', self.fwd_DP.alignment_str) 
        print('Ref matched time points ranges: ', self.match_regions_S)
        print('Query matched time points ranges: ', self.match_regions_T)
        print('Ref mismatched time points ranges: ',self.non_match_regions_S)
        print('Query mismatched time points ranges: ',self.non_match_regions_T)
        print('Alignment landscape plot: ')
        self.landscape_obj.plot_alignment_landscape() 
        
    def plotTimeSeries(self, refQueryAlignerObj, plot_cells = False, plot_mean_trend= False):
        sb.scatterplot(x=self.S.X, y=self.S.Y, color = 'forestgreen' ,alpha=0.05)#, label='Ref') 
        sb.scatterplot(x=self.T.X, y=self.T.Y, color = 'midnightblue' ,alpha=0.05)#, label ='Query')
      # plt.legend(loc='upper left')
        if(plot_cells):
            sb.scatterplot(x=refQueryAlignerObj.ref_time, y=np.asarray(refQueryAlignerObj.ref_mat[self.gene]), color = 'forestgreen' ) 
            sb.scatterplot(x=refQueryAlignerObj.query_time, y=np.asarray(refQueryAlignerObj.query_mat[self.gene]), color = 'midnightblue' )
        plt.title(self.gene)
        plt.xlabel('Pseudotime')
        plt.ylabel('Gene expression')
        
        if(plot_mean_trend):
            self.plot_mean_trends() 
            
    def plotTimeSeriesAlignment(self):  
        sb.scatterplot(x=self.S.X, y=self.S.Y, color = 'forestgreen' ,alpha=0.05)#, label='Ref') 
        sb.scatterplot(x=self.T.X, y=self.T.Y, color = 'midnightblue' ,alpha=0.05)#, label ='Query') 
      #  plt.legend(loc='upper left')
        self.plot_mean_trends() 
        plt.title(self.gene)
        plt.xlabel('Pseudotime')
        plt.ylabel('Gene expression')
        
        for i in range(self.matched_region_DE_info.shape[0]):
            S_timebin = int(self.matched_region_DE_info.iloc[i]['ref_bin'])
            T_timebin = int(self.matched_region_DE_info.iloc[i]['query_bin']) 
            x_vals = [self.matched_region_DE_info.iloc[i]['ref_pseudotime'],self.matched_region_DE_info.iloc[i]['query_pseudotime']] 
            y_vals = [self.S.mean_trend[S_timebin ], self.T.mean_trend[T_timebin]] 
            plt.plot(x_vals, y_vals, color='black', linestyle='dashed', linewidth=0.6)
    
    
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
    
    """
    This class defines the main aligner class of genes2genes alignment, acting as entry point to initialise alignment parameters and interpolation. 
    It contains all methods for running genes2genes alignment between the specified genes of the reference and query datasets.  
    
    Parameters
    ----------
    *args
        adata_ref: anndata 
        adata_query: anndata
        gene_list: list
        n_interpolation_points: int
        adaptive_kernel: boolean
    """
    
    def __init__(self, *args):
        print('===============================================================================================================')
        print('Genes2Genes ('+ __version__ +')')
        print('Dynamic programming alignment of gene pseudotime trajectories using a bayesian information-theoretic framework')
        print('===============================================================================================================')
        
        if(len(args) == 4 ):
            self.run_init1(args[0], args[1], args[2], args[3])
            adaptive_kernel = False
        elif(len(args) == 5 ):
            print('Running in adaptive interpolation mode')
            self.run_init1(args[0], args[1], args[2], args[3])
            adaptive_kernel = args[4]
        else:
            print('pls pass the required number of args')
            
        k=1;
        self.TrajInt_R = TimeSeriesPreprocessor.TrajectoryInterpolator(self.adata_ref, n_bins=self.n_artificial_time_points, adaptive_kernel=adaptive_kernel,raising_degree = k)
        self.TrajInt_R.run() 
        self.TrajInt_Q = TimeSeriesPreprocessor.TrajectoryInterpolator(self.adata_query, n_bins=self.n_artificial_time_points, adaptive_kernel=adaptive_kernel,raising_degree = k)
        self.TrajInt_Q.run() 
        print('Interpolator initialization completed')
        self.state_params = [0.99,0.1,0.7] # parameters empirically found over our simulated dataset
        self.no_extreme_cases =False
        
        print('Aligner initialised to align trajectories of', self.adata_ref.shape[0], 'reference cells &',self.adata_query.shape[0], 'query cells in terms of', len(self.gene_list), 'genes')
    
    # converts ref and query anndata objects to pd.DataFrames 
    def run_init1(self, adata_ref, adata_query, gene_list, n_artificial_time_points):
        
        self.adata_ref = adata_ref[:, gene_list] 
        self.adata_query = adata_query[:, gene_list]
        
        if(isinstance(adata_ref.X, scipy.sparse.csr.csr_matrix) 
           or isinstance(adata_ref.X,anndata._core.views.SparseCSCView)
           or isinstance(adata_ref.X,scipy.sparse.csc.csc_matrix)):
            ref_mat = pd.DataFrame(adata_ref.X.todense()) 
        else:
            ref_mat = pd.DataFrame(adata_ref.X) 
        if(isinstance(adata_query.X, scipy.sparse.csr.csr_matrix) 
           or isinstance(adata_query.X,anndata._core.views.SparseCSCView)
           or isinstance(adata_query.X,scipy.sparse.csc.csc_matrix)):
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
    
    def run_init2(self, ref_mat, ref_time, query_mat, query_time, gene_list, n_artificial_time_points):
        self.ref_mat = ref_mat
        self.query_mat = query_mat
        self.ref_time = ref_time
        self.query_time = query_time
        self.gene_list = gene_list
        self.pairs = {}
        self.n_artificial_time_points = n_artificial_time_points
    
    # util functions 
    def extract_significant_regions_only(self, regions):
        if(len(regions)==0): 
            return []
        adjacent_region_start = regions[0][0]
        filtered_regions = np.asarray([], dtype=np.float64)
        adjacent_region_indices = np.asarray([], dtype=np.float64)
        filtered_region_indices = np.asarray([], dtype=np.float64)
        for k in range(len(regions)):
            if(k!=len(regions)-1):
                if(regions[k][1] != regions[k+1][0]):
                    ended_adjacent_region_len = regions[k][1]- adjacent_region_start
                    if(ended_adjacent_region_len>0.2):
                        adjacent_region_indices = np.append(adjacent_region_indices,regions[k][0])
                        adjacent_region_indices = np.append(adjacent_region_indices, regions[k][1])
                        filtered_regions=np.append(filtered_regions,[adjacent_region_start,regions[k][1] ])
                        filtered_region_indices = np.append(filtered_region_indices,adjacent_region_indices)
                    adjacent_region_start = regions[k+1][0]
                    adjacent_region_indices = []
                else:
                    adjacent_region_indices=np.append(adjacent_region_indices,regions[k][0])
                    continue
            else:
                if(len(adjacent_region_indices)>0): # check if there is a continuing adjacent region
                    ended_adjacent_region_len = regions[k][1]- adjacent_region_start
                    if(ended_adjacent_region_len>0.2):
                        adjacent_region_indices = np.append(adjacent_region_indices,regions[k][0])
                        adjacent_region_indices=np.append(adjacent_region_indices,regions[k][1])
                        filtered_regions=np.append(filtered_regions,[adjacent_region_start,regions[k][1] ])
                        filtered_region_indices = np.append(filtered_region_indices, adjacent_region_indices)
                        
        return list(filtered_region_indices)

    def check_inconsistent_zero_region(self, gex_arr, pseudotime_arr, trajInterpolator):
        
        regions = []
        window_range = trajInterpolator.interpolation_points
        
        for i in range(1,len(window_range)):
            sliding_region = np.logical_and(pseudotime_arr>=window_range[i-1], pseudotime_arr<window_range[i]) 
            n_pos = np.count_nonzero(gex_arr[sliding_region])
            if(n_pos<=3): #almost 0 expression 
                regions.append([window_range[i-1], window_range[i]])
        regions = self.extract_significant_regions_only(regions)
        return regions
 
        
    def run_interpolation(self, gene_idx):

        gex_r = Utils.csr_mat_col_densify(self.TrajInt_R.mat , gene_idx)
        gex_q = Utils.csr_mat_col_densify(self.TrajInt_Q.mat , gene_idx)
        USER_GIVEN_STD = np.repeat(-1.0, self.TrajInt_R.n_bins)
        
        if(self.no_extreme_cases):
            
            
            if((not gex_r.any()) and (not gex_q.any()) ):# both are 0 expressed
                #print('both', g)    # complete match     
                S = ref_processor.prepare_interpolated_gene_expression_series(gene, WEIGHT_BY_CELL_DENSITY = self.WEIGHT_BY_CELL_DENSITY)
                T = query_processor.prepare_interpolated_gene_expression_series(gene, WEIGHT_BY_CELL_DENSITY = self.WEIGHT_BY_CELL_DENSITY)
            elif( (not gex_r.any()) or (np.count_nonzero(gex_r)<=50) ):# only ref is 0 expressed or only small number of cells <=50
                #print('ref 0 expressed')
                # get query mean trend and std trend estimated 
                T = query_processor.prepare_interpolated_gene_expression_series(gene, WEIGHT_BY_CELL_DENSITY = self.WEIGHT_BY_CELL_DENSITY,
                                                                             ESTIMATE=True, user_given_std=[])
                # assigning query std trend to ref
                S = ref_processor.prepare_interpolated_gene_expression_series(gene, WEIGHT_BY_CELL_DENSITY = self.WEIGHT_BY_CELL_DENSITY,
                                                                            ESTIMATE = False, user_given_std = T.intpl_stds)
            elif( (not gex_q.any()) or (np.count_nonzero(gex_q)<=50)  ): # only query is 0 expressed or only small number of cells <=50
                #print('query 0 expressed')
                # get ref mean trend and std trend estimated 
                S = ref_processor.prepare_interpolated_gene_expression_series(gene, WEIGHT_BY_CELL_DENSITY = self.WEIGHT_BY_CELL_DENSITY,
                                                                               ESTIMATE=True, user_given_std=[])
                # assigning ref std trend to query
                T = query_processor.prepare_interpolated_gene_expression_series(gene, WEIGHT_BY_CELL_DENSITY = self.WEIGHT_BY_CELL_DENSITY,
                                                                            ESTIMATE = False, user_given_std = S.intpl_stds)
            else: # both are expressed
                #print('both expressed')
                S = ref_processor.prepare_interpolated_gene_expression_series(gene, WEIGHT_BY_CELL_DENSITY = self.WEIGHT_BY_CELL_DENSITY)
                T = query_processor.prepare_interpolated_gene_expression_series(gene, WEIGHT_BY_CELL_DENSITY = self.WEIGHT_BY_CELL_DENSITY)
                # get both ref and query mean trend and std trend estimated 

            return [S,T]

        if(np.count_nonzero(gex_r)<=3 and np.count_nonzero(gex_q)<=3): # if both are almost 0 (less than 3 counts overall)

            USER_GIVEN_STD = np.repeat(0.01, self.TrajInt_R.n_bins) # use a very low constant std for both cases 
            S = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_R, user_given_std = USER_GIVEN_STD)
            T = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_Q, user_given_std = USER_GIVEN_STD)  

        elif(np.count_nonzero(gex_r)<=3):  # if only ref is almost 0 expressed

            T_temp = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_Q, user_given_std = USER_GIVEN_STD)
            regions = self.check_inconsistent_zero_region(gex_q, self.TrajInt_Q.cell_pseudotimes, self.TrajInt_Q)
            common_std = min(T_temp.intpl_stds)/10
            if(len(regions)!=0):
                USER_GIVEN_STD[np.in1d(self.TrajInt_Q.interpolation_points, regions)] = common_std 
                T = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_Q, user_given_std = USER_GIVEN_STD)  
            else:
                T = T_temp
            USER_GIVEN_STD = np.repeat(common_std, self.TrajInt_R.n_bins)
            S = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_R, user_given_std = USER_GIVEN_STD)

        elif(np.count_nonzero(gex_q)<=3): # if only query is almost 0 expressed

            S_temp = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_R, user_given_std = USER_GIVEN_STD)
            regions = self.check_inconsistent_zero_region(gex_r, self.TrajInt_R.cell_pseudotimes, self.TrajInt_R)
            common_std = min(S_temp.intpl_stds)/10
            if(len(regions)!=0):
                USER_GIVEN_STD[np.in1d(self.TrajInt_R.interpolation_points, regions)] = common_std 
                S = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_R, user_given_std = USER_GIVEN_STD)  
            else:
                S = S_temp
            USER_GIVEN_STD = np.repeat(common_std, self.TrajInt_Q.n_bins)
            T = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_Q, user_given_std = USER_GIVEN_STD)

        else:
            S_temp = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_R, user_given_std = USER_GIVEN_STD)
            T_temp = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_Q, user_given_std = USER_GIVEN_STD)
            
            regions_S = self.check_inconsistent_zero_region(gex_r, self.TrajInt_R.cell_pseudotimes, self.TrajInt_R)
            regions_T = self.check_inconsistent_zero_region(gex_q, self.TrajInt_Q.cell_pseudotimes, self.TrajInt_Q)

            if(len(regions_S) == 0 and len(regions_T)==0):
                S = S_temp; T = T_temp 

            elif(len(regions_S) != 0 and len(regions_T)!=0):
                
                common_std = min(min(S_temp.intpl_stds), min(T_temp.intpl_stds))/10
                USER_GIVEN_STD[np.in1d(self.TrajInt_R.interpolation_points, regions_S)] = common_std 
                
                S = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_R, user_given_std = USER_GIVEN_STD)
                USER_GIVEN_STD = np.repeat(-1.0, self.TrajInt_R.n_bins)
                USER_GIVEN_STD[np.in1d(self.TrajInt_Q.interpolation_points, regions_T)] = common_std
                T = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_Q, user_given_std = USER_GIVEN_STD)  

            elif(len(regions_S) != 0):         
                common_std = min(min(S_temp.intpl_stds), min(T_temp.intpl_stds))/10
                USER_GIVEN_STD[np.in1d(self.TrajInt_R.interpolation_points, regions_S)] = common_std 
                S = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_R, user_given_std = USER_GIVEN_STD)
                T = T_temp

            elif(len(regions_T) != 0):
                common_std = min(min(S_temp.intpl_stds), min(T_temp.intpl_stds))/10
                USER_GIVEN_STD[np.in1d(self.TrajInt_Q.interpolation_points, regions_T)] = common_std 
                T = TimeSeriesPreprocessor.interpolate_gene_v2(gene_idx, self.TrajInt_Q, user_given_std = USER_GIVEN_STD)  
                S = S_temp
                
        del gex_r
        del gex_q
        return [S,T]
    
    
    def align_single_pair(self, gene_idx):
        
        gene = self.gene_list[gene_idx]
        
        if(not(gene in self.pairs.keys())):
            self.pairs[gene] = self.run_interpolation(gene_idx)
        
        S = self.pairs[gene][0]
        T = self.pairs[gene][1]
        
        fwd_DP = orgalign.DP5(S,T,  free_params = self.state_params, backward_run=False,  zero_transition_costs= False, prohibit_case = False) 
        fwd_opt_cost = fwd_DP.run_optimal_alignment() 
        alignment_path = fwd_DP.backtrack() 
        fwd_DP.alignment_path = alignment_path

        landscapeObj = orgalign.AlignmentLandscape(fwd_DP, None, len(S.mean_trend), len(T.mean_trend), alignment_path, the_5_state_machine = True)
        landscapeObj.collate_fwd() #landscapeObj.plot_alignment_landscape()

        a = AligmentObj(gene, S,T,fwd_DP,None, landscapeObj)
        return a
    
    def align_all_pairs(self, concurrent=False, n_processes = None):
        
        print('Running gene-level alignment: \U0001F9EC')
        
        if(concurrent):
            if(n_processes is None):
                n_processes = multiprocessing.cpu_count()
                print('concurrent mode, running with',n_processes, 'processes')
            with Pool(n_processes) as p:
                results = list(tqdm(p.imap(self.align_single_pair, range(0,len(self.gene_list))), total=len(self.gene_list))) 
            p.close(); p.terminate(); p.join(); 
            self.results = results 
            self.results_map = {}
            for a in self.results:
                self.results_map[a.gene] = a
        else:
            self.results_map = {}
            self.results = []
            for gene_idx in tqdm(range(len(self.gene_list))):
                a = self.align_single_pair(gene_idx)
                self.results.append(a)
                self.results_map[a.gene] = a
                
        print('Alignment completed! \u2705')
        
    def get_stat_df(self):

        opt_alignment_costs = [] 
        match_percentages =[] 
        l2fc = []
        for g in self.gene_list:
            opt_alignment_costs.append(self.results_map[g].fwd_DP.opt_cost)
            match_percentages.append(self.results_map[g].match_percentage/100)
            rgex = np.asarray(self.ref_mat.loc[:,g])
            qgex = np.asarray(self.query_mat.loc[:,g])        
            l2fc.append(np.log2(np.mean(rgex)/np.mean(qgex))) 

        df = pd.DataFrame([self.gene_list, match_percentages, opt_alignment_costs, l2fc]).transpose()
        df.columns = ['Gene','alignment_similarity_percentage', 'opt_alignment_cost','l2fc']
        df.set_index('Gene')
        df['color'] = np.repeat('green',df.shape[0])
        df.loc[df['alignment_similarity_percentage']<=0.5,'color'] = 'red'
        df['abs_l2fc'] = np.abs(df['l2fc']) 
        df = df.sort_values(['alignment_similarity_percentage','abs_l2fc'],ascending=[True, False])
        
        plt.subplots(1,2,figsize=(10,4))
        plt.subplot(1,2,1)
        print('Mean alignment similarity percentage (matched %): ')
        print(round(np.mean(df['alignment_similarity_percentage']),4)*100,'%' )
        temp = np.asarray([x*100 for x in df['alignment_similarity_percentage']]) 
        p = sb.kdeplot(temp, fill=True,label='Alignment Similarity %')
        plt.xlabel('Alignment Similarity Percentage')
        plt.xlim([0,100])
        p.set_yticklabels(p.get_yticks(), size = 12)
        p.set_xticklabels(p.get_xticks(), size = 12)
        plt.xlabel('Alignment similarity percentage', fontsize='12')
        plt.ylabel('Density', fontsize='12')

        plt.subplot(1,2,2)
        VisualUtils.plot_alignmentSim_vs_optCost(df)
        
        return df
        
        
    def get_aggregate_alignment(self):
        average_alignment, alignment_path =  ClusterUtils.get_cluster_average_alignments(self, self.gene_list)
        mat = ClusterUtils.get_pairwise_match_count_mat(self, self.gene_list )
        print('Average Alignment: ', VisualUtils.color_al_str(average_alignment), '(cell-level)')
        print( '% similarity:', np.round(len(regex.findall("M|W|V",average_alignment))*100/len(average_alignment),2) )
        self.average_alignment = average_alignment
        VisualUtils.plot_alignment_path_on_given_matrix(paths = [alignment_path], mat=mat)
        
    def get_aggregate_alignment_for_subset(self, gene_subset):
        average_alignment, alignment_path =  ClusterUtils.get_cluster_average_alignments(self, gene_subset)
        mat = ClusterUtils.get_pairwise_match_count_mat(self, gene_subset )
        print('Average Alignment: ', VisualUtils.color_al_str(average_alignment), '(cell-level)')
        VisualUtils.plot_alignment_path_on_given_matrix(paths = [alignment_path], mat=mat)
   
    def show_cluster_alignment_strings(self,cluster_id):
        for i in range(len(self.cluster_ids)):
            if(self.cluster_ids[i]==cluster_id):
                print(VisualUtils.color_al_str(self.results[i].alignment_str))
                self.results[i].cluster_id = cluster_id
                
    def get_cluster_alignment_objects(self, cluster_id):
        cluster_al_objects = []
        for i in range(len(self.cluster_ids)):
            if(self.cluster_ids[i]==cluster_id):
                #print(self.results[i].alignment_str)
                self.results[i].cluster_id = cluster_id
                cluster_al_objects.append(self.results[i])
        return cluster_al_objects 

    def show_ordered_alignments(self):
        
        print('In the order of the first match occurrence along pseudotime')
        
        for a in self.results:
            a.gene_pair = a.gene
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
    


class DEAnalyser:
    
    """
    This class defines complementary functions for alignment results analysis. 
    """
    
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
            
            self.al_obj.S.data_bins[s[i]] = np.asarray(self.al_obj.S.data_bins[s[i]])
            self.al_obj.S.data_bins[t[i]] = np.asarray(self.al_obj.S.data_bins[t[i]])
            
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

    

    
    
    
    
