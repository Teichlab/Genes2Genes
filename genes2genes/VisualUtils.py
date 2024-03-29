import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import anndata
import numpy as np
from adjustText import adjust_text
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import zscore
import colorcet as cc
from optbinning import ContinuousOptimalBinning
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.patches as mpatches
import regex

from . import Main


vega_20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728',
            '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2',
            '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',]

class VisualUtils():
    
    def __init__(self, adata_ref, adata_query, cell_type_colname, S_len, T_len, titleS = 'Reference', titleT = 'Query', mode='comp', write_file=False, optimal_binning=True, PLOT=True):
        self.write_file = write_file 
        if(mode=='comp'):
            self.titleS = titleS
            self.titleT = titleT
            
            n_points = S_len
            while(True):
                # later to replace with a better optimal binning that gives exact number we request
                print('# trying max n points for optimal binning =', n_points)
                adata_ref, bm1 = self.pseudotime2bin_celltypes(adata_ref, n_points, optimal_binning=optimal_binning)
                adata_query, bm2 = self.pseudotime2bin_celltypes(adata_query, n_points, optimal_binning=optimal_binning)
                
                if(not (len(bm1) == len(bm2))):
                    n_points=n_points-1               
                    if(n_points<=5):
                        print('Consider equal length binning')
                        break
                else:
                    print('====================================================')
                    print('Optimal equal number of bins for R and Q = ',len(bm1))
                    break
                  
            if(PLOT):
                plt.subplots(1,2, figsize=(10,3))
                x = list(adata_ref.obs.time)
                plt.subplot(1,2,1)
                sb.kdeplot(list(adata_ref.obs.time), color='ForestGreen' , fill=True)
                for s in bm1: 
                    plt.axvline(x=s, color='ForestGreen')
                x = list(adata_query.obs.time)
                plt.subplot(1,2,2)
                sb.kdeplot(list(adata_query.obs.time),color='midnightblue', fill=True)
                for s in bm2: 
                    plt.axvline(x=s, color='midnightblue')
                
            meta1 = self.plot_cell_type_proportions(adata_ref, cell_type_colname, 'bin_ids',None,'tab20')
            meta2 = self.plot_cell_type_proportions(adata_query, cell_type_colname, 'bin_ids',None,'tab20')
            if(not optimal_binning):
                meta1 = self.simple_interpolate(meta1,S_len)
                meta2 = self.simple_interpolate(meta2,T_len)
          #  meta1.loc[1] = meta1.loc[0] + meta1.loc[1]
          #  meta2.loc[1] = meta2.loc[0] + meta2.loc[1]
          #  meta1.loc[0] = np.repeat(0.0,len(np.unique(adata_ref.obs[cell_type_colname])) )
          #  meta2.loc[0] = np.repeat(0.0,len(np.unique(adata_query.obs[cell_type_colname])))
            
            temp1 = pd.Series(np.repeat(0.0,len(np.unique(adata_ref.obs[cell_type_colname])) ))
            temp1.index = meta1.columns
            meta1 = pd.concat([pd.DataFrame(temp1).transpose(),meta1.loc[:]]).reset_index(drop=True)
            
            temp2 = pd.Series(np.repeat(0.0,len(np.unique(adata_query.obs[cell_type_colname])) ))
            temp2.index = meta2.columns
            meta2 = pd.concat([pd.DataFrame(temp2).transpose(),meta2.loc[:]]).reset_index(drop=True)
            
            self.metaS = meta1
            self.metaT = meta2
            
            self.optimal_bining_S = bm1
            self.optimal_bining_T = bm2
         

    def get_optimal_binning(self, time_var_arr, n_points):
        x = time_var_arr
        optb = ContinuousOptimalBinning(name='pseudotime', dtype="numerical", max_n_bins=n_points)
        # this pacakge uses mixed integer programming based optimization to determine an optimal binning
        #kde = gaussian_kde(x); #density_values = kde(x); #optb.fit(x, density_values)
        optb.fit(x, x)
        #sb.kdeplot(x, fill=True)
        #for s in optb.splits: 
        #    plt.axvline(x=s)
        #print(len(optb.splits))
        return optb.splits


    # annotates cells with their respective bins based on interpolated pseudotime points
    def pseudotime2bin_celltypes(self, adata, n_points, optimal_binning = True):

            adata.obs['bin_ids'] = np.repeat(-1,adata.shape[0])
            if(optimal_binning):
                bin_margins = self.get_optimal_binning(np.asarray(adata.obs.time) , n_points)
            else:
                #bin_margins =  np.linspace(0,1,n_points+1)
                bin_margins =  np.linspace(0,1,n_points)#[1:-1]
            #print('computed the margins for ' + str(len(bin_margins)) +  ' bins')
            #print(bin_margins)
            bin_ids = []
            k = 0 
            for i in range(len(bin_margins)):
                
                if(i==0):
                    logic = np.logical_and(adata.obs.time >= 0, adata.obs.time < bin_margins[i+1])
                    #print('i==0', adata[logic].shape[0])
                elif(i==len(bin_margins)-1):
                    logic =  np.logical_and(adata.obs.time >= bin_margins[i], adata.obs.time <= 1.0)
                else:
                    logic =  np.logical_and(adata.obs.time >= bin_margins[i], adata.obs.time < bin_margins[i+1])
                adata.obs['bin_ids'][logic] = i 
            return adata, bin_margins

    # for plotting or getting celltype freq counts per bin
    def plot_cell_type_proportions(self, adata, cell_type_colname, covariate_colname, sorter, color_scheme_name="Spectral", plot=False):
        meta = pd.DataFrame(np.vstack((adata.obs[cell_type_colname],adata.obs[covariate_colname])).transpose(),columns=[cell_type_colname,covariate_colname])
        
        meta['COUNTER'] = 1
        meta = meta.groupby([covariate_colname,cell_type_colname])['COUNTER'].sum().unstack()
        meta = meta.fillna(0)
        #meta = meta.transpose()
        #meta = meta.sort_values(by=covariate_colname, key=sorter)
        if(plot):
            p = meta.apply(lambda x: x*100/sum(x), axis=1).plot(kind='bar',stacked=True, color=sb.color_palette(color_scheme_name, 20), grid = False)
            #p.legend(labels = ['not infected','infected'], loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
            p.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        return meta
    
    
    def simple_interpolate(self,meta, n_points):
        for i in range(n_points):
            #print(k)
            if(i not in meta.index):
                k=i
                while(k not in meta.index):
                    k = k-1
                _temp = meta.loc[k].copy()
                _temp.name = i
                meta = meta.append(_temp)
        meta = meta.sort_index()
        return meta
    
    
    def get_celltype_composition_across_time(adata_ref, adata_query, n_points, ANNOTATION_COLNAME, optimal_binning=True
                                        , order_S_legend=None, order_T_legend=None, PLOT=True, ref_cmap = None, query_cmap =None, plot_celltype_counts=False):
        
        a = sb.color_palette(cc.glasbey_hv, n_colors=3)
        vega_20 = [
            '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728',
            '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2',
            '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
        ]
        
        if(ref_cmap==None):
            ref_cmap = vega_20
        if(query_cmap==None):
            query_cmap = vega_20
        
        vs = VisualUtils(adata_ref, adata_query, cell_type_colname = ANNOTATION_COLNAME, 
                        S_len=n_points, T_len=n_points, titleS='Reference', titleT='Query',
                        write_file=False, optimal_binning=optimal_binning, PLOT=PLOT)
        
        if(plot_celltype_counts):
        
            ax = vs.metaS.apply(lambda x: x, axis=1).plot(kind='bar',stacked=True,color=ref_cmap, grid = False, legend=True, width=0.7,align='edge',figsize=(10,3))
            ax.legend(bbox_to_anchor=(1.1, 1.44))
            if(order_S_legend is not None):
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=[handles[idx] for idx in order_S_legend],labels=[labels[idx] for idx in order_S_legend],bbox_to_anchor=(1.0, 1.0))

            ax = vs.metaT.apply(lambda x: x, axis=1).plot(kind='bar',stacked=True,color=query_cmap, grid = False, legend=True, width=0.7,align='edge',figsize=(10,3))
            ax.legend(bbox_to_anchor=(1.1, 1.05))
            if(order_T_legend is not None):
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=[handles[idx] for idx in order_T_legend],labels=[labels[idx] for idx in order_T_legend],bbox_to_anchor=(1.0, 1.0))
            
        vs.metaS.apply(lambda x: x*100/sum(x), axis=1).plot(kind='bar',stacked=True,color=ref_cmap, grid = False, legend=False, width=0.7,align='edge',figsize=(10,1))
        plt.axis('off')
        vs.metaT.apply(lambda x: x*100/sum(x), axis=1).plot(kind='bar',stacked=True,color=query_cmap, grid = False, legend=False, width=0.7,align='edge',figsize=(10,1))
        plt.axis('off')

        return vs 
    
    
    def visualize_gene_alignment(self, alignment, cmap=None):

            if(isinstance(alignment,Main.AligmentObj )):
                alignment = alignment.alignment_str

            matched_points_S, matched_points_T = self.get_matched_time_points(alignment)

            fig = plt.figure(figsize=(4,2))
            heights = [1, 1, 1] 
            gs = plt.GridSpec(3, 1, height_ratios=heights)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0],sharex=ax1)
            ax3 = fig.add_subplot(gs[2, 0],sharex=ax1)
            
            if(cmap is None):
                cmap = vega_20

            plt.subplot(3,1,1)
            self.metaS.apply(lambda x: x*100/sum(x), axis=1).plot(kind='bar',stacked=True,color=cmap, grid = False, legend=False, width=0.7, ax=ax1)
            self.metaT.apply(lambda x: x*100/sum(x), axis=1).plot(kind='bar',stacked=True,color=cmap, grid = False, legend=False, width=0.7,ax=ax3)
            plt.subplot(3,1,2)
            for i in range(len(matched_points_S)):
                        S_timebin = matched_points_S[i]
                        T_timebin = matched_points_T[i]
                        x_vals = [T_timebin+1, S_timebin+1]
                        y_vals = [0,1]
                        plt.plot(x_vals, y_vals, marker='.', color='black', linewidth=0.5)

            def set_grid_off(ax):
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xticks([])
                ax.xaxis.set_ticks_position('none') 
                ax.set_yticks([])
                ax.figure.tight_layout()
                ax.grid(False)

            set_grid_off(ax1); set_grid_off(ax2); set_grid_off(ax3); 
            ax1.set_ylabel('Ref', rotation=0)
            ax3.set_ylabel('Query',rotation=0)
            fig.text(0.5, -0.05, 'Pseudotime bins with cell type composition', ha='center')
            ax1.set_title('Alignment w.r.t cell type compositions')

    
    
    def plot_comprehensive_alignment_landscape_plot(self, aligner, gene = None, order_S_legend=None, order_T_legend=None, paths_to_display=None, cmap='viridis'):

        if(gene!=None):
            al_obj = aligner.results_map[gene]
            if(paths_to_display==None):
                al_obj.landscape_obj.alignment_path.append([0,0])
                paths_to_display=[al_obj.landscape_obj.alignment_path]
            match_points_S = np.unique(al_obj.match_points_S) + 1
            match_points_T = np.unique(al_obj.match_points_T) + 1
            landscape_mat = pd.DataFrame(al_obj.landscape_obj.L_matrix)
        else:
            al_str, path = self.compute_overall_alignment(aligner)
            match_points_S, match_points_T = self.get_matched_time_points(al_str)
            match_points_S = np.unique(match_points_S) + 1
            match_points_T = np.unique(match_points_T) + 1
            if(paths_to_display==None):
                paths_to_display=[path]
            landscape_mat = aligner.get_pairwise_match_count_mat()
                
        nS_points=len(aligner.results[0].S.time_points)
        nT_points=len(aligner.results[0].T.time_points)

        fig, ((ax3, ax1, cbar_ax), (dummy_ax1, ax2, dummy_ax2)) = plt.subplots(nrows=2, ncols=3, figsize=(9*2, 6*2), sharex='col', sharey='row',
                                                                               gridspec_kw={'height_ratios': [2,1], 'width_ratios': [0.5, 1, 0.5]})
        g = sb.heatmap(landscape_mat.transpose(), xticklabels=True, yticklabels=True, cmap=cmap, cbar_ax=cbar_ax,  ax=ax1, cbar=False)
        g.tick_params( labelsize=10, labelbottom = True, bottom=True, top = False)#, labeltop=True)
        ax1.set_xlabel('pseudotime')
        x_ticks = np.asarray(range(0,nS_points+1)) 
        y_ticks = np.asarray(range(0,nT_points+1)) 

        # first barplot (Reference) --- left horizontal barplot
        p= self.metaS.apply(lambda x: x*100/sum(x), axis=1).plot(kind='barh',stacked=True, title=self.titleS ,color=sb.color_palette('deep', 20), grid = False, ax=ax3,legend=False, width=0.7,align='edge')
        for patch in p.patches:
            if(patch.get_y() in match_points_S):
                p.annotate(str('M'), (100, patch.get_y() * 1.005)  )      
        handles, labels = ax3.get_legend_handles_labels()
        for spine in p.spines:
            p.spines[spine].set_visible(False)
        if(order_S_legend!=None):
            dummy_ax1.legend(handles=[handles[idx] for idx in order_S_legend],labels=[labels[idx] for idx in order_S_legend])
        else:
            dummy_ax1.legend(handles,labels)
        # second barplot (Query) --- bottom barplot
        p = self.metaT.apply(lambda x: x*100/sum(x), axis=1).plot(kind='bar',stacked=True, title=self.titleT, color=sb.color_palette('deep', 20), grid = False, ax=ax2, legend=False,width=0.7,align='edge')
        for patch in p.patches:
           # print(patch.get_height())
            if(patch.get_x() in match_points_T):
                p.annotate(str('M'), (patch.get_x() * 1.005, 100) )
        handles, labels = ax2.get_legend_handles_labels()
        for spine in p.spines:
            p.spines[spine].set_visible(False)
        if(order_T_legend!=None):
            dummy_ax2.legend(handles=[handles[idx] for idx in order_T_legend],labels=[labels[idx] for idx in order_T_legend],loc='upper left')
        else:
            dummy_ax2.legend(handles,labels, loc='upper left')
        dummy_ax1.axis('off')
        dummy_ax2.axis('off')
        cbar_ax.axis('off')
        if(paths_to_display!=None): # for max 2 paths
            styles = ['solid', 'dashed']; i = 0 
            for path in paths_to_display: 
                path_x = [p[0]+0.5 for p in path]
                path_y = [p[1]+0.5 for p in path]
                ax1.plot(path_x, path_y, color='black', linewidth=9, alpha=1.0, linestyle=styles[i]) # path plot
                i=i+1
        ax1.axis(ymin=0, ymax=nS_points+1, xmin=0, xmax=nT_points+1)
        plt.tight_layout()
       # plt.show()
        
        if(self.write_file):
            plt.savefig('comprehensive_alignment_landscape_plot.pdf',bbox_inches = 'tight')
        
    def plot_match_stat_across_all_alignments(self, aligner):
            
            nS_points = len(aligner.results[0].S.time_points)
            nT_points = len(aligner.results[0].T.time_points)
            S_line = np.repeat(0, nS_points+1)
            T_line = np.repeat(0, nT_points+1)

            for a in aligner.results:
                matchS = a.match_points_S+1
                matchT = a.match_points_T+1
                for i in range(len(matchS)):
                    S_line[matchS[i]] =  S_line[matchS[i]] + 1
                    T_line[matchT[i]] =  T_line[matchT[i]] + 1

            S_line = S_line/np.sum(S_line)*100
            T_line = T_line/np.sum(T_line)*100

            plt.subplots(2,2,figsize=(17,6))
            plt.subplot(2,2,1)
            sb.barplot(np.asarray(range(nS_points+1)) , np.cumsum(S_line), color='midnightblue') 
            plt.ylabel('cumulative match percentage')
            plt.subplot(2,2,3)
            sb.barplot(np.asarray(range(nT_points+1)) , np.cumsum(T_line), color='forestgreen') 
            plt.ylabel('cumulative match percentage')
            plt.xlabel('pseudotime bin')
            plt.subplot(2,2,2)
            sb.barplot(np.asarray(range(nS_points+1)) , S_line, color='midnightblue') 
            plt.ylabel('match percentage')
            plt.subplot(2,2,4)
            sb.barplot(np.asarray(range(nT_points+1)) , T_line, color='forestgreen') 
            plt.ylabel('match percentage')
            plt.xlabel('pseudotime bin')
          #  plt.show()
            
            if(self.write_file):
                plt.savefig('match_stat_plot_across_all_alignments.pdf',bbox_inches = 'tight')

#    def plot_alignment_path_on_given_matrix(mat, paths, cmap='viridis',annot=True):
#        fig,ax = plt.subplots(1,1, figsize=(7,7))
#        sb.heatmap(mat, square=True,  cmap='viridis', ax=ax, cbar=True, annot=annot,fmt='g')  
#        for path in paths: 
#            path_x = [p[0]+0.5 for p in path]
#            path_y = [p[1]+0.5 for p in path]
#            ax.plot(path_y, path_x, color='black', linewidth=6) # path plot
#        plt.xlabel("PAM (Reference)",fontweight='bold')
#        plt.ylabel("LPS (Query)",fontweight='bold')
#        ax.xaxis.tick_top() # x axis on top
#        ax.xaxis.set_label_position('top')

    def get_matched_time_points(self, alignment_str):
        j = 0; i = 0
        FLAG = False
        matched_points_S = [] 
        matched_points_T = [] 
        prev_c = ''
        for c in alignment_str:
            if(c=='M'):
                if(prev_c=='W'):
                    i=i+1
                if(prev_c=='V'):
                    j=j+1
                matched_points_T.append(i)
                matched_points_S.append(j)
                i=i+1
                j=j+1
            elif(c=='W'):
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
        return matched_points_S, matched_points_T
        
            # computes simple DP alignment (using match score = pairwise total match count frequency) across all gene-level alignments 
    # gap score is taken as penalising 8% of the total number of tested genes => so that it controls the matching based on the number of 
    # total matches (i.e. it controls the degree of significant matching) 
    def compute_overall_alignment(self, aligner, plot=False, GAP_SCORE = None):
                
                if(GAP_SCORE==None):
                    GAP_SCORE= -len(aligner.gene_list)*0.08

                mat = aligner.get_pairwise_match_count_mat()
                if(plot):
                    sb.heatmap(mat, cmap='viridis', square=True)

                # DP matrix initialisation 
                opt_cost_M = []
                for i in range(mat.shape[0]):
                    opt_cost_M.append(np.repeat(0.0, mat.shape[1]))
                opt_cost_M = np.matrix(opt_cost_M) 
                # backtracker matrix initialisation 
                tracker_M = []
                for i in range(mat.shape[0]):
                    tracker_M.append(np.repeat(0.0, mat.shape[1]))
                tracker_M = np.matrix(tracker_M) 
                for i in range(1,mat.shape[0]):
                    tracker_M[i,0] = 2
                for j in range(1,mat.shape[1]):
                    tracker_M[0,j] = 1

                # running DP
                for j in range(1,mat.shape[1]):
                    for i in range(1,mat.shape[0]):
                        m_dir = opt_cost_M[i-1,j-1] + mat.loc[i,j]
                        d_dir = opt_cost_M[i,j-1] +  GAP_SCORE
                        i_dir = opt_cost_M[i-1,j] +  GAP_SCORE
                       # w_dir = opt_cost_M[i,j-1] +  mat.loc[i,j]
                       # v_dir = opt_cost_M[i-1,j] +  mat.loc[i,j]

                        a = max([m_dir, d_dir, i_dir])  # ,w_dir, v_dir]) 
                        if(a==d_dir):
                            opt = d_dir
                            dir_tracker = 1
                        elif(a==i_dir):
                            opt =i_dir
                            dir_tracker = 2
                        elif(a==m_dir):
                            opt = m_dir
                            dir_tracker = 0
                       # elif(a==w_dir):
                      #      opt = w_dir
                       #     dir_tracker = 3
                       # elif(a==v_dir):
                       #     opt = v_dir
                       #     dir_tracker = 4
                        #if(i==1 and j==4):
                        #    print(a, opt_cost_M[i-1,j-1], mat.loc[i,j], opt_cost_M[i,j-1] ,opt_cost_M[i-1,j]  )

                        opt_cost_M[i,j] = opt
                        tracker_M[i,j] = dir_tracker     
              #  print(tracker_M)

                # backtracking
                i = mat.shape[0]-1
                j = mat.shape[1]-1
                alignment_str = ''
                tracked_path = []
                while(True):
                  #  print([i,j])
                    tracked_path.append([i,j])
                    if(tracker_M[i,j]==0):
                        alignment_str = 'M' + alignment_str
                        i = i-1
                        j = j-1
                    elif(tracker_M[i,j]==1):
                        alignment_str = 'D' + alignment_str
                        j = j-1
                    elif(tracker_M[i,j]==2):
                        alignment_str = 'I' + alignment_str
                        i = i-1 
                 #   elif(tracker_M[i,j]==3):
                 #       alignment_str = 'W' + alignment_str
                 #       j = j-1
                 #   elif(tracker_M[i,j]==4):
                 #       alignment_str = 'V' + alignment_str
                 #       i = i-1
                    if(i==0 and j==0) :
                        break
                tracked_path.append([0,0])
                return alignment_str, tracked_path#, opt_cost_M, tracker_M

            

            

def plot_heatmaps(mat_ref,mat_query,pathway_name, IGS, cluster=False):
    
    if(cluster):
        g=sb.clustermap(mat_ref, figsize=(0.4,0.4), col_cluster=False) 
        gene_order = g.dendrogram_row.reordered_ind
        df = pd.DataFrame(g.data2d) 
        df.index = IGS.SETS[pathway_name][gene_order]
    else:
        df=mat_ref
    
    plt.subplots(1,2,figsize=(8,12))
    max_val = np.max([np.max(mat_ref),np.max(mat_query)]) 
    min_val = np.min([np.min(mat_ref),np.min(mat_query)]) 
    plt.subplot(1,2,1)
    ax=sb.heatmap(df, vmax=max_val,vmin=min_val, cbar_kws = dict(use_gridspec=False,location="top")) 
    plt.title('Reference')
    ax.yaxis.set_label_position("left")
    for tick in ax.get_yticklabels():
        tick.set_rotation(360)
    plt.subplot(1,2,2)
    if(cluster):
        mat_query = mat_query.loc[IGS.SETS[pathway_name][gene_order]] 
    ax = sb.heatmap(mat_query,vmax=max_val,  vmin=min_val,cbar_kws = dict(use_gridspec=False,location="top"), yticklabels=False) 
    plt.title('Query')
    #plt.show()


# smoothened/interpolated mean trends + Z normalisation 
def plot_mean_trend_heatmaps(pathway_name, IGS, aligner, cluster=False):
    S_mat = []
    T_mat = []
    S_zmat = []
    T_zmat = []

    for gene in IGS.SETS[pathway_name]:

        fS = pd.DataFrame([aligner.results_map[gene].S.mean_trend, np.repeat('Ref', len(aligner.results_map[gene].S.mean_trend))]).transpose()
        fT = pd.DataFrame([aligner.results_map[gene].T.mean_trend, np.repeat('ATO', len(aligner.results_map[gene].T.mean_trend))]).transpose()
        f = pd.concat([fS,fT])
        f[0] = np.asarray(f[0], dtype=np.float64)
        from scipy.stats import zscore
        f['z_normalised'] = zscore(f[0])
        S_mat.append(np.asarray(f[f[1]=='Ref'][0]))
        T_mat.append(np.asarray(f[f[1]=='ATO'][0]))    
        S_zmat.append(np.asarray(f[f[1]=='Ref']['z_normalised']))
        T_zmat.append(np.asarray(f[f[1]=='ATO']['z_normalised']))  
    S_mat = pd.DataFrame(S_mat)
    T_mat = pd.DataFrame(T_mat)
    S_zmat = pd.DataFrame(S_zmat)
    T_zmat = pd.DataFrame(T_zmat)
    
    S_mat.index = IGS.SETS[pathway_name]
    T_mat.index = IGS.SETS[pathway_name]
    S_zmat.index = IGS.SETS[pathway_name]
    T_zmat.index = IGS.SETS[pathway_name]
    
    print('Interpolated mean trends')
    plot_heatmaps(S_mat, T_mat, pathway_name, IGS, cluster=cluster)
    print('Z-normalised Interpolated mean trends')
    return plot_heatmaps(S_zmat, T_zmat, pathway_name, IGS, cluster=cluster)


def plotTimeSeries(gene, aligner, plot_cells = False, plot_mean_trend= False):
    
        al_obj = aligner.results_map[gene]
        plt.subplots(1,3,figsize=(15,3))
        plt.subplot(1,3,1)
        plotTimeSeriesAlignment(gene, aligner) 
        plt.subplot(1,3,2)
        max_val = np.max([np.max(np.asarray(aligner.ref_mat[al_obj.gene])), np.max(np.asarray(aligner.query_mat[al_obj.gene]))])
        min_val = np.min([np.min(np.asarray(aligner.ref_mat[al_obj.gene])), np.min(np.asarray(aligner.query_mat[al_obj.gene]))])
        g = sb.scatterplot(x=aligner.query_time, y=np.asarray(aligner.query_mat[al_obj.gene]), alpha=0.7, color = 'midnightblue', legend=False,linewidth=0.3, s=20)  
        plt.title('Query')
        plt.ylim([min_val-0.5,max_val+0.5])
        plt.subplot(1,3,3)
        g = sb.scatterplot(x=aligner.ref_time, y=np.asarray(aligner.ref_mat[al_obj.gene]), color = 'forestgreen', alpha=0.7, legend=False,linewidth=0.3,s=20 ) 
        plt.title('Reference')
        plt.ylim([min_val-0.5,max_val+0.5])

def plotTimeSeriesAlignment(gene, aligner):  
    
        al_obj = aligner.results_map[gene]
        sb.scatterplot(x=al_obj.S.X, y=al_obj.S.Y, color = 'forestgreen' ,alpha=0.05, legend=False)#, label='Ref') 
        sb.scatterplot(x=al_obj.T.X, y=al_obj.T.Y, color = 'midnightblue' ,alpha=0.05, legend=False)#, label ='Query') 
        al_obj.plot_mean_trends() 
        plt.title(al_obj.gene)
        plt.xlabel('pseudotime')
        plt.ylabel('Gene expression')
        plt.axis('off')
        
        for i in range(al_obj.matched_region_DE_info.shape[0]):
            S_timebin = int(al_obj.matched_region_DE_info.iloc[i]['ref_bin'])
            T_timebin = int(al_obj.matched_region_DE_info.iloc[i]['query_bin']) 
            x_vals = [al_obj.matched_region_DE_info.iloc[i]['ref_pseudotime'],al_obj.matched_region_DE_info.iloc[i]['query_pseudotime']] 
            y_vals = [al_obj.S.mean_trend[S_timebin ], al_obj.T.mean_trend[T_timebin]] 
            plt.plot(x_vals, y_vals, color='black', linestyle='dashed', linewidth=1.5)


def plot_alignmentSim_vs_l2fc(x):
    ax=sb.scatterplot(x=x['l2fc'],y=x['alignment_similarity_percentage']*100,s=120, legend=False, hue =x['alignment_similarity_percentage'] ,
                   palette=sb.diverging_palette(0, 255, s=150, as_cmap=True),edgecolor='k',linewidth=0.3)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('Alignment Similarity %', fontsize=15, fontweight='bold')
    plt.xlabel('Log2 fold change of mean expression', fontsize = 15, fontweight='bold')
    plt.grid(False)
    plt.axhline(50, color='black')
    plt.axvline(0, color='black', linestyle='dashed')


def plot_alignmentSim_vs_optCost(x, opt_cost_cut=0):
    sb.scatterplot(x=x['opt_alignment_cost'],y=x['alignment_similarity_percentage']*100,s=120, legend=False, hue =x['alignment_similarity_percentage'] ,
                   palette=sb.diverging_palette(0, 255, s=150, as_cmap=True),edgecolor='k',linewidth=0.3)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('Alignment Similarity %', fontsize=15, fontweight='bold')
    plt.xlabel('Optimal alignment cost (nits)', fontsize = 15, fontweight='bold')
    plt.grid(False)
    plt.axhline(50, color='black')
    plt.axvline(opt_cost_cut, color='black', linestyle='dashed')
    plt.tight_layout()


def plot_alignment_path_on_given_matrix(mat, paths, cmap='viridis'):
        fig,ax = plt.subplots(1,1, figsize=(7,7))
        sb.heatmap(mat, square=True,  cmap='viridis', ax=ax, cbar=True)  
        for path in paths: 
            path_x = [p[0]+0.5 for p in path]
            path_y = [p[1]+0.5 for p in path]
            ax.plot(path_y, path_x, color='white', linewidth=6)
        plt.xlabel("Reference",fontweight='bold')
        plt.ylabel("Query",fontweight='bold')
        ax.xaxis.tick_top() # x axis on top
        ax.xaxis.set_label_position('top')

def plot_alignment_clustermap():
    
    p = sb.clustermap(aligner.DistMat,cmap='viridis', figsize=(10,10))
    p.ax_heatmap.set_xticklabels(p.ax_heatmap.get_xmajorticklabels(), fontsize = 12)
    p.ax_heatmap.set_yticklabels(p.ax_heatmap.get_ymajorticklabels(), fontsize = 12)
    p.ax_row_dendrogram.set_visible(False)


def plot_distmap_with_clusters(aligner, cmap=None, vmin = 0.0, vmax = 1.0, genes2highlight=None):
    
    godsnot_64 = [
    # "#000000",  # remove the black, as often, we have black colored annotation, 
        '#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161',
       '#fbafe4', '#949494', '#ece133', '#56b4e9', # <--added colorblind palette to this
    "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C"]
    
    # ordering genes by packing them into their clusters
    cluster_ordered_genes = []
    cluster_ids = []
    
    cluster_lens = []
    for i in aligner.gene_clusters.keys():
        cluster_lens.append(len(aligner.gene_clusters[i]))
    c_keys = np.asarray(list(aligner.gene_clusters.keys()) ) [np.argsort(cluster_lens)[::-1]] # ordered according to cluster size
    for i in c_keys:
        cluster_ordered_genes  += aligner.gene_clusters[i]
        cluster_ids += list(np.repeat(i,len(aligner.gene_clusters[i]))) 
    temp = pd.DataFrame([cluster_ordered_genes, cluster_ids]).transpose() 
    temp.columns = ['Gene','cluster_id']

    n_clusters = len(aligner.gene_clusters.keys())
    if(n_clusters<=20):
        color_list = list(sb.color_palette('colorblind'))[0:n_clusters] 
    else:
        if(cmap is not None):
            orig_cmap = plt.cm.get_cmap(cmap)
            custom_cmap = orig_cmap(np.linspace(vmin, vmax, n_clusters))
            color_list = [mcolors.rgb2hex(custom_cmap[i]) for i in range(n_clusters)]
        else:
            color_list = godsnot_64[0:n_clusters]
        #np.random.seed(3); np.random.shuffle(color_list)

    x = dict(zip(temp['cluster_id'].unique(), color_list )) 
    rcolors = pd.Series(temp['cluster_id']).map(x)
    rcolors.name = ''
    x = aligner.DistMat[cluster_ordered_genes].loc[cluster_ordered_genes]   
    p = sb.clustermap(x.reset_index(drop=True), cmap='viridis', 
                      square=True, row_cluster=False, col_cluster=False, row_colors=rcolors, figsize=(10,10), xticklabels=False,
                      cbar_pos=(1.05, 0.54, 0.02, 0.25))
    if(genes2highlight is None):
        gene_labels = []
        for tick_label in p.ax_heatmap.axes.get_yticklabels():
            tick_text = tick_label.get_text()
            gene = temp.Gene.loc[int(tick_text)]
            tick_label.set_color(rcolors[int(tick_text)])
            gene_labels.append(gene)
        p.ax_heatmap.axes.set_yticklabels(gene_labels, rotation = 0) 
    else:
        tick_indices = []
        for g in genes2highlight:
            tick_indices.append(temp.index[temp['Gene']==g][0]) 
        p.ax_heatmap.axes.set_yticks(tick_indices) 
        p.ax_heatmap.axes.set_yticklabels(genes2highlight, rotation = 0) 
        
        k=0
        for tick_label in p.ax_heatmap.axes.get_yticklabels():
            tick_label.set_color(rcolors[tick_indices[k]])
            k+=1

    # plotting the legend of clusters
    legend_labels = ['Cluster-'+str(k) for k in c_keys] 
    legend_patches = [mpatches.Patch(color=color_list[i], label=legend_labels[i]) for i in range(len(color_list))]
    ax = p.ax_row_dendrogram
    ax.legend(handles=legend_patches, loc='center')
    ax.axis('off'); ax.set_xticks([]); ax.set_yticks([]); 

    


def resolve(regions):
    for i in range(len(regions)):
        x = list(regions[i]); x[1] = x[1]-1; regions[i] = x
    return regions

def color_al_str(alignment_str):
        
        D_regions = [(m.start(0), m.end(0)) for m in regex.finditer("D+", alignment_str)]
        I_regions = [(m.start(0), m.end(0)) for m in regex.finditer("I+", alignment_str)]
        M_regions = [(m.start(0), m.end(0)) for m in regex.finditer("M+", alignment_str)] 
        W_regions = [(m.start(0), m.end(0)) for m in regex.finditer("W+", alignment_str)]
        V_regions = [(m.start(0), m.end(0)) for m in regex.finditer("V+", alignment_str)]
        M_regions = resolve(M_regions); D_regions = resolve(D_regions); 
        I_regions = resolve(I_regions)
        W_regions = resolve(W_regions); V_regions = resolve(V_regions)
        i = 0; j = 0; m_id = 0; i_id = 0; d_id = 0; v_id = 0; w_id = 0; c = 0
        colored_string=''
            
        while(c<len(alignment_str)):
            if(alignment_str[c]=='M'):
                step = (M_regions[m_id][1] - M_regions[m_id][0] + 1)
                i = i + step; j = j + step; m_id = m_id + 1
                colored_string += (Main.hcolors.MATCH + "M"*(step) + Main.hcolors.STOP)
                # process W,V separately 
            if(alignment_str[c]=='V'):
                step = (V_regions[v_id][1] - V_regions[v_id][0] + 1)
                i = i + step; v_id = v_id + 1
                colored_string += (Main.hcolors.MATCH + "V"*(step) + Main.hcolors.STOP)
            if(alignment_str[c]=='W'):
                step = (W_regions[w_id][1] - W_regions[w_id][0] + 1)
                j = j + step; w_id = w_id + 1
                colored_string += (Main.hcolors.MATCH + "W"*(step) + Main.hcolors.STOP)
            if(alignment_str[c]=='I'):
                step = (I_regions[i_id][1] - I_regions[i_id][0] + 1)
                i = i + step; i_id = i_id + 1
                colored_string += (Main.hcolors.INSERT + "I"*(step) + Main.hcolors.STOP)
            if(alignment_str[c]=='D'):
                step = (D_regions[d_id][1] - D_regions[d_id][0] + 1)
                j = j + step; d_id = d_id + 1
                colored_string += (Main.hcolors.DELETE + "D"*(step) + Main.hcolors.STOP)
            c = c + step 
            
        return colored_string


def plot_any_legend(text2color_map):
    legend_labels= list(text2color_map.keys())
    color_list = list(text2color_map.values())
    legend_patches = [mpatches.Patch(color=color_list[i], label=legend_labels[i]) for i in range(len(color_list))]
    fig, ax = plt.subplots()
    ax.legend(handles=legend_patches, loc='center')
    ax.axis('off'); ax.set_xticks([]); ax.set_yticks([]); 

def show_gene_alignment(gene, aligner, vs, cmap=None):
    vs.visualize_gene_alignment(aligner.results_map[gene].alignment_str, cmap=cmap)
    plotTimeSeries(gene, aligner, plot_cells=True)
    aligner.results_map[gene].alignment_str
    print(color_al_str(aligner.results_map[gene].alignment_str)) 
    print('Optimal alignment cost:', round(aligner.results_map[gene].fwd_DP.opt_cost,3),'nits')
    print('Alignment similarity percentage:', aligner.results_map[gene].match_percentage,'%' )
