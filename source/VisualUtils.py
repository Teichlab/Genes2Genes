import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import anndata
import numpy as np
    

class VisualUtils():
    
    def __init__(self, adata_ref, adata_query, cell_type_colname, S_len, T_len, titleS = 'Reference', titleT = 'Query', mode='comp', write_file=False):
        self.write_file = write_file 
        if(mode=='comp'):
            self.titleS = titleS
            self.titleT = titleT
            self.pseudotime2bin_celltypes(adata_ref,S_len)
            self.pseudotime2bin_celltypes(adata_query,T_len)
            meta1 = self.plot_cell_type_proportions(adata_ref, cell_type_colname, 'bin_ids',None,'tab20')
            meta2 = self.plot_cell_type_proportions(adata_query, cell_type_colname, 'bin_ids',None,'tab20')
            meta1 = self.simple_interpolate(meta1,S_len)
            meta2 = self.simple_interpolate(meta2,T_len)
            # to make bins compatible with number of artificial timepoints -- the 0th bin is taken to be empty and the first bin carries all the counts coming under both 0th and 1st bin.
            meta1.loc[1] = meta1.loc[0] + meta1.loc[1]
            meta2.loc[1] = meta2.loc[0] + meta2.loc[1]
            meta1.loc[0] = np.repeat(0.0,len(np.unique(adata_ref.obs[cell_type_colname])) )
            meta2.loc[0] = np.repeat(0.0,len(np.unique(adata_query.obs[cell_type_colname])))
            self.metaS = meta1
            self.metaT = meta2
        
    # annotates cells with their respective bins based on interpolated pseudotime points
    def pseudotime2bin_celltypes(self, adata, n_points):

        adata.obs['bin_ids'] = np.repeat(0,adata.shape[0])
        bin_margins =  np.linspace(0,1,n_points+2)
        bin_ids = []

        for i in range(len(bin_margins)-1):
            if(i==len(bin_margins)-1):
                break
            #print(bin_margins[i],i)
            if(i<len(bin_margins)-2):
                logic =  np.logical_and(adata.obs.time < bin_margins[i+1], adata.obs.time >= bin_margins[i])
            else:
                logic =  np.logical_and(adata.obs.time <= bin_margins[i+1], adata.obs.time >= bin_margins[i])
            bin_cells = adata[logic] 
            #print(bin_cells.shape)
            adata.obs['bin_ids'][logic] = i 

    # for plotting or getting celltype freq counts per bin
    def plot_cell_type_proportions(self, adata, cell_type_colname, covariate_colname, sorter, color_scheme_name="Spectral", plot=False):
        meta = pd.DataFrame(np.vstack((adata.obs[cell_type_colname],adata.obs[covariate_colname])).transpose(),columns=[cell_type_colname,covariate_colname])
        meta['COUNTER'] = 1
        meta = meta.groupby([covariate_colname,cell_type_colname])['COUNTER'].sum().unstack()
        meta = meta.fillna(0)
        meta = meta.sort_values(by=covariate_colname, key=sorter)
        if(plot):
            p = meta.apply(lambda x: x*100/sum(x), axis=1).plot(kind='bar',stacked=True, color=sb.color_palette(color_scheme_name, 20), grid = False)
            #p.legend(labels = ['not infected','infected'], loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
            p.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        return meta

    def simple_interpolate(self,meta, n_points):
        for i in range(n_points+1):
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
    
    def plot_comprehensive(self, al_obj, aligner, mode = 'overall', order_S_legend=None, order_T_legend=None, paths_to_display=None, cmap='viridis'):
        
        if(mode=='overall'):
                # constructs the matrix that gives frequency count of matches between each ref and query pair of timepoints across all alignments the aligner has tested and computes simple optimal alignment based on that matrix            
            al_str, path = aligner.compute_overall_alignment()
            self.plot_comprehensive_alignment_landscape_plot(al_obj, landscape_mat =  aligner.get_pairwise_match_count_mat() , order_S_legend=None, order_T_legend=None, paths_to_display=[path], cmap='viridis')
        else:
            self.plot_comprehensive_alignment_landscape_plot(al_obj, landscape_mat = pd.DataFrame(al_obj.landscape_obj.L_matrix), order_S_legend=None, order_T_legend=None, paths_to_display=None, cmap=cmap)

    def plot_comprehensive_alignment_landscape_plot(self, al_obj, landscape_mat, order_S_legend=None, order_T_legend=None, paths_to_display=None, cmap='viridis'):

        if(paths_to_display==None):
            al_obj.landscape_obj.alignment_path.append([0,0])
            paths_to_display=[al_obj.landscape_obj.alignment_path]
        nS_points=len(al_obj.S.time_points)
        nT_points=len(al_obj.T.time_points)
        match_points_S = np.unique(al_obj.match_points_S) + 1
        match_points_T = np.unique(al_obj.match_points_T) + 1

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
        ax1.axis(ymin=0, ymax=nT_points+1, xmin=0, xmax=nS_points+1)
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

        