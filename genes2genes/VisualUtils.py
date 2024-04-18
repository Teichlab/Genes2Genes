import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.patches as mpatches
import regex

from . import Main


vega_20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728',
            '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2',
            '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',]
    
def plot_celltype_barplot(adata, n_bins, annotation_colname, joint_cmap, plot_cell_counts = False, legend=False):

        if(plot_cell_counts):
            normalize = False
        else:
            normalize = 'columns'

        vec = adata.obs.time
        bin_edges = np.linspace(0, 1, num=n_bins)  
        bin_ids = np.digitize(vec, bin_edges, right=False) # use right=True if we don't need 1.0 cell to always be a single last bin 
        adata.obs['bin_ids'] = bin_ids
        tmp = pd.crosstab(adata.obs[annotation_colname],adata.obs['bin_ids'], normalize=normalize).T.plot(kind='bar', stacked=True,        
                                                                                                 color=joint_cmap,grid = False, legend=False, width=0.7,align='edge',figsize=(9,1))
        if(legend):    
            tmp.legend(title='Cell-type annotations', bbox_to_anchor=(1.5, 1.02),loc='upper right')
        plt.axis('off')
    
def visualize_gene_alignment(alignment, adata_ref, adata_query, annotation_colname, cmap=None):

            if(isinstance(alignment,Main.AligmentObj )):
                alignment = alignment.alignment_str

            matched_points_S, matched_points_T = get_matched_time_points(alignment)

            fig = plt.figure(figsize=(4,2))
            heights = [1, 1, 1] 
            gs = plt.GridSpec(3, 1, height_ratios=heights)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0],sharex=ax1)
            ax3 = fig.add_subplot(gs[2, 0],sharex=ax1)
            
            if(cmap is None):
                cmap = vega_20

            plt.subplot(3,1,1)
            
            metaS = pd.crosstab(adata_ref.obs.bin_ids, adata_ref.obs[annotation_colname])
            metaS.apply(lambda x: x*100/sum(x), axis=1).plot(kind='bar',stacked=True,color=cmap, grid = False, legend=False, width=0.7, ax=ax1)
            
            metaT = pd.crosstab(adata_query.obs.bin_ids, adata_query.obs[annotation_colname])
            metaT.apply(lambda x: x*100/sum(x), axis=1).plot(kind='bar',stacked=True,color=cmap, grid = False, legend=False, width=0.7,ax=ax3)
            
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
            ax1.set_ylabel('Ref', rotation=90)
            ax3.set_ylabel('Query',rotation=90)
            fig.text(0.5, -0.05, 'Pseudotime bins with cell type composition', ha='center')
            ax1.set_title('Alignment w.r.t cell type compositions')


def get_matched_time_points(alignment_str):
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
        plt.xlabel('Pseudotime')
        plt.ylabel('Gene expression')
        plt.subplot(1,3,3)
        g = sb.scatterplot(x=aligner.ref_time, y=np.asarray(aligner.ref_mat[al_obj.gene]), color = 'forestgreen', alpha=0.7, legend=False,linewidth=0.3,s=20 ) 
        plt.title('Reference')
        plt.ylim([min_val-0.5,max_val+0.5])
        plt.xlabel('Pseudotime')
        plt.ylabel('Gene expression')

def plotTimeSeriesAlignment(gene, aligner):  
    
        al_obj = aligner.results_map[gene]
        sb.scatterplot(x=al_obj.S.X, y=al_obj.S.Y, color = 'forestgreen' ,alpha=0.05, legend=False)#, label='Ref') 
        sb.scatterplot(x=al_obj.T.X, y=al_obj.T.Y, color = 'midnightblue' ,alpha=0.05, legend=False)#, label ='Query') 
        al_obj.plot_mean_trends() 
        plt.title(al_obj.gene)
        plt.xlabel('Pseudotime')
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

def show_gene_alignment(gene, aligner,  adata_ref, adata_query, annotation_colname, cmap=None):
    visualize_gene_alignment(aligner.results_map[gene].alignment_str,  adata_ref, adata_query, annotation_colname, cmap=cmap)
    plotTimeSeries(gene, aligner, plot_cells=True)
    aligner.results_map[gene].alignment_str
    print(color_al_str(aligner.results_map[gene].alignment_str)) 
    print('Optimal alignment cost:', round(aligner.results_map[gene].fwd_DP.opt_cost,3),'nits')
    print('Alignment similarity percentage:', aligner.results_map[gene].match_percentage,'%' )

    
# smoothened/interpolated mean trends + Z normalisation 
def plot_mean_trend_heatmaps(aligner, GENE_LIST, pathway_name, cluster=False, FIGSIZE=(14,7)):
    S_mat = []
    T_mat = []
    S_zmat = []
    T_zmat = []

    for gene in GENE_LIST:

        fS = pd.DataFrame([aligner.results_map[gene].S.mean_trend, np.repeat('Ref', len(aligner.results_map[gene].S.mean_trend))]).transpose()
        fT = pd.DataFrame([aligner.results_map[gene].T.mean_trend, np.repeat('Organoid', len(aligner.results_map[gene].T.mean_trend))]).transpose()
        f = pd.concat([fS,fT])
        f[0] = np.asarray(f[0], dtype=np.float64)
        f['z_normalised'] = zscore(f[0])
        S_mat.append(np.asarray(f[f[1]=='Ref'][0]))
        T_mat.append(np.asarray(f[f[1]=='Organoid'][0]))    
        S_zmat.append(np.asarray(f[f[1]=='Ref']['z_normalised']))
        T_zmat.append(np.asarray(f[f[1]=='Organoid']['z_normalised']))  
    S_mat = pd.DataFrame(S_mat)
    T_mat = pd.DataFrame(T_mat)
    S_zmat = pd.DataFrame(S_zmat)
    T_zmat = pd.DataFrame(T_zmat)
    
    S_mat.index = GENE_LIST  
    T_mat.index = GENE_LIST  
    S_zmat.index = GENE_LIST 
    T_zmat.index = GENE_LIST 
    
    print('- Plotting z-normalised interpolated mean trends')
    plot_heatmaps(S_zmat, T_zmat, GENE_LIST, pathway_name,cluster=cluster, FIGSIZE=FIGSIZE)

def plot_heatmaps(mat_ref,mat_query,GENE_LIST, pathway_name, cluster=False, FIGSIZE=(14,7), write_file=False):
    
    if(cluster):
        g=sb.clustermap(mat_ref, figsize=(0.4,0.4), col_cluster=False, cbar_pos=None) 
        gene_order = g.dendrogram_row.reordered_ind
        df = pd.DataFrame(g.data2d) 
        df.index = GENE_LIST[gene_order]
    else:
        df=mat_ref
    plt.close()
    
    plt.subplots(1,2) #8,14/7 ******************************************************
    max_val = np.max([np.max(mat_ref),np.max(mat_query)]) 
    min_val = np.min([np.min(mat_ref),np.min(mat_query)]) 
    plt.subplot(1,2,1)
    ax=sb.heatmap(df, vmax=max_val,vmin=min_val, cbar_kws = dict(use_gridspec=False,location="top"), xticklabels=True, yticklabels=True) 
    plt.title('Reference')
    ax.yaxis.set_label_position("left")
    for tick in ax.get_yticklabels():
        tick.set_rotation(360)
    plt.subplot(1,2,2)
    if(cluster):
        mat_query = mat_query.loc[GENE_LIST[gene_order]] 
    ax = sb.heatmap(mat_query,vmax=max_val,  vmin=min_val,cbar_kws = dict(use_gridspec=False,location="top"), xticklabels=True, yticklabels=False) 
    plt.title('Query')
    if(write_file):
        plt.savefig(pathway_name+'_heatmap.png', bbox_inches='tight')
    plt.show()