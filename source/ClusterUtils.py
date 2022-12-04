from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

def check_alignment_clusters(cluster_ids, alignments, n_cols = 5, figsize= (10,6), n_clusters=20): 
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
    
def plot_different_alignments(paths, S_len, T_len, ax, mat=[]): # pass alignment path coordinates
        mat=[]
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
        
        
def run_hierarchical_clustering(aligner):
    
    aligment_strings = []
    for i in range(len(aligner.gene_list)):
        aligment_strings.append(aligner.results[i].alignment_str)
    
    longestL = np.max([len(a) for a in aligment_strings]) # 30
    
    # Numerical Encoding of 5-state Alignment strings
    # None - 0; M - 1; W - 2; V - 3; D - 4; I - 5 
    E = [] # alignment encoding matrix
    for i in range(len(aligment_strings)):
        E.append(np.repeat(0, longestL)) 
    E = np.matrix(E) 

    for i in range(len(aligment_strings)):
        for j in range(len(aligment_strings[i])):
            al_state = aligment_strings[i][j]
            if(al_state=='M'):
                E[i,j] = 1
            elif(al_state=='W'):
                E[i,j] = 2
            elif(al_state=='V'):
                E[i,j] = 3
            elif(al_state=='D'):
                E[i,j] = 4
            elif(al_state=='I'):
                E[i,j] = 5
    E = pd.DataFrame(E)
    model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward',distance_threshold = 25)
    model.fit(E)
    cluster_ids = model.labels_
    
    gene_clusters = check_alignment_clusters(cluster_ids, aligner.results , n_clusters=len(np.unique(cluster_ids) ) )

    aligner.cluster_ids = cluster_ids
    aligner.gene_clusters = gene_clusters 
    aligner.DistMat = pd.DataFrame(euclidean_distances(E, E)) 
    aligner.DistMat.columns = aligner.gene_list
    aligner.DistMat.index = aligner.gene_list
    aligner.E = E
   # return cluster_ids, gene_clusters