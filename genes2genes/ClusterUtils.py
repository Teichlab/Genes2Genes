from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from numpy import argmax
import numpy as np
import pandas as pd
import seaborn as sb
import scipy
from leven import levenshtein   
import sklearn 
from tqdm import tqdm

from . import VisualUtils
from . import PathwayAnalyserV2


def compute_levenshtein_dist_matrix(set_of_strings):
    print('using levenshtein distance metric')
    E = []
    for s1 in set_of_strings:
        row = []
        for s2 in set_of_strings:
            row.append(levenshtein(s1,s2)/np.max([len(s1),len(s2)]))
        E.append(row)
    return E

def compute_hamming_dist_matrix(set_of_strings):
    
    print('using hamming distance metric')
    al_binary_encodings= []
    for a in set_of_strings:
        al_binary_encodings.append(binary_encode_alignment_path(a))
    E = distance.cdist(al_binary_encodings, al_binary_encodings, 'hamming')
    return E

def run_agglomerative_clustering(E, gene_list, DIST_THRESHOLD, linkage = "average"):
    model = sklearn.cluster.AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=linkage, distance_threshold = DIST_THRESHOLD) 
    model.fit(E)
    
    cluster_map = {}
    for i in range(len(gene_list)):
        cid = model.labels_[i]
        if(cid not in cluster_map.keys()):
            cluster_map[cid] = []
        cluster_map[cid].append(gene_list[i]) 
        
    if(len(cluster_map.keys())==1):
        return cluster_map, None, None, None, None
    silhouette_score = sklearn.metrics.silhouette_score(X=E , labels = model.labels_, metric='precomputed') 
    silhouette_score_samples = sklearn.metrics.silhouette_samples(X=E , labels = model.labels_, metric='precomputed') 
    silhouette_score_mode = scipy.stats.mode(silhouette_score_samples)[0][0]
    
    n_clusters_less_members = []
    for cid in cluster_map.keys():
        if(len(cluster_map[cid])<10):
            n_clusters_less_members.append(cid)
    n_small_clusters = len(n_clusters_less_members)
    
    #print('n clusters:', len(cluster_map.keys()), 'for','Dist threshold:', DIST_THRESHOLD, ' | silhouette score:', silhouette_score, 
    #      '| mode: ',silhouette_score_mode, ' | n small clusters: ', n_small_clusters )
    cluster_ids =  model.labels_
    return cluster_map, cluster_ids, silhouette_score, silhouette_score_mode, n_small_clusters

def run_clustering_for_subset(aligner, metric, MIN_DIST = 0.001):
    
    # extract the alignment strings to cluster
    alignment_strings = []
    for i in range(len(aligner.gene_list)):
        alignment_strings.append(aligner.results[i].alignment_str)  
        
    dist_mat_functions = {'hamming': compute_hamming_dist_matrix, 'levenshtein': compute_levenshtein_dist_matrix}
    n_iter = 50 
    np.random.seed(0)
    n_unique = []; n_clusters = []; equal=[]
    for i in range(n_iter):
        rand_indices = np.random.choice(np.arange(len(alignment_strings)), 200)
        string_subset = []; gene_subset = []
        for r in rand_indices:
            string_subset.append(alignment_strings[r])
            gene_subset.append(aligner.gene_list[r])
        
        compute_dist_matrix = dist_mat_functions[metric]
        E = compute_dist_matrix(string_subset)
        gene_clusters, cluster_ids, silhouette_score, silhouette_score_mode, n_small_cluster = run_agglomerative_clustering(E, gene_subset, MIN_DIST)
        
        n_clusters.append(len(gene_clusters.keys()))
        n_unique.append(len(np.unique(string_subset)))
        
    df = pd.DataFrame([n_unique, n_clusters]).transpose()
    df.columns = ['n_unique_strings','n_identified_clusters']
    df['equal'] = df['n_unique_strings'] == df['n_identified_clusters']
    return df

def run_clustering(aligner, metric,DIST_THRESHOLD=0.2, experiment_mode=False):
    
    # extract the alignment strings to cluster
    alignment_strings = []
    for i in range(len(aligner.gene_list)):
        alignment_strings.append(aligner.results[i].alignment_str)  

    # compute distance matrix 
    print('compute distance matrix')
    dist_mat_functions = {'hamming': compute_hamming_dist_matrix, 'levenshtein': compute_levenshtein_dist_matrix}
    compute_dist_matrix = dist_mat_functions[metric]
    E = compute_dist_matrix(alignment_strings)
    
    if(experiment_mode):
        scores = [];  n_clusters = []; dist_thresholds = np.arange(0.01,1.0,0.01); score_modes = []; n_small_clusters = [] 
        eval_dists = []
        for D_THRESH in tqdm(dist_thresholds): 
            gene_clusters, cluster_ids, silhouette_score, silhouette_score_mode, n_small_cluster = run_agglomerative_clustering(E, aligner.gene_list, D_THRESH)
            
            if(len(gene_clusters.keys())==1):
                break
            scores.append(silhouette_score)
            n_clusters.append(len(gene_clusters.keys()))
            score_modes.append(silhouette_score_mode)
            n_small_clusters.append(n_small_cluster)
            eval_dists.append(D_THRESH)
            
        plt.rcParams.update({'font.size': 14})
        plt.subplots(1,3,figsize=(10,5))
        plt.subplot(1,3,1)
        sb.lineplot(x=eval_dists, y=scores, color = 'blue', marker='o') 
        plt.xlabel('Distnace threshold')
        plt.ylabel('Mean Silhouette Score')
        plt.subplot(1,3,2)
        sb.lineplot(x=n_clusters, y=scores, color='red', marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Mean Silhouette Score')
        plt.subplot(1,3,3)
        sb.lineplot(x=eval_dists, y=n_clusters, color='green', marker='o')
        plt.xlabel('Distance threshold')
        plt.ylabel('Number of clusters')
        plt.tight_layout() 
        df = pd.DataFrame([eval_dists,scores,n_clusters]).transpose()
        df.columns = ['Distance threshold', 'Mean Silhouette Score','Number of clusters']
        return df
    else:
        print('run agglomerative clustering | ', np.round(DIST_THRESHOLD,2) )
        gene_clusters, cluster_ids, silhouette_score, silhouette_score_samples, n_small_cluster  = run_agglomerative_clustering(E, aligner.gene_list, DIST_THRESHOLD)
        print('silhouette_score: ', silhouette_score)
        aligner.cluster_ids = cluster_ids
        aligner.gene_clusters = gene_clusters 
        aligner.DistMat = pd.DataFrame(E)
        aligner.DistMat.columns = aligner.gene_list
        aligner.DistMat.index = aligner.gene_list
        aligner.E = E
        
        for i in range(len(aligner.gene_clusters)):
            for g in aligner.gene_clusters[i]:
                aligner.results_map[g].cluster_id = i
        
        #return aligner


def visualise_clusters(aligner, n_cols = 5, figsize= (10,6)): 
    
        cluster_ids = aligner.cluster_ids
        alignments = aligner.results 
        cluster_map = aligner.gene_clusters
        n_clusters = len(cluster_map.keys())

        S_len = alignments[0].fwd_DP.S_len
        T_len = alignments[0].fwd_DP.T_len 
        unique_cluster_ids = cluster_map.keys() #np.unique(cluster_ids)
        n_rows = int(np.ceil(n_clusters/n_cols)) 
        fig, axs = plt.subplots(n_rows,n_cols, figsize = (12,n_rows*3)) # custom -- only for 20 clusters -- TODO change later
        axs = axs.flatten() 
        i = 0; k=1

        for cid in range(n_clusters): 
            cluster_alignments = cluster_map[cid]
            paths = []
            for g in cluster_alignments:
                    a = aligner.results_map[g]
                    a.fwd_DP.alignment_path.append([0,0])
                    paths.append(a.fwd_DP.alignment_path)
            plot_different_alignments(paths, S_len, T_len, axs[cid])
            axs[cid].set_title('Cluster-'+str(i) + ' | '+str(len(cluster_alignments)))
            i=i+1; k=k+1
        fig.tight_layout()
        n = n_cols * n_rows; i = 1
        while(k<=n):
            axs.flat[-1*i].set_visible(False) 
            k = k+1
            i=i+1  


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


def get_significant_pathways_for_clusters(aligner, PVAL_THRESHOLD=0.01):
    results = {}
    cluster_map = aligner.gene_clusters
    for cid in tqdm(cluster_map.keys()):
        glist = []
        for g in cluster_map[cid]:
            glist.append(g)

        df = PathwayAnalyserV2.run_overrepresentation_analysis(glist)
        results[cid] = df[df['Adjusted P-value']< PVAL_THRESHOLD]

    return results



def print_cluster_average_alignments(aligner):
    
    aligner.gene_clusters_average_alignments = {}
    for i in range(len(aligner.gene_clusters)):
        average_alignment, avg_path =  get_cluster_average_alignments(aligner, aligner.gene_clusters[i] )
        print('cluster: ', i, VisualUtils.color_al_str(average_alignment), len(aligner.gene_clusters[i]))
        aligner.gene_clusters_average_alignments[i] = average_alignment



# G2G Version 1 ---- PREVIOUS CLUSTERING UTILS


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
                    a.fwd_DP.alignment_path.append([0,0])
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


def one_hot_encode(al_string):
    #print('One hot encoding for alignment string clustering')
    # Modified code in https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/ 
    alphabet = 'MWVID'
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_encoding = [char_to_int[char] for char in al_string]
    onehot_encoding = []
    for value in int_encoding:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoding.append(letter)
    
    return onehot_encoding

def run_hierarchical_clustering_with_one_hot_encoding(aligner,DIST_THRESHOLD=25):
    
    alignment_strings = []
    for i in range(len(aligner.gene_list)):
        alignment_strings.append(aligner.results[i].alignment_str)
    
    E = [] # alignment encoding matrix
    longestL = np.max([len(a) for a in alignment_strings]) 
    for alignment_str in alignment_strings:
        encoding = one_hot_encode(alignment_str)
        max_len = longestL
        for i in range(max_len - len(alignment_str)):
            encoding.append([0,0,0,0,0]) 
        E.append(list(np.asarray(encoding).flatten())) 
    E = pd.DataFrame(E)

    model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward',distance_threshold = DIST_THRESHOLD)
    model.fit(E)
    cluster_ids = model.labels_
    
    gene_clusters = check_alignment_clusters(cluster_ids, aligner.results , n_clusters=len(np.unique(cluster_ids) ) )

    aligner.cluster_ids = cluster_ids
    aligner.gene_clusters = gene_clusters 
    aligner.DistMat = pd.DataFrame(euclidean_distances(E, E)) 
    aligner.DistMat.columns = aligner.gene_list
    aligner.DistMat.index = aligner.gene_list
    aligner.E = E


# As ordinal data - numerical encoding 
def run_hierarchical_clustering_with_int_encoding_scheme(aligner,DIST_THRESHOLD=25, 
                                          codes = [1,2,3,4,5]  ):
    
    aligment_strings = []
    for i in range(len(aligner.gene_list)):
        aligment_strings.append(aligner.results[i].alignment_str)
    
    longestL = np.max([len(a) for a in aligment_strings]) # 30
    
    # Numerical Encoding of 5-state Alignment strings
    # None - 0; M - 1; W - 2; V - 3; D - 5; I - 6 
    E = [] # alignment encoding matrix
    for i in range(len(aligment_strings)):
        E.append(np.repeat(0, longestL)) 
    E = np.matrix(E) 

    for i in range(len(aligment_strings)):
        for j in range(len(aligment_strings[i])):
            al_state = aligment_strings[i][j]
            if(al_state=='M'):
                E[i,j] = codes[0]
            elif(al_state=='W'):
                E[i,j] = codes[1]
            elif(al_state=='V'):
                E[i,j] = codes[2]
            elif(al_state=='D'):
                E[i,j] = codes[3]
            elif(al_state=='I'):
                E[i,j] = codes[4]
    E = pd.DataFrame(E)
    model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward',distance_threshold = DIST_THRESHOLD)
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


# ========== BEST VERSION ====================================================
def binary_encode_alignment_path(al_str):
    S = ''
    T = ''
    for i in range(len(al_str)):
        if(al_str[i]=='M'):
            S+='1'
            T+='1'
        elif(al_str[i]=='W'):
            S+='1'
        elif(al_str[i]=='V'):
            T+='1'
        elif(al_str[i]=='D'):
            S+='0'
        elif(al_str[i]=='I'):
            T+='0'
    binary_encoding = S+T
    return np.asarray([int(c) for c in binary_encoding]  ) 


def run_hierarchical_clustering_with_binary_encode_alignment_path_euclidean(aligner,DIST_THRESHOLD=25):
    
    alignment_strings = []
    for i in range(len(aligner.gene_list)):
        alignment_strings.append(aligner.results[i].alignment_str)
    
    E = [] # alignment encoding matrix
    longestL = np.max([len(a) for a in alignment_strings]) 
    for alignment_str in alignment_strings:
        encoding = binary_encode_alignment_path(alignment_str)
        E.append(list(np.asarray(encoding).flatten())) 
    E = pd.DataFrame(E)

    model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward',distance_threshold = DIST_THRESHOLD)
    model.fit(E)
    cluster_ids = model.labels_
    
    gene_clusters = check_alignment_clusters(cluster_ids, aligner.results , n_clusters=len(np.unique(cluster_ids) ) )

    aligner.cluster_ids = cluster_ids
    aligner.gene_clusters = gene_clusters 
    aligner.DistMat = pd.DataFrame(euclidean_distances(E, E)) 
    aligner.DistMat.columns = aligner.gene_list
    aligner.DistMat.index = aligner.gene_list
    aligner.E = E

def run_hierarchical_clustering_with_binary_encode_alignment_path_hamming(aligner,DIST_THRESHOLD=25):
    
    alignment_strings = []
    for i in range(len(aligner.gene_list)):
        alignment_strings.append(aligner.results[i].alignment_str)
    
    E = [] # alignment encoding matrix

    al_binary_encodings= []
    for a in aligner.results:
        al_binary_encodings.append(binary_encode_alignment_path(a.alignment_str))
    E = distance.cdist(al_binary_encodings, al_binary_encodings, 'hamming')

    model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',distance_threshold = DIST_THRESHOLD)
    #model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward',distance_threshold = DIST_THRESHOLD)
    model.fit(E)
    cluster_ids = model.labels_
    print('run')
    gene_clusters = check_alignment_clusters(cluster_ids, aligner.results , n_clusters=len(np.unique(cluster_ids) ) )

    aligner.cluster_ids = cluster_ids
    aligner.gene_clusters = gene_clusters 
    aligner.DistMat = pd.DataFrame(E)
    aligner.DistMat.columns = aligner.gene_list
    aligner.DistMat.index = aligner.gene_list
    aligner.E = E

# ========== BEST VERSION ====================================================

























# Cluster Average Util



# This computes an average alignment across all the alignments for a given set of gene alignments 
def get_cluster_average_alignments(aligner, gene_set, deterministic=True):
    
            cluster_alobjs = []
            for g in gene_set:
                cluster_alobjs.append(aligner.results_map[g])
            i = aligner.results[0].fwd_DP.T_len
            j = aligner.results[0].fwd_DP.S_len
            
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
                # record the count of each state at this [i,j] cell across all alignments 
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
                # compute the proportion of the state for the [i,j] cell
                for state in backtrack_states_probs.keys(): 
                    backtrack_states_probs[state] = backtrack_states_probs[state]/len(cluster_alobjs) 

                if(deterministic):
                    # take the most probable state based on max frequent state of this [i,j] cell
                    cs = np.argmax(np.asarray(list(backtrack_states_probs.values())) )
                else:
                    # sample a state from the state frequency distribution
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

def get_pairwise_match_count_mat(aligner, gene_set):
            mat = []
            nT_points = len(aligner.results[0].T.time_points)
            nS_points = len(aligner.results[0].S.time_points)
            for i in range(nT_points + 1):
                mat.append(np.repeat(0.0, nS_points+1))

            # counts of total matches between the each pair of ref and query timepoints across all alignments 
            for g in gene_set:
                a = aligner.results_map[g]
                matchS = a.match_points_S+1
                matchT = a.match_points_T+1
                for i in range(len(matchS)):
                    mat[matchT[i]][matchS[i]] = mat[matchT[i]][matchS[i]] + 1

            return pd.DataFrame(mat) 

        

        

        

        

# computes simple DP alignment (using match score = pairwise total match count frequency) across all gene-level alignments 
    # gap score is taken as penalising 8% of the total number of tested genes => so that it controls the matching based on the number of 
    # total matches (i.e. it controls the degree of significant matching) 
def compute_overall_alignment(aligner,mat, plot=False, GAP_SCORE = None):
                
                if(GAP_SCORE==None):
                    GAP_SCORE= -len(aligner.gene_list)*0.08

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

                        a = max([m_dir, d_dir, i_dir])
                    
                        if(a==d_dir):
                            opt = d_dir
                            dir_tracker = 1
                        elif(a==i_dir):
                            opt =i_dir
                            dir_tracker = 2
                        elif(a==m_dir):
                            opt = m_dir
                            dir_tracker = 0

                        opt_cost_M[i,j] = opt
                        tracker_M[i,j] = dir_tracker     

                # backtracking
                i = mat.shape[0]-1
                j = mat.shape[1]-1
                alignment_str = ''
                tracked_path = []
                while(True):
                    tracked_path.append([i,j])
                    if(tracker_M[i,j]==0):
                        alignment_str = 'M' + alignment_str
                        i = i-1
                        j = j-1
                    elif(tracker_M[i,j]==1):
                        if(mat.loc[i,j]>0):
                            alignment_str = 'W' + alignment_str
                        else:
                            alignment_str = 'D' + alignment_str
                        j = j-1
                    elif(tracker_M[i,j]==2):
                        if(mat.loc[i,j]>0):
                            alignment_str = 'V' + alignment_str
                        else:
                            alignment_str = 'I' + alignment_str
                        i = i-1 

                    if(i==0 and j==0) :
                        break
                tracked_path.append([0,0])
                # NOTE: This alignment string does not have the same interpretation as of the 5-state gene alignment string we get.
                # Here we are only interested in the path 
                return alignment_str, tracked_path#, opt_cost_M, tracker_M
