import gseapy as gp
from gseapy import barplot, dotplot
import anndata
import time 
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sb
import scipy.stats as stats
import matplotlib.pyplot as plt
import os,sys,inspect
import pickle
from tqdm import tqdm
from tabulate import tabulate
from gsea_api.molecular_signatures_db import MolecularSignaturesDatabase
from adjustText import adjust_text
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import zscore

from . import ClusterUtils
from . import VisualUtils


def run_overrepresentation_analysis(gene_set, TARGET_GENESETS=['MSigDB_Hallmark_2020','KEGG_2021_Human']):
    enr = gp.enrichr(gene_list=gene_set,
                     gene_sets=TARGET_GENESETS,
                     organism='human',
                     outdir=None,
                       )
    df = enr.results[enr.results['Adjusted P-value']<0.05]
    if(df.shape[0]==0):
        return df
    df = df.sort_values('Adjusted P-value')
    df['-log10 Adjusted P-value'] = [-np.log10(q) for q in df['Adjusted P-value']]
    max_q = max(df['-log10 Adjusted P-value'][df['-log10 Adjusted P-value']!=np.inf])
    #df.columns = ['Gene_set']+list(df.columns[1:len(df.columns)])
    qvals = []
    for q in df['-log10 Adjusted P-value']:
        if(q==np.inf):
            q = -np.log10(0.00000000001) # NOTE: For -log10(p=0.0) we replace p with a very small p-val to avoid inf
        qvals.append(q)
    df['-log10 FDR q-val'] = qvals 
    df = df.sort_values('Adjusted P-value',ascending=True)
    return df

def plot_overrep_results(df):
    height = df.shape[0]*(1/(np.log2(df.shape[0])+1)) 
    ax = barplot(df,
                  column="Adjusted P-value",
                  group='Gene_set', # set group, so you could do a multi-sample/library comparsion
                  size=10,
                  top_term=20,
                  figsize=(5,height),
                  color=['darkred', 'darkblue'], # set colors for group
                 )

def plot_gsea_dotplot(df, size=100, figsize=(3,4), n_top_terms = 5):
        ax = dotplot(df,
                  column="P-value",
                  x='-log10 Adjusted P-value', # set x axis, so you could do a multi-sample/library comparsion
                  size=size,
                  top_term=n_top_terms,
                  figsize=figsize,
                  xticklabels_rot=45, # rotate xtick labels
                  show_ring=False, # set to False to revmove outer ring
                  marker='o',
                 )   

def run_cluster_overrepresentation_analysis(aligner):

    overrep_cluster_results = {}
    cluster_overrepresentation_results = [] 

    for cluster_id in tqdm(range(len(aligner.gene_clusters))):
        df = run_overrepresentation_analysis(aligner.gene_clusters[cluster_id])
        if(df.shape[0]==0):
            continue
        n_genes = len(aligner.gene_clusters[cluster_id])
        pathways = list(df.Term) 
        pathway_specific_genes = list(df.Genes) 
        sources = [str(s).split('_')[0] for s in list(df.Gene_set)] 

        if(n_genes<15):
            genes = aligner.gene_clusters[cluster_id]
        else:
            genes = aligner.gene_clusters[cluster_id][1:7] + [' ... '] +  aligner.gene_clusters[cluster_id][n_genes-7:n_genes]

        cluster_overrepresentation_results.append([cluster_id,n_genes,genes,pathways, pathway_specific_genes, sources]) 
        overrep_cluster_results[cluster_id] = df 

    results= pd.DataFrame(cluster_overrepresentation_results)
    print(tabulate(results,  headers=['cluster_id','n_genes', 'Cluster genes', 'Pathways','Pathway genes','Source'],tablefmt="grid",maxcolwidths=[3, 3, 3,30,40,40,10])) 

        


def get_pathway_alignment_stat(aligner, GENE_LIST, pathway_name, cluster=False, FIGSIZE = (14,7)):
    
   # print('PATHWAY ======= ',pathway_name)
   # GENE_LIST = IGS.SETS[pathway_name]
    perct_A = []
    perct_S = []
    perct_T = []
    for gene in GENE_LIST:
        series_match_percent = aligner.results_map[gene].get_series_match_percentage()
        perct_A.append(series_match_percent[0])
        perct_S.append(series_match_percent[1])
        perct_T.append(series_match_percent[2])

    print('mean matched percentage: ', round(np.mean(perct_A),2),'%' )
    print('mean matched percentage wrt ref: ',round(np.mean(perct_S),2),'%'  )
    print('mean matched percentage wrt query: ', round(np.mean(perct_T),2),'%' )
    average_alignment, alignment_path =  ClusterUtils.get_cluster_average_alignments(aligner, GENE_LIST)
    mat = ClusterUtils.get_pairwise_match_count_mat(aligner,GENE_LIST )
    print('Average Alignment: ', average_alignment)
    VisualUtils.plot_alignment_path_on_given_matrix(paths = [alignment_path], mat=mat) #AAAAAAAA
 #  plt.xlabel('Ref pseudotime')
    # plt.ylabel('Organoid pseudotime')
   # plt.savefig('Ref_organoid_'+pathway_name+'_overall_alignment.png')
    plot_mean_trend_heatmaps(aligner,GENE_LIST, pathway_name,cluster=cluster, FIGSIZE=FIGSIZE) 

def plot_DE_genes(pathway_name):
    PATHWAY_SET = IGS.SETS[pathway]
    ax=sb.scatterplot(x['l2fc'],x['sim']*100,s=50, legend=False, hue =x['sim'] ,palette=sb.diverging_palette(15, 133, s=50, as_cmap=True),edgecolor='k',linewidth=0.3)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.ylabel('Alignment Similarity %', fontsize=12, fontweight='bold')
    plt.xlabel('L2FC mean expression', fontsize = 12, fontweight='bold')
    plt.grid(False)
    plt.tight_layout()

    TEXTS = [] 
    for label, a, b in zip(x.index, x['l2fc'],x['sim']*100):
        if(label in PATHWAY_SET):# and b<=50):
            TEXTS.append(ax.text(a, b, label, color='white', fontsize=9, fontweight='bold',bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.75)))
    adjust_text(TEXTS, expand_points=(2, 2),arrowprops=dict(arrowstyle="->", color='black', lw=2))
    plt.title(pathway_name,fontweight='bold', fontsize=15)

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
        from scipy.stats import zscore
        f['z_normalised'] = zscore(f[0])
        S_mat.append(np.asarray(f[f[1]=='Ref'][0]))
        T_mat.append(np.asarray(f[f[1]=='Organoid'][0]))    
        S_zmat.append(np.asarray(f[f[1]=='Ref']['z_normalised']))
        T_zmat.append(np.asarray(f[f[1]=='Organoid']['z_normalised']))  
    S_mat = pd.DataFrame(S_mat)
    T_mat = pd.DataFrame(T_mat)
    S_zmat = pd.DataFrame(S_zmat)
    T_zmat = pd.DataFrame(T_zmat)
    
    S_mat.index = GENE_LIST #IGS.SETS[pathway_name]
    T_mat.index = GENE_LIST #IGS.SETS[pathway_name]
    S_zmat.index = GENE_LIST#IGS.SETS[pathway_name]
    T_zmat.index = GENE_LIST#IGS.SETS[pathway_name]
    
   # print('Interpolated mean trends')
   # plot_heatmaps(S_mat, T_mat, pathway_name, cluster=cluster)
    print('Z-normalised Interpolated mean trends')
    plot_heatmaps(S_zmat, T_zmat, GENE_LIST, pathway_name,cluster=cluster, FIGSIZE=FIGSIZE)

def plot_heatmaps(mat_ref,mat_query,GENE_LIST, pathway_name, cluster=False, FIGSIZE=(14,7)):
    
    if(cluster):
        g=sb.clustermap(mat_ref, figsize=(0.4,0.4), col_cluster=False, cbar_pos=None) 
        gene_order = g.dendrogram_row.reordered_ind
        df = pd.DataFrame(g.data2d) 
        df.index = GENE_LIST[gene_order]
    else:
        df=mat_ref
    plt.close()
    
    plt.subplots(1,2,figsize=FIGSIZE) #8,14/7 ******************************************************
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
        mat_query = mat_query.loc[GENE_LIST[gene_order]] 
    ax = sb.heatmap(mat_query,vmax=max_val,  vmin=min_val,cbar_kws = dict(use_gridspec=False,location="top"), yticklabels=False) 
    plt.title('Query')
    plt.savefig(pathway_name+'_heatmap.png', bbox_inches='tight')
    plt.show()

        

class InterestingGeneSets:
    
    def __init__(self, MSIGDB_PATH ='../OrgAlign/msigdb/' ):        
        self.SETS = {}
        self.dbs = {}
        self.msigdb = MolecularSignaturesDatabase(MSIGDB_PATH , version='7.5.1')
        self.dbs['kegg'] = self.msigdb.load('c2.cp.kegg', 'symbols')
        self.dbs['hallmark'] = self.msigdb.load('h.all', 'symbols')
        #self.dbs['gobp'] = self.msigdb.load('c5.go.bp', 'symbols')
        #self.dbs['gocc'] = self.msigdb.load('c5.go.cc', 'symbols')
        #self.dbs['reac'] = self.msigdb.load('c2.cp.reactome', 'symbols')
        
    def add_new_set_from_msigdb(self, db_name, dbsetname, avail_genes, usersetname):
        self.SETS[usersetname] = np.intersect1d(list(self.dbs[db_name].gene_sets_by_name[dbsetname].genes), avail_genes)

    def add_new_set(self, geneset, usersetname, avail_genes):
        geneset = np.asarray(geneset)
        #print(geneset)
        self.SETS[usersetname] = geneset[np.where([g in avail_genes for g in geneset])]

        

        

        

        

        

        

        

        

        

# ATTIC

def get_ranked_genelist(aligner):
        #print('make ranked gene list')
        matched_percentages = {}
        for al_obj in aligner.results:
            matched_percentages[al_obj.gene] = (al_obj.get_series_match_percentage()[0]/100 )
        x = sorted(matched_percentages.items(), key=lambda x: x[1], reverse=False)
        x = pd.DataFrame(x)
        x.columns = ['Gene','Alignment_Percentage']
        x = x.set_index('Gene')
        return x

# get the top k DE genes (k decided on matched percentage threshold) 
def topkDE(aligner, DIFF_THRESHOLD=0.5):
        ranked_list = get_ranked_genelist(aligner)
        top_k = np.unique(ranked_list ['Alignment_Percentage'] < DIFF_THRESHOLD , return_counts=True)[1][1]
        print(top_k, ' # of DE genes to check')
        clusters = pd.DataFrame([ranked_list[0:top_k].index, np.repeat(0,top_k)]).transpose()
        clusters.columns = ['Gene','ClusterID']
        clusters = clusters.set_index('Gene')
        return list(clusters.index), ranked_list 

def run_GSEA_on_rankedlist(rankedDEgenes):
    pre_res = gp.prerank(rnk=rankedDEgenes, # or rnk = rnk,
                         gene_sets=['MSigDB_Hallmark_2020','KEGG_2021_Human','Reactome_2022','GO_Biological_Process_2021'],#targets5,
                         threads=4,
                         min_size=5,
                         max_size=1000,
                         permutation_num=1000,
                         outdir=None, 
                         seed=6,
                         verbose=True, 
                        )
    pre_res.res2d[pre_res.res2d['FDR q-val']<0.05]

    df = pre_res.res2d[pre_res.res2d['FDR q-val']<0.05]
    df['Name'] = [str(t).split('_')[0] for t in df.Term]  
    df = df.sort_values('FDR q-val')
    df['-log10 FDR q-val'] = [-np.log10(q) for q in df['FDR q-val']]
    max_q = max(df['-log10 FDR q-val'][df['-log10 FDR q-val']!=np.inf])
    #df.columns = ['Gene_set']+list(df.columns[1:len(df.columns)])
    qvals = []
    for q in df['-log10 FDR q-val']:
        if(q==np.inf):
            q = -np.log10(0.00000000001) # NOTE: For -log10(p=0.0) we replace p with a very small p-val to avoid inf
        qvals.append(q)
    df['-log10 FDR q-val'] = qvals 
    #df['Name'] = df['Gene_set']
    sb.set(rc={'figure.figsize':(10,15)})
    sb.factorplot(y='Term', x='-log10 FDR q-val', data=df, kind='bar', hue='Name',dodge=False)
    plt.xlim([0,max_q])
    
    return pre_res
