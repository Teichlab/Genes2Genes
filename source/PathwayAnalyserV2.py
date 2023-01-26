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
