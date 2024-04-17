import gseapy as gp
from gseapy import barplot, dotplot
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from gsea_api.molecular_signatures_db import MolecularSignaturesDatabase

from . import ClusterUtils
from . import VisualUtils

"""
This script defines Wrappers for GSEAPY enrichr and other functions related to analysing pathway gene sets. 
"""

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
    
    print('Gene set: ======= ', pathway_name)
    perct_A = []
    perct_S = []
    perct_T = []
    for gene in GENE_LIST:
        series_match_percent = aligner.results_map[gene].get_series_match_percentage()
        perct_A.append(series_match_percent[0])
        perct_S.append(series_match_percent[1])
        perct_T.append(series_match_percent[2])

    print('mean matched percentage: ', round(np.mean(perct_A),2),'%' )
    #print('mean matched percentage wrt ref: ',round(np.mean(perct_S),2),'%'  )
    #print('mean matched percentage wrt query: ', round(np.mean(perct_T),2),'%' )
    average_alignment, alignment_path =  ClusterUtils.get_cluster_average_alignments(aligner, GENE_LIST)
    mat = ClusterUtils.get_pairwise_match_count_mat(aligner,GENE_LIST )
    print('Average Alignment: ', VisualUtils.color_al_str(average_alignment), '(cell-level)')
    print('- Plotting average alignment path')
    VisualUtils.plot_alignment_path_on_given_matrix(paths = [alignment_path], mat=mat) 
    VisualUtils.plot_mean_trend_heatmaps(aligner,GENE_LIST, pathway_name,cluster=cluster, FIGSIZE=FIGSIZE) 
        

class InterestingGeneSets:
    
    def __init__(self, MSIGDB_PATH, version):        
        self.SETS = {}
        self.dbs = {}
        self.msigdb = MolecularSignaturesDatabase(MSIGDB_PATH , version=version)
        self.dbs['kegg'] = self.msigdb.load('c2.cp.kegg', 'symbols')
        self.dbs['hallmark'] = self.msigdb.load('h.all', 'symbols')
        #self.dbs['gobp'] = self.msigdb.load('c5.go.bp', 'symbols')
        #self.dbs['gocc'] = self.msigdb.load('c5.go.cc', 'symbols')
        #self.dbs['reac'] = self.msigdb.load('c2.cp.reactome', 'symbols')
        
    def add_new_set_from_msigdb(self, db_name, dbsetname, avail_genes, usersetname):
        self.SETS[usersetname] = np.intersect1d(list(self.dbs[db_name].gene_sets_by_name[dbsetname].genes), avail_genes)

    def add_new_set(self, geneset, usersetname, avail_genes):
        geneset = np.asarray(geneset)
        self.SETS[usersetname] = geneset[np.where([g in avail_genes for g in geneset])]

        
