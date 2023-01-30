import anndata
import numpy as np
import pandas as pd
import seaborn as sb
import scanpy as sc
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os,sys,inspect
# setting the path to source
# sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) + '/source') 
sys.path.append('../source') 

# new source imports 
import OrgAlign as orgalign
import Main
import MyFunctions 
import TimeSeriesPreprocessor
# import PathwayAnalyser

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import normalize
import multiprocessing

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mm_type",
                    type=str)
parser.add_argument("mm_size",
                    type=int,
                    help="path to working directory")
args = parser.parse_args()

mm_type = args.mm_type
mm_size = args.mm_size

def simulate_alignment2(adata, true_align_string, 
                       frac_query = 0.5,
                       seed=42352,
                       gene = 'Msi1',
                       n_stds = 1):
    np.random.seed(seed)
    n_bins=len(true_align_string)
    adata.obs['time_bins'] = pd.cut(adata.obs['time'], n_bins).astype('category').cat.codes
    q_cells= np.array([])

    ## Split in ref and query
    for i,b in enumerate(true_align_string):
        n_cells = sum(adata.obs['time_bins'] == i)
        q_cells_bin = np.random.choice(adata.obs_names[adata.obs['time_bins'] == i], size=int(np.round(n_cells*frac_query)), replace=False)
        q_cells = np.hstack([q_cells, q_cells_bin])

    adata_query = adata[q_cells].copy()
    adata_ref = adata[~adata.obs_names.isin(q_cells)].copy()
    
    ## Calculate shift for insertion
    X_query = adata_query.X.copy()
    X_gene = X_query[:,adata_query.var_names == gene]
    ins_shift = n_stds*X_gene.std()
    
    for i,b in enumerate(true_align_string):
        bcells = adata_query.obs_names[adata_query.obs['time_bins'] == i]
        if b == 'D': ## delete cells
            adata_query = adata_query[~adata_query.obs_names.isin(bcells)].copy()
        if b == 'I': # change values for gene expression            
            X_query = adata_query.X.copy()
            X_gene = X_query[adata_query.obs_names.isin(bcells),adata_query.var_names == gene]
            X_query[adata_query.obs_names.isin(bcells),adata_query.var_names == gene] = X_gene + ins_shift
            adata_query.X = X_query.copy()
    
    # Algorithm expect time spanning from 0 to 1
    adata_ref.obs['time'] = normalize(adata_ref.obs['time'].values.reshape(1,-1), norm='max').ravel()
    adata_query.obs['time'] = normalize(adata_query.obs['time'].values.reshape(1,-1), norm='max').ravel()
    # adata_query.obs.loc[adata_query.obs['time'].idxmax(), 'time'] = 1.0
    return(adata_ref, adata_query)

def make_align_string(mm_type, mm_start = 10, n_bins = 40, mm_size=10):
    mm_ixs = range(mm_start, mm_start+mm_size)
    true_align_string = ''.join([mm_type if i in mm_ixs else 'M' for i in range(n_bins)])
    return(true_align_string)

def alignment_viz(aligner, al_obj):
    # plt.subplots(1,2,figsize=(10,3))
    # plt.subplot(1,2,1)
    # al_obj.plotTimeSeries(aligner, plot_cells=True)
    # plt.subplot(1,2,2)
    # al_obj.plotTimeSeriesAlignment()
    print(al_obj.al_visual)
    
def predict_alignment(adata_ref, adata_query, gene, n_bins=40):
    gene_list = adata_ref.var_names 
    aligner = Main.RefQueryAligner(adata_ref, adata_query, gene_list, n_bins)
    aligner.WEIGHT_BY_CELL_DENSITY = True
    aligner.WINDOW_SIZE = 0.1
    al_obj = aligner.align_single_pair(gene)
    alignment_viz(aligner, al_obj)
    return(al_obj)

def get_ref_aling_str(al_obj):
    ref_ixs = (al_obj.al_visual.split('\n')[1]).split(' Reference')[0]
    al_str = al_obj.alignment_str
    ref_aling_str = ''.join([al_str[i] for i,p in enumerate(ref_ixs) if p!=' ' and al_str[i] != 'V'])
    return(ref_aling_str)


def run_match_accuracy(params):
    adata, gene, align_params = params
    match_dict = {'D':'mismatch', 'I':'mismatch', 'M':'match', 'V':'match', 'W':'match'}
    true_align_string = make_align_string(**align_params)
    rdata, qdata = simulate_alignment2(adata, true_align_string, gene=gene)
    al_obj = predict_alignment(rdata, qdata, gene=gene)

    true_ref_align_str = get_ref_aling_str(al_obj)

    # get mismatch accuracy
    outcome_df = pd.DataFrame([(i, match_dict[true_align_string[i]], match_dict[c]) for i,c in enumerate(get_ref_aling_str(al_obj) )],
                 columns=['position', 'true', 'predicted']
                )
    outcome_df['correct'] = outcome_df['true'] == outcome_df['predicted']
    accuracy = outcome_df['correct'].sum()/outcome_df['correct'].shape[0]
    outcome_df['accuracy'] = accuracy
    outcome_df['gene'] = gene
    for p in align_params.keys():
        outcome_df[p] = align_params[p]
    outcome_df = outcome_df[list(align_params.keys()) + ['gene', 'accuracy']].drop_duplicates()
    return(outcome_df)

adata = sc.read_h5ad("./data/match_accuracy_pancreas.h5ad")
match_outcome = pd.DataFrame()
pool = multiprocessing.Pool(7)
outcomes_g = pool.map(run_match_accuracy, 
                      [(adata, g, {'mm_type':mm_type, 'n_bins':40, 'mm_start':0, 'mm_size':mm_size}) for g in adata.var_names[adata.var['simulation_gene']]])
pool.close()
outcomes_g = pd.concat(outcomes_g)
match_outcome = pd.concat([match_outcome, outcomes_g])
match_outcome.to_csv(f'./data/match_accuracy_pancreas.{mm_type}.size{str(mm_size)}.csv')