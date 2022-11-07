from gsea_api.molecular_signatures_db import MolecularSignaturesDatabase
import scanpy as sc
from gprofiler import GProfiler
from tabulate import tabulate
from tqdm import tqdm
import enum
import numpy as np
import pandas as pd

class DB_NAMES(enum.Enum):
    
    BIOCARTA = 'c2.cp.biocarta'
    KEGG = 'c2.cp.kegg'
    PID = 'c2.cp.pid'
    REACTOME = 'c2.cp.reactome'
    WIKIPATHWAYS = 'c2.cp.wikipathways'
    IMMUNESIGDB = 'c7.immunesigdb'
    GOBP = 'c5.go.bp'
    
class PathwayAnalyser:
     
    def __init__(self):
        self.msigdb = MolecularSignaturesDatabase('msigdb', version='7.5.1')
        self.loaded_dbs = {}
        self.loaded_genesets = {}
        
    def load_db(self, DB_NAME):
        self.loaded_dbs[DB_NAME] = self.msigdb.load(DB_NAME, 'symbols')
            
    def load_genesets(self, DB_NAME, avail_genes):
        
        if(DB_NAME in self.loaded_genesets.keys()):
            return 
        if(DB_NAME not in self.loaded_dbs.keys()):
            self.load_db(DB_NAME)
        
        pathways = self.loaded_dbs[DB_NAME].gene_sets_by_name.keys()
        pathways_to_test = {}
        
        for pathway in pathways:
            test_genes = list(self.loaded_dbs[DB_NAME].gene_sets_by_name[pathway].genes)
            test_genes = np.intersect1d(test_genes, avail_genes)
            n_genes = len(test_genes)
            if(n_genes>0):
                pathways_to_test[pathway] = test_genes
        
        self.loaded_genesets[DB_NAME] = pathways_to_test
        
class PathwayAlignmentObj:
    def __init__(self, aligner):
        self.aligner = aligner
        
        
class PathwayEnrichmentAnalyser:
    
    def __init__(self, aligner):
        self.aligner = aligner
        
    def make_ranked_genelist(self):
        if hasattr(self, 'x'):
            return
        #print('make ranked gene list')
        matched_percentages = {}
        for al_obj in self.aligner.results:
            matched_percentages[al_obj.gene] = (al_obj.get_series_match_percentage()[0]/100 )
        x = sorted(matched_percentages.items(), key=lambda x: x[1], reverse=False)
        self.x = x 
        return x
        
    # runs gprofiler overrepresentation analysis over the top k DE genes (k decided on matched percentage threshold) 
    def run_gprofiler(self, DIFF_THRESHOLD=0.5):
        self.make_ranked_genelist()
        top_k = np.unique(pd.DataFrame(self.x)[1] < DIFF_THRESHOLD , return_counts=True)[1][1]
        print(top_k, ' # of DE genes to check')
        clusters = pd.DataFrame([pd.DataFrame(self.x)[0:top_k][0], np.repeat(0,top_k)]).transpose()
        clusters.columns = ['Gene','ClusterID']
        clusters = clusters.set_index('Gene')
        self.gprofiler_results = sc.queries.enrich(list(clusters.index))
        return self.gprofiler_results
    
    # runs gprofiler overrepresentationa analysis for each cluster 
    def run_gprofiler_for_alignment_clusters(self):
        
        self.make_ranked_genelist()
        #list(clusters.index)
        gprofiler_results_all = []
        for cluster_id in tqdm(range(len(self.aligner.gene_clusters))):
            #print('processing cluster: ', cluster_id)
            temp = pd.DataFrame(self.x)
            temp = temp.set_index(0)
            temp = temp.filter(items = self.aligner.gene_clusters[cluster_id], axis=0)
            clusters = temp
            clusters['Gene'] = clusters.index
            clusters.columns = ['ClusterID','Gene']
            clusters['ClusterID'] = np.repeat(cluster_id, len(clusters))
            #print('# of genes: ', len(clusters))
            #print(list(clusters.index))
            gprofiler_results = sc.queries.enrich(list(clusters.index))
            gprofiler_results_all.append(gprofiler_results)
            
        return gprofiler_results_all
    
    # KEGG and REAC focused 
    def run_cluster_enrichment(self):
    
        gprofiler_results_all = self.run_gprofiler_for_alignment_clusters()
        cluster_overrepresentation_results = []
        for cluster_id in range(len(self.aligner.gene_clusters)):
            _kegg=gprofiler_results_all[cluster_id][gprofiler_results_all[cluster_id].source == 'KEGG']
            _reac=gprofiler_results_all[cluster_id][gprofiler_results_all[cluster_id].source == 'REAC']
            if(len(_kegg)>0 or len(_reac)>0):
               # print('Cluster:', cluster_id, ' ----- ', len(self,aligner.gene_clusters[cluster_id])) 
                n_genes = len(self.aligner.gene_clusters[cluster_id])
                if(n_genes<15):
                    genes = self.aligner.gene_clusters[cluster_id]
                else:
                    genes = self.aligner.gene_clusters[cluster_id][1:7] + [' ... '] +  self.aligner.gene_clusters[cluster_id][n_genes-7:n_genes]

                cluster_overrepresentation_results.append([cluster_id,len(self.aligner.gene_clusters[cluster_id]),genes,np.asarray(_kegg.name) ,np.asarray(_reac.name) ])
        self.results = pd.DataFrame(cluster_overrepresentation_results)
        print(tabulate(self.results,  headers=['cluster_id','n_genes', 'geneset', 'KEGG_pathways','REACTOME_pathways'],tablefmt="grid",maxcolwidths=[3, 3, 3,30,50,50])) 
        
    def write_results_csv(self, file_name, write_markdown=False):
            self.results.to_csv(file_name + '.csv')
            if(write_markdown):
                self.results.to_markdown(file_name + '.md')
                
                
    def get_cluster_table(self, write_file=False):
        self.aligner.compute_cluster_MVG_alignments(MVG_MODE_KL=False)
        self.aligner.cluster_pathway_results = self.results
        al_visuals = []
        c1 = []; c2 = []; c3 = []
        for cluster_id in self.results[0]:
            mvg_obj = self.aligner.mvg_cluster_average_alignments[cluster_id]
            al_str = mvg_obj.al_visual
            al_str = al_str.replace('5-state string','')
            al_str = al_str.replace('Alignment index','')
            al_str = al_str.replace('Reference index','')
            al_str = al_str.replace('Query index','')
            al_visuals.append(al_str) 
            c1.append(mvg_obj.get_series_match_percentage()[0])
            c2.append(mvg_obj.get_series_match_percentage()[1])
            c3.append(mvg_obj.get_series_match_percentage()[2])
        self.aligner.cluster_pathway_results[5] =  al_visuals
        self.aligner.cluster_pathway_results[6] = c1
        self.aligner.cluster_pathway_results[7] = c2
        self.aligner.cluster_pathway_results[8] = c3
        print(tabulate(self.aligner.cluster_pathway_results,  headers=['cluster','n_genes', 'geneset', 'KEGG_pathways','REACTOME_pathways','cell-level alignment', 'A%','S%','T%'],
                       tablefmt="grid",maxcolwidths=[None,None,None,25,25,25,80])) 

        if(write_file):
            self.aligner.cluster_pathway_results.to_markdown('cluster_info' + '.md')
        
    
# GOLDRATH_NAIVE_VS_EFF_CD8_TCELL_DN
# GOBP_INTERFERON_ALPHA_PRODUCTION



