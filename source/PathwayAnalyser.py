from gsea_api.molecular_signatures_db import MolecularSignaturesDatabase
import enum
import numpy as np

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
        
# GOLDRATH_NAIVE_VS_EFF_CD8_TCELL_DN
# GOBP_INTERFERON_ALPHA_PRODUCTION



