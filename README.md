# Genes2Genes (G2G)
### A new framework for single-cell gene expression trajectory alignment 

G2G aims to guide downstream comparative analysis of reference and query systems along any axis of progression (e.g. pseudotime), 
using dynamic programming alignment (that handles both matches and mismatches). 

You can use this framework to perform comparisons such as:
<ul>
    <li>Organoid vs. Reference tissue
    <li>Healthy vs. Disease
    <li>Control vs. Treatment
</ul>   

#### Input to Genes2Genes
(1) Reference anndata (with `adata_ref.X` storing log1p normalised expression), 
(2) Query anndata (with `adata_query.X` storing log1p normalised expression), and
(3) Pseudotime estimates stored in each anndata object under  `adata.obs['time']`.

**<span style="color:red">Note: This is the initial and testable version of G2G (IN CONFIDENCE -- Manuscript in Progress)</span>**

### Tutorial

Please refer to the Tutorial Notebook which gives an example analysis between a reference and query dataset: 2 treatment groups of mouse-bone-marrow-derived Dendritic cells from Shalek et al (2014). The respective single-cell datasets along with their pseudotime estimates were downloaded from CellAlign (Alpert et al 2018) and packaged into adata objects. 

