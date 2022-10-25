# Genes2Genes
### A new algorithm and framework for single-cell trajectory alignment 

Genes2Genes aims to guide downstream comparative analysis of reference and query systems along any axis of progression (e.g. pseudotime) through both gene-level and cell-level alignment. 

You can use this framework to perform comparisons such as:
<ul>
    <li>Organoid vs. Reference tissue
    <li>Healthy vs. Disease
    <li>Control vs. Treatment
</ul>   

#### Input to Genes2Genes
(1) Reference adata (with `adata_ref.X` storing log1p normalised expression), 
(2) Query adata (with `adata_query.X` storing log1p normalised expression), and
(3) Pseudotime estimates stored in each adata object under  `adata.obs['time']`.

**<span style="color:red">Note: This is the initial and testable version of Genes2Genes (in confidence -- still unpublished and under refinement) so you might run into unforseen errors and bugs. Please do let me know so that I can correct them before releasing the stable version. Thanks!</span>**

### Tutorial

Please refer to the tutorial notebook which gives steps to analyse an example reference and query dataset: 2 treatment groups of mouse-bone-marrow-derived Dendritic cells from Shalek et al (2014). The respective single-cell datasets along with pseudotime estimates were downloaded from CellAlign (Alpert et al 2018) and packaged into adata objects. 

