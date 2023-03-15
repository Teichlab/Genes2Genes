<p align="left"><img src="G2G_logo.png"></p>

# Genes2Genes
## A new framework for aligning single-cell trajectories of gene expression 
G2G aims to guide downstream comparative analysis of single-cell reference and query systems along any axis of progression (e.g. pseudotime). 
This is done by employing a new dynamic programming (DP) based alignment algorithm which unifies both matches and mismatches. Our DP algorithm 
incorporates a Bayesian information-theoretic scoring scheme with a five-state probabilistic machine to generate an optimal alignment between a reference trajectory and
query trajectory of a given gene in terms of their scRNA expression. 

We can use the G2G framework to perform comparisons across pseudotime such as:
<ul>
    <li>Organoid vs. Reference tissue
    <li>Control vs. Treatment
    <li>Healthy vs. Disease
</ul>  
by inferring fully-descriptive gene-specific alignments and single-aggregate alignments. 
These alignment results enable us to pinpoint dynamic similarities and differences in gene expression between a reference and query, as well as to group genes with similar alignment patterns.  

### Manuscript preprint 
***"Gene-level alignment of single cell trajectories informs the progression of in vitro T cell differentiation"*** <br>
**Authors**: Dinithi Sumanaweera†, Chenqu Suo†, Daniele Muraro, Emma Dann, Krzysztof Polanski, Alexander S. Steemers, Jong-Eun Park, Bianca Dumitrascu, Sarah A. Teichmann* <br>
Available at: https://www.biorxiv.org/content/10.1101/2023.03.08.531713v1  

### **Installing G2G**

For now, G2G needs to be installed from GitHub:
```bash
pip install git+https://github.com/Teichlab/Genes2Genes.git
```
The package will be made available on PyPi soon.

### **Input to G2G**
(1) Reference anndata object (with `adata_ref.X` storing log1p normalised gene expression), 
(2) Query anndata object (with `adata_query.X` storing log1p normalised gene expression), and
(3) Pseudotime estimates stored in each anndata object under `adata_ref.obs['time']` and `adata_query.obs['time']`.

### Tutorial

Please refer to the notebook [`notebooks/G2G_Tutorial.ipynb`](https://github.com/Teichlab/Genes2Genes/blob/main/notebooks/G2G_Tutorial.ipynb) which gives an example analysis between a reference and query dataset from literature. 



