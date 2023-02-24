<p align="left"><img src="G2G_logo.png"></p>

# Genes2Genes
## A new framework for aligning single-cell trajectories of gene expression 
G2G aims to guide downstream comparative analysis of single-cell reference and query systems along any axis of progression (e.g. pseudotime), 
using a new dynamic programming alignment algorithm (which unifies both matches and mismatches). 

You can use this framework to perform comparisons such as:
<ul>
    <li>Organoid vs. Reference tissue
    <li>Control vs. Treatment
    <li>Healthy vs. Disease
</ul>   

by inferring fully-descriptive gene-specific alignments and single-aggregate alignments, allowing to identify and pinpoint dynamic similarities and differences in transcriptomics between a reference and query. 

### Manuscript preprint 
*"Gene-level alignment of single cell trajectories informs the progression of in vitro T cell differentiation"* <br>
Authors: Dinithi Sumanaweera†, Chenqu Suo†, Daniele Muraro, Emma Dann, Krzysztof Polanski, Alexander S. Steemers, Jong-Eun Park, Bianca Dumitrascu, Sarah A. Teichmann* <br>
Available at: TBD 

### **Installing G2G**

Download the GitHub repository and create a new Conda environment with the Genes2Genes dependencies:
```
git clone https://github.com/Teichlab/Genes2Genes
cd Genes2Genes
conda env create -f environment.yaml
conda activate genes2genes-env
python -m ipykernel install --user --name genes2genes-env
```
This will also register your environment for use with Jupyter. Once you do so, write jupyter notebook, then navigate to the notebooks folder to [get started](https://github.com/Teichlab/Genes2Genes/blob/main/notebooks/G2G_Tutorial.ipynb)!

Note: Package TBA

### **Input to Genes2Genes**
(1) Reference anndata (with `adata_ref.X` storing log1p normalised gene expression), 
(2) Query anndata (with `adata_query.X` storing log1p normalised gene expression), and
(3) Pseudotime estimates stored in each anndata object under  `adata_ref.obs['time']` and `adata_query.obs['time']`.

### Tutorial

Please refer to the notebook `G2G_Tutorial.ipynb` which gives an example analysis between a reference and query dataset from literature. 

