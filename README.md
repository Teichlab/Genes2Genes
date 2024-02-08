<p align="left"><img src="G2G_logo.png"></p>

# Genes2Genes
Project page: https://teichlab.github.io/Genes2Genes

## A new framework for aligning single-cell trajectories of gene expression 
G2G aims to guide downstream comparative analysis of single-cell reference and query systems along any axis of progression (e.g. pseudotime). 
This is done by employing a new dynamic programming (DP) based alignment algorithm which unifies dynamic time warping (DTW) and gap modelling to capture both matches and mismatches between time points. Our DP algorithm 
incorporates a Bayesian information-theoretic scoring scheme with a five-state probabilistic machine to generate an optimal alignment between a reference trajectory and query trajectory of a given gene in terms of their scRNA-seq expression. 

We can use the G2G framework to perform comparisons across pseudotime such as:
<ul>
    <li>Organoid vs. Reference tissue
    <li>Control vs. Treatment
    <li>Healthy vs. Disease
</ul>  
by inferring fully-descriptive gene-specific alignments and single-aggregate alignments. 
These alignment results enable us to pinpoint dynamic similarities and differences in gene expression between a reference and query, as well as to group genes with similar alignment patterns.  

### Manuscript preprint 
***"Gene-level alignment of single cell trajectories"*** <br>
**Authors**: Dinithi Sumanaweera†, Chenqu Suo†, Ana-Maria Cujba, Daniele Muraro, Emma Dann, Krzysztof Polanski, Alexander S. Steemers, Woochan Lee, Amanda J. Oliver, Jong-Eun Park, Kerstin B. Meyer, Bianca Dumitrascu, Sarah A. Teichmann* <br>
Available at: https://doi.org/10.1101/2023.03.08.531713 

### **Installing G2G**

For now, G2G needs to be installed from GitHub in a Python >=3.8 environment by running the following:
```bash
pip install git+https://github.com/Teichlab/Genes2Genes.git
```
The package will be made available on PyPi soon.

### **Input to G2G**
(1) Reference anndata object (with `adata_ref.X` storing log1p normalised gene expression), 
(2) Query anndata object (with `adata_query.X` storing log1p normalised gene expression), and
(3) Pseudotime estimates stored in each anndata object under `adata_ref.obs['time']` and `adata_query.obs['time']`.

### Tutorial

Please refer to the notebook [`notebooks/Tutorial.ipynb`](https://github.com/Teichlab/Genes2Genes/blob/main/notebooks/Tutorial.ipynb) which gives an example analysis between a reference and query dataset from literature. <br>

**Note**: The runtime of the G2G algorithm depends on the number of cells in the reference and query datasets, the number of interpolation time points, and the number of genes to align. 
G2G currently utilizes concurrency through Python multiprocessing to speed up the gene-level alignment process. It creates a number of processes equal to the number of cores in the system, and each process performs a single gene-level alignment at one time. 


### Funding Acknowledgement
Marie Skłodowska-Curie grant agreement No: 101026506 (Marie Curie Individual Fellowship) under the European Union’s Horizon 2020 research and innovation programme; Wellcome Trust Ph.D. Fellowship for Clinicians; Wellcome Trust (WT206194); ERC Consolidator Grant (646794); Wellcome Sanger Institute’s Translation Committee Fund.
