# 🧬 Beyond Averages: Understanding Cell-Type Specific Adaptations with Single-Cell RNA Sequencing
Single-cell RNA sequencing (scRNA-seq) has transformed our understanding of cellular biology by allowing us to peer into the transcriptional machinery of individual cells with extraordinary precision. When I first learned about next generation sequencing technologies, we were limited to bulk measurements that averaged signals aross millions of cells, blurring the boundaries between distinct cell types and states. Today, however, single-cell sequencing technologies allow us to resolve the unique transcriptional fingerprint of each cell in complex tissues, revealing a level of heterogeneity that was previously invisible to us.

This level of resolution helps up answer questions that were previously impossible to address: How do rare cell populations respond to treatment? What cellular communication networks exist within tissues? How do cells transition between states during development?

The technology isn't just an incremental improvement - it's a fundamental shift in our approach to biology, trading averaged signals for a high-resolution view of the cellular landscape, much like how modern microscopy revealed subcellular structures invisible to earlier scientists. In this guide, you’ll how to work with scRNA-seq data, including unique aspects of data processing specific to these analyses, as well as how to identify genes that are differentially expressed between cell-types, or across experimental conditions within a given cell type. Let's dive in!

## 🧬 The Conceptual Leap - From Bulk to Single-Cell Analysis
### Bulk RNA-Sequencing Basics
For years, bulk RNA sequencing has been essential for transcriptomics research, helping us understand gene expression differences between experimental conditions. When working with bulk RNA-seq data from repositories like GEO, the process typically follows these steps:

- Organization: Arrange counts into dataframes with genes as rows and samples as columns. This structure addresses our main question: "How does gene expression vary between samples?"
- Preprocessing: Convert Ensembl IDs to gene symbols, handle missing values, and perform quality checks.
- Normalization: After preprocessing the data and performing quality control we need to normalize counts across samples. I prefer a DESeq2-style approach that uses counts per million to filter out low-expressed genes and applies geometric means to account for library size differences, which helps mitigate technical variation between samples.
- Differential expression analysis: Identify statistically significant differences between conditions using methods like Welch's t-test and multiple testing correction.
- Downstream analysis: Explore the biological meaning through [co-expression networks analysis](https://github.com/evanpeikon/co_expression_network), [protein-protein interaction mapping](https://github.com/evanpeikon/PPI_Network_Analysis), and [functional enrichment analysis](https://github.com/evanpeikon/functional_enrichment_analysis).


### Single-Cell RNA-Seq - A Complete Perspective Shift
Single-cell RNA-seq isn't just an upgrade to bulk RNA-seq — it completely transforms how we think about transcriptomics. While we're still investigating gene expression and biological function, almost everything about the workflow changes. The most significant difference is in data structure. In bulk RNA-seq, we analyze genes across conditions. In single-cell work, we examine cells across genes. This flips our perspective entirely—cell identity, not experimental condition, becomes the primary focus. Our matrix inverts (cells as rows, genes as columns), requiring specialized data structures like AnnData to manage the complexity.

Additionaly, quality control also takes on new dimensions. Instead of evaluating samples, we assess thousands of individual cells and examine metrics like genes per cell, UMI counts, and mitochondrial percentage. We alsop filter out stressed cells with high mitochondrial content, doublets, and empty droplets, all of which are unique to single-cell analysis.

Normalization becomes more complex too. Single-cell data is incredibly sparse—most genes in most cells show zero counts due to technical limitations. This requires specialized approaches for handling zero-inflation and dropout events. Furthermore, dimensionality reduction shifts from optional to essential. With thousands of cells and genes creating a high-dimensional space, techniques like PCA, t-SNE, and UMAP become crucial for visualization and analysis.

Cell type identification emerges as an entirely new step. We cluster similar cells and determine their biological identity using marker genes and reference databases, transforming abstract numerical clusters into meaningful entities like "CD8+ T cells." Finally, differential expression analysis gains new dimensions. Instead of simply comparing conditions, we can now compare:
- Different cell types within a condition
- The same cell type across conditions
- Changes in cell type proportions between conditions

This cellular resolution reveals signals that would be completely missed in bulk analysis. A gene upregulated in rare cells might show no change in bulk samples, where abundant cell types mask the signal. The shift from bulk to single-cell RNA-seq is like moving from measuring average ocean temperatures to mapping detailed thermal profiles at thousands of specific points. The computational complexity increases dramatically—but so does our ability to understand the currents and patterns driving biological systems.

## 🧬 The Technical Foundation: AnnData and Specialized Data Structures
The complexity of single-cell data necessitated new computational frameworks. The AnnData ("Annotated Data") format emerged as a powerful solution, organizing the multidimensional aspects of single-cell data into a coherent structure, as demonstrated below. Unlike conventional data frames, AnnData objects store not just the gene expression matrix, but also cell metadata, gene annotations, dimensional reductions, and nearest-neighbor graphs – all essential components for sophisticated analyses.

<img width="375" alt="Screenshot 2025-04-08 at 12 33 01 PM" src="https://github.com/user-attachments/assets/b5c9b290-55d1-4501-83e6-0c46cb534613" />

Let's examine a function to download and process scRNA-seq data from public repositories, using the data made avaialble by a study titled [Single-cell sequencing deconvolutes cellular responses to exercise in human skeletal muscle](https://www.nature.com/articles/s42003-022-04088-z) as an example (raw and processed sequencing data from this study is  available through Gene-Expression omnibus accession number GSE214544):

```python
def download_and_process_scRNA_data(samples: List[Dict[str, str]], output_dir: str = "data") -> an.AnnData:
    """Downloads and processes scRNA-seq data files from 10X Genomics into a single AnnData object."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    adatas = []
    labels = []
    
    for sample in samples:
        # Extract sample info
        accession = sample['accession']
        file_prefix = sample['file_prefix']
        label = sample['label']
        
        # Define file paths
        files = {
            'matrix': f"{file_prefix}_matrix.mtx",
            'barcodes': f"{file_prefix}_barcodes.tsv",
            'features': f"{file_prefix}_features.tsv"
        }
        file_paths = {k: Path(output_dir) / v for k, v in files.items()}
        
        # Download and decompress each file if needed
        for file_type, target_file in file_paths.items():
            gz_file = target_file.with_suffix(target_file.suffix + '.gz')
            
            # Skip if target file exists
            if target_file.exists():
                print(f"File {target_file} already exists, skipping download")
                continue
                
            # Download if needed
            if not gz_file.exists():
                ext = 'mtx' if file_type == 'matrix' else 'tsv'
                url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={accession}&format=file&file={file_prefix}_{file_type}.{ext}.gz"
                url = url.replace('_', '%5F').replace('.', '%2E')
                print(f"Downloading {file_type} file for {accession}...")
                urlretrieve(url, gz_file)
            
            # Decompress file
            print(f"Decompressing {gz_file}...")
            with gzip.open(gz_file, 'rb') as f_in, open(target_file, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Create AnnData object
        adata = sc.read_mtx(file_paths['matrix']).T
        
        # Add barcodes with unique sample prefix
        barcodes = pd.read_csv(file_paths['barcodes'], header=None)[0]
        adata.obs_names = [f"{label}_{bc}" for bc in barcodes]
        
        # Add feature information
        features = pd.read_csv(file_paths['features'], sep='\t', header=None)
        adata.var_names = features[0]  # Ensembl IDs
        adata.var['gene_ids'] = features[1].values  # Gene symbols
        
        # Add feature types if available
        if features.shape[1] > 2:
            adata.var['feature_type'] = features[2].values
            
        # Add sample metadata
        for k, v in sample.items():
            if k not in ['accession', 'file_prefix']:
                adata.obs[k] = v
        
        adatas.append(adata)
        labels.append(label)
    
    # Combine AnnData objects
    adata_combined = an.concat(adatas, axis=0, join='outer', label='batch', keys=labels)
    
    # Make observation names unique
    adata_combined.obs_names_make_unique()
    
    # Use gene symbols instead of Ensembl IDs
    features_file = Path(output_dir) / f"{samples[0]['file_prefix']}_features.tsv"
    features_df = pd.read_csv(features_file, sep='\t', header=None)
    ensembl_to_symbol = dict(zip(features_df[0], features_df[1]))
    adata_combined.var_names = [ensembl_to_symbol.get(id, id) for id in adata_combined.var_names]
    adata_combined.var_names_make_unique()
    
    # Identify mitochondrial genes
    adata_combined.var['mt'] = adata_combined.var_names.str.startswith('MT-')
    print(f"Created AnnData object with {adata_combined.shape[0]} cells and {adata_combined.shape[1]} genes")
    return adata_combined
```
```
Downloading matrix file for GSM6611295...
Decompressing data/GSM6611295_P15306_5001_matrix.mtx.gz...
Downloading barcodes file for GSM6611295...
Decompressing data/GSM6611295_P15306_5001_barcodes.tsv.gz...
Downloading features file for GSM6611295...
Decompressing data/GSM6611295_P15306_5001_features.tsv.gz...
Downloading matrix file for GSM6611296...
Decompressing data/GSM6611296_P15306_5002_matrix.mtx.gz...
Downloading barcodes file for GSM6611296...
Decompressing data/GSM6611296_P15306_5002_barcodes.tsv.gz...
Downloading features file for GSM6611296...
Decompressing data/GSM6611296_P15306_5002_features.tsv.gz...
Downloading matrix file for GSM6611297...
Decompressing data/GSM6611297_P14601_4004_matrix.mtx.gz...
Downloading barcodes file for GSM6611297...
Decompressing data/GSM6611297_P14601_4004_barcodes.tsv.gz...
Downloading features file for GSM6611297...
Decompressing data/GSM6611297_P14601_4004_features.tsv.gz...
Downloading matrix file for GSM6611298...
Decompressing data/GSM6611298_P14601_4005_matrix.mtx.gz...
Downloading barcodes file for GSM6611298...
Decompressing data/GSM6611298_P14601_4005_barcodes.tsv.gz...
Downloading features file for GSM6611298...
Decompressing data/GSM6611298_P14601_4005_features.tsv.gz...
Downloading matrix file for GSM6611299...
Decompressing data/GSM6611299_P15306_5003_matrix.mtx.gz...
Downloading barcodes file for GSM6611299...
Decompressing data/GSM6611299_P15306_5003_barcodes.tsv.gz...
Downloading features file for GSM6611299...
Decompressing data/GSM6611299_P15306_5003_features.tsv.gz...
Downloading matrix file for GSM6611300...
Decompressing data/GSM6611300_P15306_5004_matrix.mtx.gz...
Downloading barcodes file for GSM6611300...
Decompressing data/GSM6611300_P15306_5004_barcodes.tsv.gz...
Downloading features file for GSM6611300...
Decompressing data/GSM6611300_P15306_5004_features.tsv.gz...
Created AnnData object with 37008 cells and 33538 genes
```

This function accomplishes several critical tasks. It downloads the sparse matrix, barcodes, and features files typically generated by 10X Genomics platforms, and assembles them into an AnnData object. Importantly, it preserves sample metadata, converts Ensembl IDs to more readable gene symbols, and identifies mitochondrial genes – a crucial quality control metric. The function also elegantly handles multiple samples, combining them into a unified dataset while maintaining their batch identities.

## 🧬 Quality Control - Separating Signal from Noise
In single-cell experiments, quality control isn't merely a preliminary step – it's a fundamental determinant of analytical success. Unlike bulk sequencing, where RNA quality reflects an average across millions of cells, scRNA-seq captures the transcriptional state of cells that may be stressed, dying, or compromised during sample preparation.

Let's examine a comprehensive QC workflow:

```python
# Check for NaN values (with handling for sparse matrices)
if sp.issparse(adata.X):
    print("Sparse matrix detected, checking for invalid values in non-zero elements...")
    if np.isnan(adata.X.data).any():
        print(f"Number of NaNs in non-zero elements: {np.sum(np.isnan(adata.X.data))}")
    else:
        print("No NaN values found in non-zero elements.")
else:
    nan_rows = np.isnan(adata.X).any(axis=1)
    nan_cols = np.isnan(adata.X).any(axis=0)
    print(f"Number of rows with NaN values: {np.sum(nan_rows)}")
    print(f"Number of columns with NaN values: {np.sum(nan_cols)}")

# Calculate comprehensive QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=[], percent_top=None, log1p=False, inplace=True)
# Identify and calculate mitochondrial content
mt_gene_mask = adata.var_names.str.startswith(('MT-', 'mt-'))
mt_count = np.sum(mt_gene_mask)
print(f"Found {mt_count} mitochondrial genes")
if mt_count > 0:
    adata.var['mt'] = mt_gene_mask
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    print("Mitochondrial metrics calculated.")
else:
    print("No mitochondrial genes found with standard prefixes.")
    adata.obs['pct_counts_mt'] = 0
```
```
Sparse matrix detected, checking for invalid values in non-zero elements...
No NaN values found in non-zero elements.
```

The code above checks for missing values – a critical step given the sparse nature of single-cell data – and calculates key quality metrics including genes per cell, UMIs per cell, and percentage of mitochondrial reads. Elevated mitochondrial content often indicates cellular stress or death, a phenomenon we simply couldn't detect in bulk analysis.

The visualization component is equally important:

```python
# Visualize QC metrics 
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# Distribution of genes per cell
sns.histplot(adata.obs['n_genes_by_counts'], bins=50, kde=True, ax=axes[0, 0])
axes[0, 0].set_xlabel('Number of Genes per Cell')
axes[0, 0].set_ylabel('Number of Cells')
axes[0, 0].set_title('Distribution of Number of Genes per Cell')
axes[0, 0].axvline(200, color='red', linestyle='--', label='Filter threshold (200)')
axes[0, 0].legend()

# Distribution of UMI counts per cell
sns.histplot(adata.obs['total_counts'], bins=50, kde=True, ax=axes[0, 1])
axes[0, 1].set_xlabel('Total UMI Counts per Cell')
axes[0, 1].set_ylabel('Number of Cells')
axes[0, 1].set_title('Distribution of UMI Counts per Cell')
axes[0, 1].set_xscale('log')

# Distribution of mitochondrial gene percentage
sns.histplot(adata.obs['pct_counts_mt'], bins=50, kde=True, ax=axes[1, 0])
axes[1, 0].set_xlabel('Percentage of Mitochondrial Genes')
axes[1, 0].set_ylabel('Number of Cells')
axes[1, 0].set_title('Distribution of Mitochondrial Gene % per Cell')
axes[1, 0].axvline(20, color='red', linestyle='--', label='Typical threshold (20%)')
axes[1, 0].legend()

# Scatter plot of UMI count vs genes per cell colored by mito percent
scatter = axes[1, 1].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], c=adata.obs['pct_counts_mt'], cmap='viridis', s=10, alpha=0.7)
axes[1, 1].set_xlabel('Total UMI Counts per Cell')
axes[1, 1].set_ylabel('Number of Genes per Cell')
axes[1, 1].set_title('Genes vs UMIs, Colored by Mito%')
axes[1, 1].set_xscale('log')
cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_label('Mitochondrial %')
plt.tight_layout()
plt.show()
```

![Unknown-3](https://github.com/user-attachments/assets/311428d2-d9ad-47f3-9095-f7bbcc27a1d6)

The four-panel display above captures key metrics that guide our filtering decisions, with patterns that tell a compelling biological story. In the upper left, we see the distribution of genes detected per cell, with a pronounced right-skewed distribution. The red dashed line at 200 genes marks a common filtering threshold. The UMI count distribution (upper right) similarly shows most cells with counts between 10³ and 10⁴, with a long tail of high-count cells.

The mitochondrial gene percentage (lower left) presents a quality control metric unique to single-cell analysis. While most cells show healthy levels under 10%, we observe a small population with elevated mitochondrial content approaching our 20% threshold—these likely represent stressed or dying cells that require careful consideration.

Perhaps most informative is the scatter plot (lower right) showing the relationship between genes and UMI counts, colored by mitochondrial percentage. The strong correlation between genes and UMIs follows the expected biological relationship, but several interesting patterns emerge. The dense cluster of cells with high mitochondrial content (lighter colors) at the bottom of the distribution likely represents dying cells, while the small population of cells with unusually high gene and UMI counts may represent doublets.

What's fascinating about single-cell data is that appropriate filtering thresholds vary significantly between tissues, experimental protocols, and biological questions. A naive application of standard cutoffs might eliminate valuable biological signal—for instance, some neuronal populations naturally express fewer genes than immune cells, and overly stringent filtering could selectively remove these important populations. These visualizations allow us to make informed decisions about quality thresholds tailored to the specific biological context of our experiment.

Beyond examining overall quality metrics, we must also consider how these parameters vary across experimental batches—a critical step that can reveal technical artifacts before they contaminate downstream analysis. To investigate potential batch effects, we generate the following violin plots comparing key quality metrics across samples:

```python
# Visualize QC metrics by sample to identify batch effects
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# UMI counts by batch
sc.pl.violin(adata, 'total_counts', groupby='batch', ax=axes[0], show=False)
axes[0].set_title('UMI Counts by Sample')
axes[0].set_ylabel('Total UMI Counts')
axes[0].set_yscale('log')

# Genes per cell by batch
sc.pl.violin(adata, 'n_genes_by_counts', groupby='batch', ax=axes[1], show=False)
axes[1].set_title('Genes per Cell by Sample')
axes[1].set_ylabel('Number of Genes')

# Mitochondrial % by batch
sc.pl.violin(adata, 'pct_counts_mt', groupby='batch', ax=axes[2], show=False)
axes[2].set_title('Mitochondrial % by Sample')
axes[2].set_ylabel('Mitochondrial %')
plt.tight_layout()
plt.show()
```

![Unknown-4](https://github.com/user-attachments/assets/110b3968-8971-4b7b-b8c0-6b98cef3cb61)

After visualizing the data, we apply carefully considered filters:

```python
# Set filtering thresholds based on QC metrics
min_genes = 300  # Minimum genes per cell
min_cells = 20   # Minimum cells per gene
max_mito = 15    # Maximum percentage of mitochondrial genes (adjust based on your data)
min_counts = 1000  # Minimum UMI counts per cell
max_counts = 15000  # Maximum UMI counts 

print(f"Before filtering: {adata.n_obs} cells, {adata.n_vars} genes")

# Filter cells based on QC metrics
adata_filtered = adata[(adata.obs['n_genes_by_counts'] >= min_genes) & (adata.obs['pct_counts_mt'] <= max_mito)]

if min_counts:
    adata_filtered = adata_filtered[adata_filtered.obs['total_counts'] >= min_counts]

if max_counts:
    adata_filtered = adata_filtered[adata_filtered.obs['total_counts'] <= max_counts]

# Filter genes based on minimum cells
sc.pp.filter_genes(adata_filtered, min_cells=min_cells)
print(f"After filtering: {adata_filtered.n_obs} cells, {adata_filtered.n_vars} genes")
```
```
Before filtering: 37008 cells, 33538 genes
After filtering: 10809 cells, 14400 genes
```

This filtering approach removes cells that are likely of poor quality (too few genes, too high mitochondrial content) and genes that are detected in too few cells to be informative. The upper limit on UMI counts helps remove potential doublets – a uniquely single-cell artifact where two cells are captured together and incorrectly treated as one.

## 🧬 Normalization: Accounting for Technical Variation
Single-cell data presents distinctive normalization challenges that demand specialized approaches tailored to its unique characteristics. Unlike bulk RNA-seq, single-cell data is extraordinarily sparse and zero-inflated—most genes in most cells register zero counts, primarily due to technical limitations in capturing and detecting transcripts rather than true biological absence.

```python
# Normalization and log transformation
print("Normalizing data...")
sc.pp.normalize_total(adata_filtered, target_sum=1e4)
sc.pp.log1p(adata_filtered)

# Highly variable gene selection with improved parameters
print("Identifying highly variable genes...")
sc.pp.highly_variable_genes(adata_filtered, n_top_genes=2000, subset=True)
```
Our normalization workflow addresses these challenges through the following approach: first, we scale each cell's expression values to a common denominator (10,000 counts) with ```sc.pp.normalize_total()```, effectively neutralizing differences in sequencing depth between cells without overcorrecting for biological differences in RNA content. We then apply log-transformation with ```sc.pp.log1p()```, which stabilizes variance across the expression spectrum and compresses the extreme dynamic range typical of gene expression data. Finally, we select the 2,000 most highly variable genes with ```sc.pp.highly_variable_genes()```, focusing our analysis on genes with genuine biological signal while filtering out the vast majority of genes showing primarily technical noise or housekeeping patterns. This three-step approach—scaling, transformation, and feature selection—creates a foundation for dimensionality reduction and clustering that emphasizes biological variation while mitigating the confounding effects of technical artifacts and random noise.

## 🧬 Dimensionality Reduction: Finding Structure in High-Dimensional Space
The dimensionality of single-cell data is staggering – thousands of genes measured across thousands of cells. Effective dimensionality reduction is not merely a computational convenience but a biological necessity, helping us separate signal from noise and visualize complex cellular relationships:

```python
# apply z-transformation
sc.pp.scale(adata_filtered, zero_center=True)
# perform dimensionality reduction via PCA
sc.tl.pca(adata_filtered, svd_solver='arpack')

# construct graph of nearest neighbors
sc.pp.neighbors(adata_filtered, n_neighbors=20, n_pcs=30)
# apply leiden clustering algorithm
sc.tl.leiden(adata_filtered, key_added='clusters', resolution=0.3, n_iterations=3, flavor='igraph', directed=False)
# create and visualize UMAP
sc.tl.umap(adata_filtered)
sc.pl.umap(adata_filtered, color='clusters', add_outline=True, legend_loc='on data', legend_fontsize=12, legend_fontoutline=2, frameon=True)

labels = adata_filtered.obs['clusters']
sil_score = silhouette_score(adata_filtered.obsm['X_pca'], labels)
print(f'Silhouette Score: {sil_score}')
```
```
Silhouette Score: 0.15531152486801147
```
![Unknown-5](https://github.com/user-attachments/assets/75d35eb4-06bb-4ad1-84eb-8d96c9ef4483)

Our dimensional reduction and clustering pipeline begins with gene-wise scaling (```sc.pp.scale```), ensuring that highly-expressed genes don't disproportionately influence our analysis—this creates a level playing field where rare transcription factors can compete with abundant housekeeping genes. The subsequent PCA step accomplishes four critical tasks simultaneously: filtering technical noise, addressing the "curse of dimensionality" that would otherwise distort distance calculations, revealing the primary axes of biological variation, and dramatically reducing computational requirements. By retaining 30 principal components, we capture the substantive biological signal while discarding the random variation that would otherwise obscure cellular relationships.

With this dimensionally-reduced representation, we construct a nearest-neighbor graph with 20 neighbors per cell. This graph serves as the substrate for the Leiden community detection algorithm, which identifies 12 distinct clusters at a resolution of 0.3—a relatively conservative setting that prevents over-fragmentation of cell types. The resulting clusters, visualized through UMAP, reveal remarkable structure, with clearly separated populations representing distinct cell types.

The silhouette score of 0.155 provides a quantitative assessment of clustering quality. While this value may seem low, it's actually quite reasonable for single-cell data, where biological continuums and transitional states often create cluster boundaries that are inherently blurred. The UMAP visualization confirms this quality assessment, showing well-defined clusters with minimal overlap—the hallmark of successful dimensionality reduction and clustering in the complex landscape of single-cell transcriptomics.

## 🧬 Cell Type Identification: From Clusters to Biology
Identifying cell types is where computational analysis meets biological interpretation. We first identify genes that are differentially expressed between clusters:

```python
sc.tl.rank_genes_groups(adata_filtered, groupby='clusters', method='wilcoxon', corr_method='bonferroni')
top_markers = sc.get.rank_genes_groups_df(adata_filtered, group=None)
top_markers_df = pd.DataFrame(top_markers)

# initialize a dictionary to store top markers for each cluster
top_genes_per_cluster = {}
# store list of clusters
clusters = adata_filtered.uns['rank_genes_groups']['names'].dtype.names
# iterate over each cluster to get top markers and store them in top_genes_per_cluster dictioary
for cluster in clusters:
    top_genes = top_markers_df[top_markers_df['group'] == cluster].head(3)
    top_genes_per_cluster[cluster] = top_genes
# convert dictionary to data frame
top_genes_summary = pd.concat(top_genes_per_cluster.values(), keys=top_genes_per_cluster.keys())
print(top_genes_summary)
```
```
         group     names     scores  logfoldchanges          pvals   pvals_adj  
0  0         0     FABP4  51.705452             NaN   0.000000e+00   0.000000e+00 
   1         0     FABP5  51.684387             NaN   0.000000e+00   0.000000e+00  
   2         0      RGS5  51.101852             NaN   0.000000e+00   0.000000e+00  
1  2000      1     IFI27  52.558998             NaN   0.000000e+00   0.000000e+00  
   2001      1     ISG15  49.790508             NaN   0.000000e+00   0.000000e+00  
   2002      1  MTRNR2L1  42.505508             NaN   0.000000e+00   0.000000e+00  
2  4000      2       DCN  65.856918             NaN   0.000000e+00   0.000000e+00  
   4001      2       GSN  64.210709             NaN   0.000000e+00   0.000000e+00  
   4002      2       CFD  60.884686             NaN   0.000000e+00   0.000000e+00  
3  6000      3      MYL2  58.421303             NaN   0.000000e+00   0.000000e+00  
   6001      3     TNNC1  58.342094             NaN   0.000000e+00   0.000000e+00  
   6002      3        MB  58.325264             NaN   0.000000e+00   0.000000e+00  
...etc.  
```

This analysis identifies genes that are significantly overexpressed in each cluster compared to all other clusters, using the Wilcoxon rank-sum test with Bonferroni correction for multiple testing. These marker genes provide our first insight into the biological identity of each cluster.

To systematically annotate clusters, we can use a custom database of known cell type markers:

```python
def annotate_with_custom_db(adata, top_n=50):
    """
    Annotate clusters using a custom database of cell type markers
    """
    print("Starting custom DB annotation...")
    # Define cell marker database structured for efficient lookup -- This is based on major cell type markers from multiple sources
    cell_markers = {
        # Immune cells
        'T cells': ['CD3D', 'CD3E', 'CD3G', 'CD2', 'CD7', 'IL7R', 'LCK', 'CD28'],
        'CD4+ T cells': ['CD4', 'IL7R', 'CCR7', 'LEF1', 'TCF7', 'MAL'],
        'CD8+ T cells': ['CD8A', 'CD8B', 'GZMK', 'GZMA', 'CCL5', 'GNLY'],
        'Regulatory T cells': ['FOXP3', 'IL2RA', 'CTLA4', 'TIGIT', 'IKZF2'],
        'B cells': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'HLA-DRA', 'CD74'],
        'Plasma cells': ['JCHAIN', 'MZB1', 'SSR4', 'XBP1', 'IGHA1', 'IGHG1'],
        'NK cells': ['NCAM1', 'NKG7', 'GNLY', 'KLRD1', 'KLRF1', 'FCGR3A'],
        'Monocytes': ['CD14', 'LYZ', 'VCAN', 'S100A9', 'S100A8', 'FCN1'],
        'Macrophages': ['CD68', 'MSR1', 'MARCO', 'VSIG4', 'C1QA', 'C1QB', 'APOE'],
        'Dendritic cells': ['CLEC9A', 'CLEC10A', 'CD1C', 'FCER1A', 'ITGAX', 'IRF8'],
        'Neutrophils': ['ELANE', 'MPO', 'S100A8', 'S100A9', 'CEACAM8', 'FCGR3B'],
        'Mast cells': ['CPA3', 'TPSAB1', 'TPSB2', 'MS4A2', 'HDC', 'KIT'],
        
        # Endothelial/Vascular
        'Endothelial cells': ['PECAM1', 'VWF', 'CDH5', 'CLDN5', 'SELE', 'KDR', 'TEK'],
        'Lymphatic endothelial': ['PROX1', 'PDPN', 'FLT4', 'CCL21', 'LYVE1'],
        'Pericytes': ['RGS5', 'PDGFRB', 'DES', 'ACTA2', 'MYH11', 'MCAM', 'CSPG4'],
        
        # Epithelial
        'Epithelial cells': ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'CDH1', 'CLDN4', 'CLDN7'],
        
        # Stromal/Mesenchymal
        'Fibroblasts': ['DCN', 'LUM', 'COL1A1', 'COL1A2', 'COL3A1', 'COL6A1', 'PDGFRA', 'FAP'],
        'Smooth muscle': ['ACTA2', 'TAGLN', 'MYH11', 'CNN1', 'DES', 'TPM2', 'MYL9'],
        'Skeletal muscle': ['MYH1', 'MYH2', 'ACTA1', 'TTN', 'MYBPC1', 'CKM', 'MB'],
        'Adipocytes': ['ADIPOQ', 'LEP', 'FABP4', 'PLIN1', 'CFD', 'PPARG'],
        
        # Other
        'Neurons': ['MAP2', 'RBFOX3', 'TUBB3', 'SYP', 'SNAP25', 'NEFL', 'NEFM'],
        'Oligodendrocytes': ['MBP', 'MOG', 'MAG', 'PLP1', 'OLIG1', 'OLIG2'],
        'Astrocytes': ['GFAP', 'AQP4', 'SLC1A3', 'SLC1A2', 'ALDH1L1'],
        'Microglia': ['CX3CR1', 'P2RY12', 'ITGAM', 'TMEM119', 'TREM2', 'APOE'],
        'Hepatocytes': ['ALB', 'APOB', 'HP', 'FGA', 'FGB', 'APOA1', 'TTR'],
        'Erythrocytes': ['HBA1', 'HBA2', 'HBB', 'ALAS2', 'GYPA', 'SLC4A1'],
        'Interferon-responsive': ['ISG15', 'IFI6', 'IFI27', 'IFIT1', 'IFIT3', 'MX1', 'OAS1'],}
    
    # Flatten marker list for easier lookup
    marker_to_celltype = {}
    for cell_type, markers in cell_markers.items():
        for marker in markers:
            if marker in marker_to_celltype:
                marker_to_celltype[marker].append(cell_type)
            else:
                marker_to_celltype[marker] = [cell_type]
    print(f"Marker database contains {len(marker_to_celltype)} unique genes across {len(cell_markers)} cell types")
    
    # Get clusters
    clusters = adata.uns['rank_genes_groups']['names'].dtype.names
    cluster_annotations = {}
    print(f"Annotating {len(clusters)} clusters...")
    
    # Process each cluster
    for cluster in clusters:
        print(f"Processing cluster {cluster}")
        # Get top markers for this cluster
        top_markers = []
        for i in range(min(top_n, len(adata.uns['rank_genes_groups']['names'][cluster]))):
            marker = adata.uns['rank_genes_groups']['names'][cluster][i]
            score = adata.uns['rank_genes_groups']['scores'][cluster][i]
            pval = adata.uns['rank_genes_groups']['pvals'][cluster][i]
            if pval < 0.05:  # Only consider statistically significant markers
                top_markers.append((marker, score, i))
        
        # Match markers to cell types
        cell_type_matches = {}
        for marker, score, rank in top_markers:
            if marker in marker_to_celltype:
                for cell_type in marker_to_celltype[marker]:
                    if cell_type not in cell_type_matches:
                        cell_type_matches[cell_type] = []
                    # Store marker, score, and rank
                    cell_type_matches[cell_type].append((marker, score, rank))
        
        # Score cell types (weighted by marker rank and score)
        cell_type_scores = {}
        for cell_type, matches in cell_type_matches.items():
            # Calculate a combined score based on:
            # 1. Number of markers
            # 2. Rank of markers (earlier = better)
            # 3. Score of markers (higher = better)
            score = sum([m[1] * (1 - m[2]/top_n) for m in matches])
            # Also consider the proportion of cell type markers found
            proportion = len(matches) / len(cell_markers[cell_type])
            final_score = score * (1 + proportion)
            cell_type_scores[cell_type] = (final_score, [m[0] for m in matches])
        
        # Get top cell types
        sorted_cell_types = sorted(cell_type_scores.items(), key=lambda x: x[1][0], reverse=True)
        
        if sorted_cell_types:
            # Take top 2 matches
            top_matches = sorted_cell_types[:2]
            annotation = " / ".join([f"{ct} ({', '.join(genes[:3])})" 
                                    for ct, (_, genes) in top_matches])
            cluster_annotations[cluster] = annotation
        else:
            # If no matches, use top marker genes
            markers = [m[0] for m in top_markers[:3]] if top_markers else ["No significant markers"]
            cluster_annotations[cluster] = f"Unknown (Top genes: {', '.join(markers)})"
    
    # Add annotations to adata
    adata.obs['custom_cell_type'] = adata.obs['clusters'].map(cluster_annotations)
    print("Cell type annotation complete.")  
    return cluster_annotations

# Run the annotation function
cluster_annotations = annotate_with_custom_db(adata_filtered)

# Print the cluster annotations
print("\nCluster Annotations:")
for cluster, annotation in cluster_annotations.items():
    print(f"Cluster {cluster}: {annotation}")

# Visualize the annotated clusters on UMAP
sc.pl.umap(adata_filtered, color='custom_cell_type', legend_loc='on data', 
           legend_fontsize=8, title='Cell Types (Custom DB)')

# Also visualize the clusters with their numeric IDs for reference
sc.pl.umap(adata_filtered, color='clusters', legend_loc='on data', 
           legend_fontsize=10, title='Cluster IDs')
```
```
Starting custom DB annotation...
Marker database contains 164 unique genes across 27 cell types
Annotating 14 clusters...
Processing cluster 0
Processing cluster 1
Processing cluster 2
Processing cluster 3
Processing cluster 4
Processing cluster 5
Processing cluster 6
Processing cluster 7
Processing cluster 8
Processing cluster 9
Processing cluster 10
Processing cluster 11
Processing cluster 12
Processing cluster 13
Cell type annotation complete.

Cluster Annotations:
Cluster 0: Pericytes (RGS5, ACTA2) / Smooth muscle (ACTA2, TAGLN)
Cluster 1: Interferon-responsive (IFI27, ISG15, IFIT3) / B cells (CD74, HLA-DRA)
Cluster 2: Fibroblasts (DCN, COL1A2, LUM) / Adipocytes (CFD)
Cluster 3: Skeletal muscle (MB, ACTA1, CKM) / Smooth muscle (DES, TPM2)
Cluster 4: B cells (HLA-DRA, CD74) / Macrophages (C1QA, C1QB, VSIG4)
Cluster 5: Monocytes (LYZ, S100A9, FCN1) / Neutrophils (S100A9, S100A8)
Cluster 6: T cells (CD3D, IL7R, CD7) / CD8+ T cells (CCL5, GZMA, GNLY)
Cluster 7: Microglia (APOE) / Macrophages (APOE)
Cluster 8: B cells (CD74, HLA-DRA) / Interferon-responsive (IFI27, IFIT3)
Cluster 9: Smooth muscle (TAGLN, ACTA2, MYH11) / Pericytes (ACTA2, MYH11)
Cluster 10: B cells (HLA-DRA, CD74, CD79A) / Erythrocytes (HBB, HBA2, HBA1)
Cluster 11: Fibroblasts (DCN, COL1A2, COL3A1) / Adipocytes (CFD)
Cluster 12: Skeletal muscle (CKM, ACTA1, MYH2) / Smooth muscle (TPM2, DES)
Cluster 13: Smooth muscle (ACTA2, TAGLN, TPM2) / Pericytes (ACTA2, RGS5, MYH11)
```

<img width="882" alt="Screenshot 2025-04-08 at 8 30 25 AM" src="https://github.com/user-attachments/assets/6e3e4111-2b1e-4e91-a8ea-a61fd973b4d2" />

This function compares the marker genes from each cluster with a database of known cell type markers, scoring cell types based on the number and significance of matching markers. The result is a biologically meaningful annotation for each cluster, transforming abstract numerical clusters into interpretable cell types.

Additionally, we can look at the cellular composition of our skeletal muscle sample and visualize genes that are differentially expressed between clusters using the code below:

```python
# get the counts of cells in each cluster
cluster_counts = adata_filtered.obs['clusters'].value_counts()

# calculate the total number of cells
total_cells = len(adata_filtered.obs)

# calculate the percentage of cells in each cluster
cluster_percentages = (cluster_counts / total_cells) * 100

# display the results
print("\nPercentage of cells in each cluster:")
print(cluster_percentages)
```
```
Percentage of cells in each cluster:
clusters
0     23.202886
1     16.930336
5     15.255805
2     13.572023
4      8.058100
11     7.031178
6      5.402905
7      5.227126
3      1.942825
8      1.359978
10     1.100934
9      0.730872
```

This concordance between our computational clustering and the known cellular architecture of skeletal muscle not only validates our analytical pipeline but also highlights the power of single-cell approaches to accurately capture tissue heterogeneity at unprecedented resolution.

```python
# Create a function to visualize top marker genes for all clusters
def visualize_top_markers(adata, n_genes=10):
    # Get results for each cluster
    cluster_markers = {}
    
    for cluster in adata.obs['clusters'].unique():
        # Get top markers for this cluster
        markers = pd.DataFrame(
            {
                'names': adata.uns['rank_genes_groups']['names'][cluster][:n_genes],
                'scores': adata.uns['rank_genes_groups']['scores'][cluster][:n_genes],
                'pvals': adata.uns['rank_genes_groups']['pvals'][cluster][:n_genes],
                'pvals_adj': adata.uns['rank_genes_groups']['pvals_adj'][cluster][:n_genes]})
        cluster_markers[cluster] = markers
    
    # Get unique top markers across all clusters
    all_markers = []
    for cluster, markers in cluster_markers.items():
        all_markers.extend(markers['names'].tolist())
    
    # Remove duplicates while preserving order
    unique_markers = []
    seen = set()
    for marker in all_markers:
        if marker not in seen:
            unique_markers.append(marker)
            seen.add(marker)
    
    # Limit to a reasonable number for visualization
    if len(unique_markers) > 50:
        unique_markers = unique_markers[:50]
        
    # Create dotplot of marker genes
    sc.pl.dotplot(adata, unique_markers, groupby='clusters', dendrogram=True)
    
    # Create heatmap
    sc.pl.heatmap(adata, unique_markers, groupby='clusters', 
                 swap_axes=True, show_gene_labels=True, 
                 vmin=-2, vmax=2, cmap='viridis')
    return cluster_markers

cluster_markers = visualize_top_markers(adata_filtered, n_genes=10)
```

![Unknown-8](https://github.com/user-attachments/assets/f5c52c09-11d7-4f2e-844d-4d1519955981)

![Unknown-9](https://github.com/user-attachments/assets/8e6cf742-b95a-4d8c-8bc4-082144eec0ec)

# 🧬 Differential Expression Analysis -  Comparing Cell Types Across Conditions
Once we've identified cell types, we unlock one of the most powerful analytical capabilities unique to single-cell RNA-sequencing: the ability to examine gene expression changes in two distinct dimensions. This represents a fundamental advance over bulk RNA-seq, where cellular heterogeneity is compressed into a single averaged signal.

The first dimension involves finding marker genes between clusters, which is what we've already accomplished through our clustering process. When we executed ```sc.tl.rank_genes_groups()``` with ```groupby='clusters'```, we identified genes that distinguish each cluster from all others using the Wilcoxon rank-sum test. These differentially expressed genes are what define the transcriptional identity of each cell type. For instance, muscle stem cells might be characterized by PAX7 and MYF5 expression, while mature myofibers show high levels of myosin and tropomyosin genes. This between-cluster differential expression forms the foundation of our cell type annotations.

However, the second dimension—comparing the same cell type across experimental conditions—reveals the targeted biological responses of specific cell populations to interventions. This within-cluster, between-condition analysis requires a fundamentally different approach:

```python
def intra_cluster_de_analysis(adata, cluster_key='clusters', condition_key='condition', pval_cutoff=0.05, min_cells=3):
    print(f"Performing differential expression analysis between conditions for each cluster")
    
    # Check that we have exactly 2 conditions
    conditions = adata.obs[condition_key].unique()
    if len(conditions) != 2:
        raise ValueError(f"Expected exactly 2 conditions, but found {len(conditions)}")
        
    condition_a, condition_b = conditions
    print(f"Comparing '{condition_b}' vs '{condition_a}' (reference) for each cluster")
    
    # Dictionary to store results
    results_dict = {}
    
    # For each cluster, perform DE analysis between conditions
    for cluster in adata.obs[cluster_key].unique():
        print(f"\nAnalyzing cluster {cluster}...")
        
        # Subset data to cells from this cluster
        cluster_adata = adata[adata.obs[cluster_key] == cluster].copy()
        
        # Check if we have cells from both conditions in this cluster
        condition_counts = cluster_adata.obs[condition_key].value_counts()
        
        # Get counts safely
        count_a = condition_counts.get(condition_a, 0)
        count_b = condition_counts.get(condition_b, 0)
        
        print(f"  Cells: {count_a} {condition_a}, {count_b} {condition_b}")
        
        # Skip if any condition has fewer than min_cells
        if count_a < min_cells or count_b < min_cells:
            print(f"  Skipping - insufficient cells in at least one condition (minimum {min_cells} required)")
            continue
            
        # Perform DE analysis
        try:
            sc.tl.rank_genes_groups(cluster_adata, groupby=condition_key, groups=[condition_b], reference=condition_a, method='wilcoxon', corr_method='bonferroni')
            
            # Extract results to dataframe
            de_results = sc.get.rank_genes_groups_df(cluster_adata, group=condition_b)
            
            # Filter for significant genes
            sig_results = de_results[de_results['pvals_adj'] < pval_cutoff]
            
            # Handle results
            if not sig_results.empty:
                # Handle possible NaN values in logfoldchanges
                sig_results = sig_results.dropna(subset=['logfoldchanges'])
                
                if not sig_results.empty:
                    sig_results['direction'] = sig_results['logfoldchanges'].apply(
                        lambda x: 'up' if x > 0 else 'down')
                    sig_results['abs_logfc'] = abs(sig_results['logfoldchanges'])
                
                    # Count up/down regulated genes
                    n_up = sum(sig_results['direction'] == 'up')
                    n_down = sum(sig_results['direction'] == 'down')
                    
                    print(f"  Found {len(sig_results)} significant DE genes (adj p < {pval_cutoff})")
                    print(f"  {n_up} upregulated in {condition_b}, {n_down} downregulated in {condition_b}")
                    
                    # Store results
                    results_dict[cluster] = sig_results.sort_values('abs_logfc', ascending=False)
                else:
                    print(f"  No genes with valid fold changes in cluster {cluster}")
                    results_dict[cluster] = None
            else:
                print(f"  No significantly differentially expressed genes found in cluster {cluster}")
                results_dict[cluster] = None
                
        except ValueError as e:
            print(f"  Error: {e}")
            results_dict[cluster] = None
    return results_dict

# Execute the analysis with minimum cell count requirement
de_results = intra_cluster_de_analysis(adata_filtered, min_cells=5)
for cluster, results in de_results.items():
    if results is not None and not results.empty:
        results.to_csv(f'de_results/cluster_{cluster}_de_genes.csv')
        
        # Print top DE genes for each cluster
        print(f"\nTop 5 DE genes for cluster {cluster}:")
        print(results[['names', 'logfoldchanges', 'pvals_adj', 'direction']].head(5))
```
```
Top 5 DE genes for cluster 0:
        names  logfoldchanges     pvals_adj direction
1987      LXN       -4.429174  4.952524e-02      down
1999  MT-ATP6       -2.603790  1.948892e-85      down
5       HYAL2        2.508442  1.579726e-13        up
1991    REEP2       -2.484809  2.276777e-04      down
7       RAMP3        2.174930  2.324267e-08        up
...etc
```

The function above systematically isolates each cell type, ensuring adequate representation in both experimental conditions, and then performs targeted differential expression analysis between conditions within that specific cell population. The approach reveals cell type-specific responses that would be diluted or entirely masked in bulk analysis.

For example, in a skeletal muscle injury model, we might find that satellite cells show upregulation of proliferation and cell cycle genes, while mature myofibers activate stress response pathways, and resident macrophages transition from inflammatory to regenerative phenotypes. These distinct responses occurring simultaneously in different cell populations would appear as a confusing averaged signal in bulk RNA-seq, but emerge with clarity in single-cell analysis.

The ability to perform this two-dimensional differential expression analysis—between clusters to identify cell types, and within clusters across conditions to detect cell type-specific responses—represents one of the most transformative advances of single-cell technology, enabling unprecedented resolution of complex biological processes.

## 🧬 Conclusion - The Future of Single-Cell Genomics
Single-cell RNA sequencing has transformed our understanding of cellular biology, revealing heterogeneity and dynamics that were previously invisible. The analytical approaches outlined here provide a foundation for exploring this rich data, but the field continues to evolve rapidly.

As we look to the future, the integration of multi-omic measurements – simultaneously profiling gene expression, chromatin accessibility, protein levels, and spatial location within a tissue – promises to further enhance our understanding of cellular identity and function. The computational challenges of these integrated approaches are substantial, but they offer the potential for a truly comprehensive view of cellular biology.

In the meantime, the analytical pipeline described here provides a robust framework for extracting biological insights from single-cell RNA sequencing data, enabling researchers to dissect complex tissues with unprecedented resolution and discover new cellular states and dynamics that were previously hidden from view.
