!pip install scanpy
!pip install leidenalg

import pandas as pd
import scanpy as sc
import numpy as np
import anndata as an
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import os
import gzip
from urllib.request import urlretrieve
from pathlib import Path
from typing import List, Dict
import scipy.sparse as sp

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

# Example usage:
if __name__ == "__main__":
    samples = [
        {'accession': 'GSM6611295', 'file_prefix': 'GSM6611295_P15306_5001', 'condition': 'pre', 'subject': '1', 'label': 'subject1_pre'},
        {'accession': 'GSM6611296', 'file_prefix': 'GSM6611296_P15306_5002', 'condition': 'post', 'subject': '1', 'label': 'subject1_post'},
        {'accession': 'GSM6611297', 'file_prefix': 'GSM6611297_P14601_4004', 'condition': 'pre', 'subject': '2', 'label': 'subject2_pre'},
        {'accession': 'GSM6611298', 'file_prefix': 'GSM6611298_P14601_4005', 'condition': 'post', 'subject': '2', 'label': 'subject2_post'},
        {'accession': 'GSM6611299', 'file_prefix': 'GSM6611299_P15306_5003', 'condition': 'pre', 'subject': '3', 'label': 'subject3_pre'},
        {'accession': 'GSM6611300', 'file_prefix': 'GSM6611300_P15306_5004', 'condition': 'post', 'subject': '3','label': 'subject3_post'}]
    adata = download_and_process_scRNA_data(samples)

num_genes = adata.n_vars # variables = columns (genes)
print(f"Number of Genes: {num_genes}")

num_cells = adata.n_obs # observations = rows (cells IDs)
print(f"Number of Cells: {num_cells}")

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

# Visualize QC metrics with improved plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Distribution of genes per cell
sns.histplot(adata.obs['n_genes_by_counts'], bins=50, kde=True, ax=axes[0, 0])
axes[0, 0].set_xlabel('Number of Genes per Cell')
axes[0, 0].set_ylabel('Number of Cells')
axes[0, 0].set_title('Distribution of Number of Genes per Cell')
axes[0, 0].axvline(200, color='red', linestyle='--', label='Filter threshold (200)')
axes[0, 0].legend()

# Plot 2: Distribution of UMI counts per cell
sns.histplot(adata.obs['total_counts'], bins=50, kde=True, ax=axes[0, 1])
axes[0, 1].set_xlabel('Total UMI Counts per Cell')
axes[0, 1].set_ylabel('Number of Cells')
axes[0, 1].set_title('Distribution of UMI Counts per Cell')
axes[0, 1].set_xscale('log')

# Plot 3: Distribution of mitochondrial gene percentage
sns.histplot(adata.obs['pct_counts_mt'], bins=50, kde=True, ax=axes[1, 0])
axes[1, 0].set_xlabel('Percentage of Mitochondrial Genes')
axes[1, 0].set_ylabel('Number of Cells')
axes[1, 0].set_title('Distribution of Mitochondrial Gene % per Cell')
axes[1, 0].axvline(20, color='red', linestyle='--', label='Typical threshold (20%)')
axes[1, 0].legend()

# Plot 4: Scatter plot of UMI count vs genes per cell, colored by mito percent
scatter = axes[1, 1].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], c=adata.obs['pct_counts_mt'], cmap='viridis', s=10, alpha=0.7)
axes[1, 1].set_xlabel('Total UMI Counts per Cell')
axes[1, 1].set_ylabel('Number of Genes per Cell')
axes[1, 1].set_title('Genes vs UMIs, Colored by Mito%')
axes[1, 1].set_xscale('log')
cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_label('Mitochondrial %')
plt.tight_layout()
plt.show()

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

# Normalization and log transformation
print("Normalizing data...")
sc.pp.normalize_total(adata_filtered, target_sum=1e4)
sc.pp.log1p(adata_filtered)

# Highly variable gene selection with improved parameters
print("Identifying highly variable genes...")
sc.pp.highly_variable_genes(adata_filtered, n_top_genes=2000, subset=True)

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

# get the counts of cells in each cluster
cluster_counts = adata_filtered.obs['clusters'].value_counts()

# calculate the total number of cells
total_cells = len(adata_filtered.obs)

# calculate the percentage of cells in each cluster
cluster_percentages = (cluster_counts / total_cells) * 100

# display the results
print("\nPercentage of cells in each cluster:")
print(cluster_percentages)

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
