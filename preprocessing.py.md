# Annotations on script_preprocessing.py

Summary: 
1. **Environment Setup & Data Entry**: Loading dependencies and mapping sample manifests.
2. **Quality Control (QC)**: Filtering low-quality spots and mitochondrial contamination.
3. **Normalization & Feature Selection**: Scaling data and identifying highly variable genes.
4. **Embedding & Clustering**: Dimensionality reduction (PCA, UMAP) and neighborhood graphs.

## Setting up the environment & importing data
Bioinformatics packages
* **scanpy**: Comprehensive toolkit for single-cell RNA-seq analysis. Handles preprocessing, clustering, trajectory inference, and differential expression
* **squidpy**: Extension for spatial omics. Integrates spatial coordinates with omics data; used for neighborhood analysis and image-based feature extraction
* **anndata**: Specialized data structure (Annotated Data) designed for high-dimensional observations. Stores the primary expression matrix ($X$) alongside observation metadata ($obs$) and variable metadata ($var$)

Mathy math 
* **pandas**: High-performance library for tabular data manipulation (DataFrames). Primary tool for managing cell/gene metadata and CSV/TSV exports.
* **numpy**: Fundamental package for scientific computing. Provides N-dimensional array objects and the mathematical functions required for large-scale matrix operations.

misc/others
* **os**: Interface for operating system interaction. Facilitates file path management, directory creation, and environment variable handling.
* **matplotlib.pyplot**: make fancy fancy graphs. Provides the backend for Scanpy/Squidpy visualizations; used for fine-tuning figure aesthetics and multi-panel layouts.

COGS 108: How to panda 🐼 (https://github.com/COGS108/Lectures-Sp26/blob/main/05-Pandas.ipynb)
```
import scanpy as sc
import squidpy as sq
import anndata as ad
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
SAMPLE_MANIFEST = {
    # accession : (path, disease_stage)
    "GSE254829"  : ("data/raw/GSE254829",  "PanIN"),
    "GSE233293"  : ("data/raw/GSE233293",  "IPMN"),
    "GSE327056"  : ("data/raw/GSE327056",  "Normal_PDAC"),
    "GSE274103"  : ("data/raw/GSE274103",  "Primary_PDAC"),
    "GSE310353"  : ("data/raw/GSE310353",  "Primary_PDAC"),
    "Syn61831984": ("data/raw/Syn61831984","Primary_PDAC"),
    "GSE278694"  : ("data/raw/GSE278694",  "Primary_PDAC"),
    "GSE272362"  : ("data/raw/GSE272362",  "Metastasis"),
    "GSE274557"  : ("data/raw/GSE274557",  "Metastasis"),
    "10x_VisiumHD": ("data/raw/10x_VisiumHD", "Primary_PDAC"),
}
OUTPUT_DIR   = "data/processed"
FIGURE_DIR   = "figures/qc"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Loaders ──────────────────────────────────────────────────────────────────
def load_sample(sample_id: str, path: str, stage: str) -> ad.AnnData:
    """Load a Visium sample from SpaceRanger output dir or .h5ad file."""
    if path.endswith(".h5ad"):
        adata = sc.read_h5ad(path)
    else:
        adata = sc.read_visium(path)
    adata.var_names_make_unique()
    adata.obs["sample_id"]     = sample_id
    adata.obs["disease_stage"] = stage
    print(f"  Loaded {sample_id}: {adata.n_obs} spots, {adata.n_vars} genes")
    return adata

```

## QC thresholds
ensures that we only analyze spots containing high-quality biological information. We look for three main red flags:

* **Low Count Spots:** Likely empty spots or technical failures.
* **Low Gene Diversity**: Spots with very few unique genes detected.
* **Mitochondrial Overexpression:** High % of MT-DNA often indicates cell stress or lysis, where the cytoplasmic mRNA has leaked out, leaving only mitochondrial transcripts.

"Cells with a very low number of genes (<500) are considered of low quality and hence are removed from the analysis. Cells with high mitochondria read percentage (>10%) are also removed as high expression level of mitochondrial genes indicate damaged or dead cells."
source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10663991/
```
MIN_COUNTS   = 500
MIN_GENES    = 250
MAX_PCT_MITO = 25
MIN_CELLS    = 10     # min spots a gene must appear in

# ── QC ───────────────────────────────────────────────────────────────────────
def run_qc(adata: ad.AnnData) -> ad.AnnData:
    sid = adata.obs["sample_id"].iloc[0]
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # QC violin plots
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, m in zip(axes, ["total_counts", "n_genes_by_counts", "pct_counts_mt"]):
        adata.obs[m].hist(bins=50, ax=ax)
        ax.set_title(f"{sid} — {m}")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/{sid}_qc.png", dpi=120)
    plt.close()

    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=MIN_COUNTS)
    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    adata = adata[adata.obs["pct_counts_mt"] < MAX_PCT_MITO].copy()
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    print(f"  {sid}: {n_before} → {adata.n_obs} spots after QC")
    return adata

```
## Normalizating & Clustering

### normalizing
To compare gene expression across different spots, we must normalize the data to account for differences in capture efficiency.
**normalize_total**: scale each spot to a fixed target sum ($10,000$ counts) and then apply a $log(1+x)$ transformation (cues BILD 5 week 5 EC ahem ahem) to compress the dynamic range of highly expressed genes
**Highly Variable Genes (HVGs)**: genes that vary significantly across the tissue and are most likely to define different cell types or tissue zones.

### clustering
This final step projects the high-dimensional gene data into a lower-dimensional space to identify "clusters" (communities of spots with similar expression).
* **PCA**: Reduces the 3,000 HVGs into 50 Principal Components.
  * COGS 108 slide bout PCA is linked under integration
* **Neighborhood Graph**: A mathematical representation of cell-to-cell similarity based on gene expression.
  * identifies the nearest neighbors for each cell in the latent space
  * foundation for clustering (e.g., Leiden/Louvain) and UMAP generation.
* **UMAP**(Uniform Manifold Approximation and Projection): A 2D visualization of the high-dimensional relationships.
* **Spatial Neighbors**: use squidpy to build a graph based on the physical $(x, y)$ coordinates of the spots on the slide.
  * Essential for identifying spatial patterns, such as cellular niches, tissue boundaries, or ligand-receptor interactions between neighboring cells.
 
**note (error)**: _sq.gr.spatial_neighbors(adata, coord_type="visium")_: visium isnt a valid parameter for coord type (outdated?) use one of the following (likely grid with visium)
* grid: For data where spots are in a regular, repeating pattern (like Visium's hexagonal grid).
* generic: For data with irregular coordinates (like single-cell spatial data where cells are scattered randomly).
    sq.gr.spatial_neighbors(adata, coord_type="grid")
```
# ── Normalization & Embedding ─────────────────────────────────────────────────
def normalize_and_embed(adata: ad.AnnData) -> ad.AnnData:
    adata.layers["counts"] = adata.X.copy()        # preserve raw counts for cell2location
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata                               # store log-norm for DE

    sc.pp.highly_variable_genes(
        adata, n_top_genes=3000, flavor="seurat_v3", layer="counts"
    )
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)

    # Spatial neighborhood graph (for downstream squidpy analyses)
    sq.gr.spatial_neighbors(adata, coord_type="visium")
    return adata
```
## Main
calls all aforementioned functions and ties everything together
```
# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    adatas = []
    for sid, (path, stage) in SAMPLE_MANIFEST.items():
        print(f"\nProcessing {sid} [{stage}]...")
        adata = load_sample(sid, path, stage)
        adata = run_qc(adata)
        adata = normalize_and_embed(adata)
        out_path = f"{OUTPUT_DIR}/{sid}_processed.h5ad"
        adata.write_h5ad(out_path)
        print(f"  Saved → {out_path}")
        adatas.append(adata)

    print(f"\nDone. {len(adatas)} samples processed.")
```
