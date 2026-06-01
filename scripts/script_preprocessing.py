"""
01_preprocessing.py
QC, normalization, and dimensionality reduction for Visium spatial transcriptomics datasets.
Handles both standard Visium (SpaceRanger output) and pre-processed count matrices.

Dependencies: scanpy, squidpy, anndata, pandas, numpy, matplotlib
"""

import scanpy as sc
import squidpy as sq
import anndata as ad
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


# ── Config ───────────────────────────────────────────────────────────────────
SAMPLE_MANIFEST = {
    # accession : (path, disease_stage)
    "GSM7421790"  : ("data/GSM7421790",  "IPMN"),
    "GSM8443449"  : ("data/GSE274103",  "Primary_PDAC"),
    "GSM8443450"  : ("data/GSE274103",  "Primary_PDAC"),
    "GSM8443451"  : ("data/GSE274103",  "Primary_PDAC"),
    "GSM8443452"  : ("data/GSE274103",  "Primary_PDAC"),
    "GSM8443453"  : ("data/GSE274103",  "Primary_PDAC"),
    "GSM8452857"  : ("data/GSM8452857",  "Metastasis"),
}
OUTPUT_DIR   = "data/processed"
FIGURE_DIR   = "G:/My Drive/ongkeko/CRC project"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# QC thresholds
MIN_COUNTS   = 500
MIN_GENES    = 250
MAX_PCT_MITO = 25
MIN_CELLS    = 10     # min spots a gene must appear in


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_sample(sample_id: str, path: str, stage: str) -> ad.AnnData:
    """Load a Visium sample from SpaceRanger output dir or .h5ad file."""
    if path.endswith(".h5ad"):
        adata = sc.read_h5ad(path)
    else:
        adata = sq.read.visium(path)
    adata.var_names_make_unique()
    adata.obs["sample_id"]     = sample_id
    adata.obs["disease_stage"] = stage
    print(f"  Loaded {sample_id}: {adata.n_obs} spots, {adata.n_vars} genes")
    return adata


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
    sc.tl.pca(adata, n_comps=50, mask_var="highly_variable")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5, flavor='igraph', n_iterations=2, directed = False)

    # Spatial neighborhood graph (for downstream squidpy analyses)
    return adata


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    adatas = []

    expanded_manifest = {}

    for gse_id, (gse_path, stage) in SAMPLE_MANIFEST.items():

        for gsm_dir in Path(gse_path).iterdir():
            if (not gsm_dir.is_dir()) or (gsm_dir.name == 'spatial'):
                continue
            gsm_id = gsm_dir.name

            expanded_manifest[gsm_id] = (
                str(gsm_dir),
                stage
            )
    print(expanded_manifest)

    for sid, (path, stage) in expanded_manifest.items():
        print(f"\nProcessing {sid} [{stage}]...")
        
        print(path)

        adata = load_sample(sid, path, stage)
        adata = run_qc(adata)
        adata = normalize_and_embed(adata)
        out_path = f"{OUTPUT_DIR}/{sid}_processed.h5ad"
        adata.write_h5ad(out_path)
        print(f"  Saved → {out_path}")
        adatas.append(adata)

    print(f"\nDone. {len(adatas)} samples processed.")

    # grid: For data where spots are in a regular, repeating pattern (like Visium's hexagonal grid).
    # generic: For data with irregular coordinates (like single-cell spatial data where cells are scattered randomly).
    sq.gr.spatial_neighbors(adata, coord_type="grid")
