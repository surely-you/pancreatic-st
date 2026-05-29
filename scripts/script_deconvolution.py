"""
02_deconvolution.py
Cell-type deconvolution of Visium spots using cell2location.
Requires a single-cell reference (e.g. from a PDAC scRNA-seq atlas).

Dependencies: cell2location, scanpy, anndata, torch, numpy, matplotlib
"""

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cell2location
from cell2location.utils.filtering import filter_genes
import os
import scipy.sparse as sp


# ── Config ───────────────────────────────────────────────────────────────────
SC_REF_PATH  = "data/reference/pdac_scrna_reference.h5ad"  # scRNA-seq reference
SPATIAL_DIR  = "data/processed"
OUTPUT_DIR   = "data/deconvolved"
FIGURE_DIR   = "G:/My Drive/ongkeko/CRC project"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

CELL_TYPE_COL   = "Cell_type"    # column in sc_ref.obs with cell type labels
N_CELLS_PER_LOC = 8              # expected avg cells per Visium spot
DETECTION_ALPHA = 200            # cell2location hyperparameter
EPOCHS_REF      = 250
EPOCHS_SPATIAL  = 30000

SPATIAL_SAMPLES = [
    "GSM7421790", "GSM8443449", "GSM8452857"
]


# ── Step 1: Train NB regression model on scRNA-seq reference ─────────────────
def train_reference_model(sc_ref: ad.AnnData):
    """Estimate per-cell-type gene expression signatures."""

    # data wrangling
    print(sc_ref.shape) 
    sc.pp.subsample(sc_ref, fraction=0.025) #CHANGE THISSSSS THIS IS WAY TOO LOW TEST ONLY DONT FORGET ABOUT THIS
    print(sc_ref.shape) 
    gene_sums = np.array(sc_ref.X.sum(axis=0)).flatten()
    sc_ref = sc_ref[:, gene_sums > 0].copy()
    
    idx = (sc_ref.obs.groupby(CELL_TYPE_COL)
              .apply(lambda x: x.sample(min(len(x), 500)))
              .index.get_level_values(1))
    sc_ref_sub = sc_ref[idx].copy()
    print(sc_ref.shape) 


    # Filter to informative genes
    selected = filter_genes(sc_ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03,
                            nonz_mean_cutoff=1.12)
    print(1.5)
    sc_ref = sc_ref[:, selected].copy()
    print(2)
    cell2location.models.RegressionModel.setup_anndata(
        sc_ref,
        batch_key="Patient",
        labels_key=CELL_TYPE_COL,
    )
    print(3)
    ref_model = cell2location.models.RegressionModel(sc_ref)
    ref_model.train(max_epochs=EPOCHS_REF)
    print(4)
    # Export posterior — inf_aver is the per-cell-type expression signature
    sc_ref = ref_model.export_posterior(sc_ref, sample_kwargs={"num_samples": 1000})
    inf_aver = sc_ref.varm["means_per_cluster_mu_fg"][
        [f"means_per_cluster_mu_fg_{ct}" for ct in sc_ref.uns["mod"]["factor_names"]]
    ].copy()
    inf_aver.columns = sc_ref.uns["mod"]["factor_names"]
    print(5)
    ref_model.save("models/reference_model", overwrite=True)
    return inf_aver


# ── Step 2: Deconvolve each spatial sample ───────────────────────────────────
def deconvolve_sample(adata: ad.AnnData, inf_aver: pd.DataFrame, sid: str) -> ad.AnnData:
    """Run cell2location spatial mapping for one sample."""
    # Keep only genes shared between reference and spatial data
    shared = inf_aver.index.intersection(adata.var_names)
    adata  = adata[:, shared].copy()
    inf_av = inf_aver.loc[shared]

    cell2location.models.Cell2location.setup_anndata(adata, batch_key=None)
    model = cell2location.models.Cell2location(
        adata,
        cell_state_df=inf_av,
        N_cells_per_location=N_CELLS_PER_LOC,
        detection_alpha=DETECTION_ALPHA,
    )
    model.train(
        max_epochs=EPOCHS_SPATIAL,
        batch_size=None,
        train_size=1
    )

    adata = model.export_posterior(
        adata,
        sample_kwargs={"num_samples": 1000, "batch_size": model.adata.n_obs},
    )

    # Summarise: mean cell abundance per spot
    adata.obs[adata.uns["mod"]["factor_names"]] = \
        adata.obsm["means_cell_abundance_w_sf"]

    # Plot spatial cell-type abundance maps
    ct_cols = adata.uns["mod"]["factor_names"]
    sc.pl.spatial(
        adata, color=ct_cols[:min(8, len(ct_cols))],
        ncols=4, size=1.3, img_key="hires",
        save=f"_{sid}_celltype_abundance.png",
    )

    model.save(f"models/{sid}_spatial_model", overwrite=True)
    return adata


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading scRNA-seq reference...")
    sc_ref = sc.read_h5ad(SC_REF_PATH)

    print("Training reference model...")
    print(sc_ref is None)
    inf_aver = train_reference_model(sc_ref)
    inf_aver.to_csv(f"{OUTPUT_DIR}/inf_aver_signatures.csv")

    for sid in SPATIAL_SAMPLES:
        path = f"data/processed/{sid}_processed.h5ad"
        if not os.path.exists(path):
            print(f"  Skipping {sid} — processed file not found")
            continue
        print(f"\nDeconvolving {sid}...")
        adata = sc.read_h5ad(path)
        adata = deconvolve_sample(adata, inf_aver, sid)
        adata.write_h5ad(f"{OUTPUT_DIR}/{sid}_deconvolved.h5ad")
        print(f"  Saved → {OUTPUT_DIR}/{sid}_deconvolved.h5ad")
