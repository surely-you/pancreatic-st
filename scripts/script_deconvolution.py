"""
02_deconvolution.py
Cell-type deconvolution of Visium spots using cell2location.
Requires a single-cell reference (e.g. from a PDAC scRNA-seq atlas).

Dependencies: cell2location, scanpy, anndata, torch, numpy, matplotlib
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import cell2location
from cell2location.utils.filtering import filter_genes
import scipy.sparse as sp
import torch
from datetime import datetime


# ── Timing helper ─────────────────────────────────────────────────────────────
def ts():
    """Return current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def elapsed(start: datetime) -> str:
    """Return human-readable elapsed time since start."""
    delta = datetime.now() - start
    h, rem = divmod(int(delta.total_seconds()), 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    elif m:
        return f"{m}m {s}s"
    else:
        return f"{s}s"

def log(msg: str, start: datetime = None):
    """Print timestamped message, optionally with elapsed time."""
    if start:
        print(f"[{ts()}]  {msg}  (elapsed: {elapsed(start)})")
    else:
        print(f"[{ts()}]  {msg}")


# ── Config ───────────────────────────────────────────────────────────────────
SC_REF_PATH  = "data/reference/pdac_scrna_reference.h5ad"
SPATIAL_DIR  = "data/processed"
OUTPUT_DIR   = "data/deconvolved"
FIGURE_DIR   = "G:/My Drive/ongkeko/CRC project"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

CELL_TYPE_COL   = "Cell_type"
N_CELLS_PER_LOC = 8
DETECTION_ALPHA = 200
EPOCHS_REF      = 250
EPOCHS_SPATIAL  = 30000

NEED2TRAIN = False
USE_GPU = True

SPATIAL_SAMPLES = [
     # accession : (path, disease_stage)
    "GSM8443452" ,
    "GSM8452857" 
]


# ── Step 1: Train NB regression model on scRNA-seq reference ─────────────────
def train_reference_model(sc_ref: ad.AnnData):
    """Estimate per-cell-type gene expression signatures."""
    step_start = datetime.now()
    log("Starting reference model training")

    log("Subsampling and filtering reference data...")
    print(sc_ref.shape)
#    sc.pp.subsample(sc_ref, fraction=0.01)
    print(sc_ref.shape)
    gene_sums = np.array(sc_ref.X.sum(axis=0)).flatten()
    sc_ref = sc_ref[:, gene_sums > 0].copy()

    #samples up to 500 cells per cell type
    idx = (sc_ref.obs.groupby(CELL_TYPE_COL)
              .apply(lambda x: x.sample(min(len(x), 500)))
              .index.get_level_values(1))
    sc_ref = sc_ref[idx].copy()
    print(sc_ref.shape)

    t = datetime.now()
    log("Filtering genes...")
    selected = filter_genes(sc_ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03,
                            nonz_mean_cutoff=1.12)
    log("Gene filtering done", t)

    sc_ref = sc_ref[:, selected].copy()

    log("Setting up AnnData for regression model...")
    cell2location.models.RegressionModel.setup_anndata(
        sc_ref,
        batch_key="Patient",
        labels_key=CELL_TYPE_COL,
    )

    t = datetime.now()
    log("Training regression model...")
    ref_model = cell2location.models.RegressionModel(sc_ref)

    if(USE_GPU):
        ref_model.train(max_epochs=EPOCHS_REF, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    else:
        ref_model.train(max_epochs=EPOCHS_REF, accelerator = 'cpu')
    log(f"Regression model training done ({EPOCHS_REF} epochs)", t)

    log("Exporting posterior...")
    t = datetime.now()
    sc_ref = ref_model.export_posterior(sc_ref, sample_kwargs={"num_samples": 1000})
    log("Posterior export done", t)

    inf_aver = sc_ref.varm["means_per_cluster_mu_fg"][
        [f"means_per_cluster_mu_fg_{ct}" for ct in sc_ref.uns["mod"]["factor_names"]]
    ].copy()
    inf_aver.columns = sc_ref.uns["mod"]["factor_names"]

    ref_model.save("models/reference_model", overwrite=True)

    log("Reference model training complete", step_start)
    return inf_aver


# ── Step 2: Deconvolve each spatial sample ───────────────────────────────────
def deconvolve_sample(adata: ad.AnnData, inf_aver: pd.DataFrame, sid: str) -> ad.AnnData:
    """Run cell2location spatial mapping for one sample."""
    step_start = datetime.now()
    log(f"[{sid}] Starting deconvolution")

    shared = inf_aver.index.intersection(adata.var_names)
    adata  = adata[:, shared].copy()
    inf_av = inf_aver.loc[shared]
    log(f"[{sid}] Shared genes: {len(shared)}")

    cell2location.models.Cell2location.setup_anndata(adata,  batch_key=None)
    model = cell2location.models.Cell2location(
        adata,
        cell_state_df=inf_av,
        N_cells_per_location=N_CELLS_PER_LOC,
        detection_alpha=DETECTION_ALPHA,
    )

    t = datetime.now()
    log(f"[{sid}] Training spatial model ({EPOCHS_SPATIAL} epochs)...")

    if (USE_GPU):
        model.train(
            max_epochs=EPOCHS_SPATIAL,
            batch_size=None,
            train_size=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu"
        )
    else:
        model.train(
            max_epochs=EPOCHS_SPATIAL,
            batch_size=None,
            train_size=1,
            accelerator="cpu"
        )
    log(f"[{sid}] Spatial model training done", t)

    t = datetime.now()
    log(f"[{sid}] Exporting posterior...")
    adata = model.export_posterior(
        adata,
        sample_kwargs={"num_samples": 1000, "batch_size": model.adata.n_obs},
    )
    log(f"[{sid}] Posterior export done", t)

    adata.obs[adata.uns["mod"]["factor_names"]] = \
        adata.obsm["means_cell_abundance_w_sf"]

    ct_cols = adata.uns["mod"]["factor_names"]
    sc.pl.spatial(
        adata, color=ct_cols[:min(8, len(ct_cols))],
        ncols=4, size=1.3, img_key="hires",
        save=f"_{sid}_celltype_abundance.png",
    )

    model.save(f"models/{sid}_spatial_model", overwrite=True)

    log(f"[{sid}] Deconvolution complete", step_start)
    return adata


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline_start = datetime.now()
    log("=" * 55)
    log("Pipeline started")
    log(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    log("=" * 55)

    if (NEED2TRAIN):
        log("Loading scRNA-seq reference...")
        t = datetime.now()
        sc_ref = sc.read_h5ad(SC_REF_PATH)
        log(f"Reference loaded — shape: {sc_ref.shape}", t)

        log("Training reference model...")
        inf_aver = train_reference_model(sc_ref)
        inf_aver.to_csv(f"{OUTPUT_DIR}/inf_aver_signatures.csv")
    else:
        inf_aver = pd.read_csv(
            f"{OUTPUT_DIR}/inf_aver_signatures.csv",
            index_col=0
        )       
    for sid in SPATIAL_SAMPLES:
        path = f"data/processed/{sid}_processed.h5ad"
        if not os.path.exists(path):
            log(f"Skipping {sid} — processed file not found")
            continue

        t = datetime.now()
        log(f"Loading {sid}...")
        adata = sc.read_h5ad(path)
        adata.X = adata.layers["counts"]

        print(adata.X.max())   # should be integers like 50, 200, not 1.0 or 0.93
        print(adata.X.min())   # should be 0

        log(f"{sid} loaded — shape: {adata.shape}", t)

        adata = deconvolve_sample(adata, inf_aver, sid)
        adata.write_h5ad(f"{OUTPUT_DIR}/{sid}_deconvolved.h5ad")
        log(f"Saved → {OUTPUT_DIR}/{sid}_deconvolved.h5ad")

    log("=" * 55)
    log("Pipeline complete", pipeline_start)
    log("=" * 55)