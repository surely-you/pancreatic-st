# Annotations on script_deconvolution.py
This workflow describes the process of cell-type deconvolution for Spatial Transcriptomics. Because a single Visium spot (55µm) usually contains multiple cells (typically 1–20), we cannot assume one spot equals one cell. This script uses Cell2location, a Bayesian model, to integrate single-cell RNA-seq (scRNA-seq) "signatures" into spatial data to estimate exactly which cell types are present in each spot.

## Summary 
1. Reference Mapping
2. Deconvonlution

## Background Info: Cell2location
### Model Logic: How Bayesian deconvolution works.
"Bayesian" means the model uses prior beliefs to stay grounded in biological reality.Instead of just guessing any number, we give the model "Priors":
* **Cell Density Prior** (*N_CELLS_PER_LOC*): expected number of cells per spot
  * acts as a regularizer: Without this, the model might try to explain a very high signal by jamming 100 cells into one spot, which is biologically impossible.
  * High Density (e.g., Lymph node): Set to 15–20.
  * Low Density (e.g., Fatty tissue): Set to 2–5.
  * Standard (e.g., PDAC/Tumor): Set to 8–12.
* **Technical Noise Prior** (*DETECTION_ALPHA*): controls how much the model accounts for technical variation in gene detection.
  * represents the "sensitivity" of the spatial technology compared to the single-cell reference.
  * Lower values (e.g., 20): The model is "relaxed." It assumes there is a lot of technical noise and will be more flexible in how it maps cells.
  * Higher values (e.g., 200): The model is "strict." It assumes the gene detection is quite consistent. This is generally recommended for modern Visium datasets to prevent the model from over-fitting to noise.
* **Gene Sensitivity**: "Some genes are captured more easily than others."

COGS 108 ML & overfitting (https://github.com/COGS108/Lectures-Sp26/blob/main/17-Machine-Learning.pdf)

### Posterior Estimation
The model runs hundreds of iterations (# of iterations = *EPOCHS_REF*). In each iteration, it tries to solve the puzzle:
* Guess: It guesses the number of each cell type in a spot.
* Simulate: It multiplies those guesses by the scRNA-seq signatures to see if the result matches the real spatial data.
* Adjust: If the guess was too high for a specific gene, it lowers the estimate for the cell types that express that gene.
* Final Result (Posterior): After ~30,000 rounds, the model provides a probability distribution of cell abundance. It doesn't just say "there are 5 T-cells"; it says "based on the evidence and our priors, the most likely number of T-cells is 5.2."


### Cell2location 
works in two distinct phases:
* **The Reference Model**: It looks at a single-cell atlas and learns what a "T-cell," "Fibroblast," or "Tumor Cell" looks like in terms of average gene expression ($mu$).
* **The Spatial Model**: It looks at the mixed signal in a Visium spot and calculates the combination of those signatures that best explains the observed counts, accounting for technical noise ($\alpha$).

## Steps
1. Setting things up
2. Reference Mapping: Establishing gene expression signatures for cell types.
3. Spatial Mapping: Projecting signatures onto tissue spots.
4. Visualization: Interpreting cell abundance maps.

## Setting things up
new libraries: 
* **cell2location**: the stuff above
* **filter_genes**: preprocessing step used to remove low-quality or uninformative genes from the dataset (used under section Reference Mapping)
```
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cell2location
from cell2location.utils.filtering import filter_genes
import os

# ── Config ───────────────────────────────────────────────────────────────────
SC_REF_PATH  = "data/reference/pdac_scrna_reference.h5ad"  # scRNA-seq reference
SPATIAL_DIR  = "data/processed"
OUTPUT_DIR   = "data/deconvolved"
FIGURE_DIR   = "figures/deconvolution"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

SPATIAL_SAMPLES = [
    "GSE254829", "GSE233293", "GSE327056",
    "GSE274103", "GSE310353", "Syn61831984",
    "GSE278694", "GSE272362", "GSE274557",
]
```
## Reference Mapping: Establishing gene expression signatures for cell types.
Before looking at the tissue, we must "teach" the model our cell types. This requires high-quality scRNA-seq data where every cell is already labeled.
This code segment is the "Learning Phase" (Reference Signature Estimation) of the cell2location pipeline. It uses Negative Binomial (NB) regression to transform a raw scRNA-seq atlas into a clean dictionary of cell-type signatures
* **Negative Binomial distribution (NBD)**: probability distribution used to model count data
  * biological data is "noisy." In real tissue samples, we see that the variance is often much higher than the mean ($Var > Mean$). This is called **overdispersion**. The NB distribution adds a "dispersion parameter" that allows the model to stretch and account for this extra noise.
  * **The "Zero" Problem**: Count data is strictly non-negative, cuz u can't have $-5$ counts of a gene. NB handles the "long tail" of low-expression genes (where many spots have 0 counts) much better than a standard Bell Curve (Normal distribution).
  * source (warning: contains very mathy math): https://www.datacamp.com/tutorial/negative-binomial-distribution
 
*filter_genes*: 
- **cell_count_cutoff**: A gene must be expressed in at least x cells (absolute count)
- **cell_percentage_cutoff2**: A gene must be expressed in at least x% of cells in at least one cell type
- **nonz_mean_cutoff**:  mean expression among cells that DO express the gene must be > x
```

CELL_TYPE_COL   = "cell_type"    # column in sc_ref.obs with cell type labels
EPOCHS_REF      = 250            # Number of training iterations for the reference

def train_reference_model(sc_ref: ad.AnnData):

# that package we imported earlier
# Filter genes to reduce noise (removing genes expressed in too few cells)
    selected = filter_genes(sc_ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03,
                            nonz_mean_cutoff=1.12)
    sc_ref = sc_ref[:, selected].copy()

# Setup the model to account for 'batch effects' (different patients/samples)
    cell2location.models.RegressionModel.setup_anndata(
        sc_ref,
        batch_key="sample_id",
        labels_key=CELL_TYPE_COL,
    )

# Train the model to learn per-cell-type signatures
    ref_model = cell2location.models.RegressionModel(sc_ref) # initializes a Bayesian model that assumes gene expression follows a NBD
    ref_model.train(max_epochs=EPOCHS_REF)

# Export posterior — inf_aver is the per-cell-type expression signature
    sc_ref = ref_model.export_posterior(sc_ref, sample_kwargs={"num_samples": 1000}) # takes 1,000 random samples from that distribution to calculate the mean (average) expression. This ensures the signature is statistically robust and not skewed by a few outlier cells.

# extracts the results from the hidden layers of the AnnData object
# Result: a clean DataFrame (inf_aver) where every column is a cell type "fingerprint."
    inf_aver = sc_ref.varm["means_per_cluster_mu_fg"][ #table where rows are genes and columns are the Inferred Average ($\mu$) expression per cell type
        [f"means_per_cluster_mu_fg_{ct}" for ct in sc_ref.uns["mod"]["factor_names"]]
    ].copy()
    inf_aver.columns = sc_ref.uns["mod"]["factor_names"]

    #saving the model
    ref_model.save("models/reference_model", overwrite=True)
    return inf_aver

```
## Deconvonlution
"Solving Phase." It takes the "Dictionary" from in the previous step and applies it to the real spatial tissue samples to determine exactly where each cell type is located.
* *EPOCHS_SPATIAL = 30000*: defines how many times the model will iterate over the spatial data to refine its estimates.
  * Cell2location uses [Variational Inference](https://towardsdatascience.com/variational-inference-the-basics-f70ac511bcea/) (a type of Bayesian machine learning). Unlike simple models that converge quickly, Bayesian models start with a "fuzzy" guess and slowly sharpen it.
  * The "Burn-in" Process:
     * **Epochs 1–5,000**: The model is mostly figuring out the background noise and the "big" cell types (e.g., "This spot is definitely mostly Tumor").
     * **Epochs 5,000–20,000**: The model begins to distinguish between very similar cell types (e.g., "Is this an Inflammatory Fibroblast or a Myofibroblast?").
     * **Epochs 20,000–30,000**: The model fine-tunes the absolute abundance. It ensures that the predicted counts (e.g., 2.4 cells) are mathematically stable and that the "loss function" (the error rate) has flattened out.
```
# ── Step 2: Deconvolve each spatial sample ───────────────────────────────────
N_CELLS_PER_LOC = 8              # expected avg cells per Visium spot
DETECTION_ALPHA = 200            # cell2location hyperparameter
EPOCHS_SPATIAL  = 30000

def deconvolve_sample(adata: ad.AnnData, inf_aver: pd.DataFrame, sid: str) -> ad.AnnData:
    """Run cell2location spatial mapping for one sample."""
    # Keep only genes shared between reference and spatial data
    shared = inf_aver.index.intersection(adata.var_names)
    adata  = adata[:, shared].copy()
    inf_av = inf_aver.loc[shared]

    cell2location.models.Cell2location.setup_anndata(adata, batch_key=None)
    model = cell2location.models.Cell2location(
        adata,
        cell_state_df=inf_av,                   # handing the model the "Dictionary" (signatures) thats just trained.
        N_cells_per_location=N_CELLS_PER_LOC,   # prevents the model from mathematically over-fitting (e.g., trying to put 50 cells in one tiny spot).
        detection_alpha=DETECTION_ALPHA,        # sets the confidence level for technical noise
    )

    model.train(
        max_epochs=EPOCHS_SPATIAL,
        batch_size=None,
        train_size=1,
        use_gpu=True,
    )

   # Extracting the Results: takes the estimated cell counts and adds them directly to sample's metadata (adata.obs)
   # Result: If the dictionary had "T-cell" and "Fibroblast," the adata.obs now has columns named "T-cell" and "Fibroblast." Each row (spot) now has a number, like 2.4 or 0.1, representing how many of that cell type are in that spot.
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
```
## Main function that calls all aforementioned functions
```
# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading scRNA-seq reference...")
    sc_ref = sc.read_h5ad(SC_REF_PATH)

    print("Training reference model...")
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
```
