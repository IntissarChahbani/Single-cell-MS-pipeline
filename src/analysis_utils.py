# ============================================================
# Reusable functions for donor-aware single-cell RNA-seq analysis
# ============================================================
import scanpy as sc
import pandas as pd
import numpy as np
import gseapy as gp

from sklearn.decomposition import PCA
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
# ============================================================
# Quality control
# ============================================================


def calculate_qc_metrics(adata, mt_prefix="MT-"):
    """
    Calculate basic per-cell QC metrics.

    Parameters
    ----------
    adata
        AnnData object containing raw count data.

    mt_prefix : str
        Prefix used to identify mitochondrial genes.
        For human gene symbols, this is usually "MT-".

    Returns
    -------
    adata
        AnnData object with QC metrics added.
    """

    # Mark mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)

    # Calculate QC metrics for every cell
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True
    )

    return adata#

def filter_cells_qc(adata, min_genes, max_pct_mt):
    """
    Filter cells using detected-gene and mitochondrial-percentage thresholds.

    Parameters
    ----------
    adata
        AnnData object containing calculated QC metrics.

    min_genes : int
        Minimum number of detected genes required for a cell.

    max_pct_mt : float
        Maximum allowed percentage of mitochondrial counts.

    Returns
    -------
    adata_filtered
        AnnData object containing only cells that pass QC.
    """

    # Identify cells that pass both QC conditions
    qc_pass = (
        (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["pct_counts_mt"] < max_pct_mt)
    )

    # Keep only cells that pass QC
    adata_filtered = adata[qc_pass].copy()

    return adata_filtered

def detect_doublets(adata, batch_key):
    """
    Detect potential doublets using Scrublet.

    Doublet detection is performed separately for each
    sample/library defined by batch_key.

    Parameters
    ----------
    adata
        AnnData object containing count data.

    batch_key : str
        Column in adata.obs identifying the sample/library.

    Returns
    -------
    adata
        AnnData object with Scrublet doublet predictions added.
    """

    # Run Scrublet separately for each sample/library
    sc.pp.scrublet(
        adata,
        batch_key=batch_key
    )

    return adata

def filter_predicted_doublets(
    adata,
    doublet_col="predicted_doublet"
):
    """
    Remove cells predicted to be doublets.

    Parameters
    ----------
    adata
        AnnData object containing doublet predictions.

    doublet_col : str
        Column in adata.obs containing the doublet prediction.

    Returns
    -------
    adata_filtered
        AnnData object containing only predicted singlets.
    """

    # Keep cells that are NOT predicted doublets
    adata_filtered = adata[
        ~adata.obs[doublet_col]
    ].copy()

    return adata_filtered
def normalize_log1p(adata, target_sum=1e4):
    """
    Normalize counts per cell and log-transform expression values.

    Parameters
    ----------
    adata
        AnnData object containing count data in adata.X.

    target_sum : float
        Total count to which each cell is normalized.

    Returns
    -------
    adata
        AnnData object with normalized and log-transformed values in adata.X.
    """

    # Normalize each cell to the same total count
    sc.pp.normalize_total(
        adata,
        target_sum=target_sum
    )

    # Log-transform the normalized values
    sc.pp.log1p(adata)

    return adata

def select_hvgs(adata, n_top_genes=3000, flavor="seurat", batch_key=None):
    """
    Identify highly variable genes for dimensionality reduction
    and clustering.

    Parameters
    ----------
    adata
        AnnData object containing normalized expression values.

    n_top_genes : int
        Number of highly variable genes to select.

    flavor : str
        Method used for highly variable gene selection.

    batch_key : str or None
        Column in adata.obs identifying samples or batches.

    Returns
    -------
    adata
        AnnData object with highly variable genes marked in
        adata.var["highly_variable"].
    """

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        batch_key=batch_key
    )

    return adata

def run_pca(
    adata,
    n_comps=50,
    use_highly_variable=True
):
    """
    Run principal component analysis (PCA).

    Parameters
    ----------
    adata
        AnnData object containing normalized expression data.

    n_comps : int
        Number of principal components to calculate.

    use_highly_variable : bool
        Whether PCA should use only highly variable genes.

    Returns
    -------
    adata
        AnnData object with PCA coordinates stored in
        adata.obsm["X_pca"].
    """

    sc.tl.pca(
        adata,
        n_comps=n_comps,
        use_highly_variable=use_highly_variable
    )

    return adata
def build_neighbor_graph(
    adata,
    n_neighbors=15,
    n_pcs=30
):
    """
    Build a nearest-neighbor graph from PCA coordinates.

    Parameters
    ----------
    adata
        AnnData object with PCA already calculated.

    n_neighbors : int
        Number of neighboring cells used to construct
        the local neighborhood graph.

    n_pcs : int
        Number of principal components used to calculate
        cell-cell similarity.

    Returns
    -------
    adata
        AnnData object with the neighbor graph stored.
    """

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs
    )

    return adata

def run_umap(adata):
    """
    Calculate a two-dimensional UMAP embedding from the
    previously constructed neighbor graph.

    Parameters
    ----------
    adata
        AnnData object containing a neighbor graph.

    Returns
    -------
    adata
        AnnData object with UMAP coordinates stored in
        adata.obsm["X_umap"].
    """

    sc.tl.umap(adata)

    return adata
def run_leiden(
    adata,
    resolution=0.5,
    key_added="leiden",
    flavor="igraph",
    n_iterations=2,
    directed=False
):
    """
    Cluster cells using the Leiden community-detection algorithm.

    Parameters
    ----------
    adata
        AnnData object containing a neighbor graph.

    resolution : float
        Controls clustering granularity. Higher values generally
        produce more clusters.

    key_added : str
        Name of the column in adata.obs where cluster labels
        will be stored.

    flavor : str
        Leiden implementation to use.

    n_iterations : int
        Number of Leiden optimization iterations.

    directed : bool
        Whether to treat the graph as directed.

    Returns
    -------
    adata
        AnnData object with Leiden cluster labels added to adata.obs.
    """

    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=key_added,
        flavor=flavor,
        n_iterations=n_iterations,
        directed=directed
    )

    return adata

def find_cluster_markers(
    adata,
    groupby,
    method="wilcoxon",
    use_raw=False
):
    """
    Identify marker genes for groups of cells.

    Parameters
    ----------
    adata
        AnnData object containing normalized expression data.

    groupby : str
        Column in adata.obs defining the groups to compare,
        for example Leiden clusters.

    method : str
        Statistical method used to rank marker genes.

    use_raw : bool
        Whether to use adata.raw instead of adata.X.

    Returns
    -------
    adata
        AnnData object containing ranked marker-gene results.
    """

    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        use_raw=use_raw
    )

    return adata

def calculate_cell_composition(
    obs,
    donor_col,
    cell_type_col,
    group_cols=None
):
    """
    Calculate cell-type proportions within each biological sample.

    Missing sample × cell-type combinations are represented as
    zero cells rather than missing values.

    Parameters
    ----------
    obs : pandas.DataFrame
        Cell-level metadata, typically adata.obs.

    donor_col : str
        Column identifying biological donors.

    cell_type_col : str
        Column containing cell-type annotations.

    group_cols : list of str, optional
        Additional columns defining the sample structure,
        such as disease status or compartment.

    Returns
    -------
    pandas.DataFrame
        Complete sample-by-cell-type composition table containing
        cell counts, total cells, and percentages.
    """

    if group_cols is None:
        group_cols = []

    sample_cols = group_cols + [donor_col]
    count_cols = sample_cols + [cell_type_col]

    # --------------------------------------------------------
    # Count cells of each cell type in each biological sample
    # --------------------------------------------------------
    cell_counts = (
        obs
        .groupby(count_cols, observed=True)
        .size()
        .reset_index(name="n_cells")
    )

    # --------------------------------------------------------
    # Calculate the total number of cells in each sample
    # --------------------------------------------------------
    sample_totals = (
        obs
        .groupby(sample_cols, observed=True)
        .size()
        .reset_index(name="total_cells")
    )

    # --------------------------------------------------------
    # Create every sample × cell-type combination
    #
    # This ensures that an undetected cell type is represented
    # by n_cells = 0 rather than by a missing row.
    # --------------------------------------------------------
    samples = obs[sample_cols].drop_duplicates()

    cell_types = (
        obs[cell_type_col]
        .dropna()
        .unique()
    )

    complete = (
        samples.assign(_key=1)
        .merge(
            pd.DataFrame({
                cell_type_col: cell_types,
                "_key": 1
            }),
            on="_key"
        )
        .drop(columns="_key")
    )

    # --------------------------------------------------------
    # Add observed cell counts
    # --------------------------------------------------------
    composition = complete.merge(
        cell_counts,
        on=count_cols,
        how="left"
    )

    composition["n_cells"] = (
        composition["n_cells"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Add sample totals
    # --------------------------------------------------------
    composition = composition.merge(
        sample_totals,
        on=sample_cols,
        how="left"
    )

    # --------------------------------------------------------
    # Calculate percentage within each sample
    # --------------------------------------------------------
    composition["percentage"] = (
        composition["n_cells"]
        / composition["total_cells"]
        * 100
    )

    return composition

def select_analysis_cells(
    adata,
    cell_type,
    groups,
    compartment=None,
    library=None,
    cell_type_col="cell_type",
    group_col="disease_status",
    compartment_col="compartment",
    library_col="library",
):
    """
    Select cells for a downstream biological analysis.

    Parameters
    ----------
    adata : AnnData
        Annotated single-cell dataset.

    cell_type : str
        Cell type to retain.

    groups : list of str
        Biological groups to retain, for example
        ["Healthy", "Relapsing MS"].

    compartment : str, optional
        Compartment or tissue to retain.

    library : str, optional
        Library type or sequencing chemistry to retain.

    cell_type_col : str
        Column containing cell-type annotations.

    group_col : str
        Column containing biological group labels.

    compartment_col : str
        Column containing compartment labels.

    library_col : str
        Column containing library information.

    Returns
    -------
    AnnData
        A copy containing only cells matching the requested
        analysis criteria.
    """

    mask = np.ones(adata.n_obs, dtype=bool)

    mask &= adata.obs[cell_type_col] == cell_type
    mask &= adata.obs[group_col].isin(groups)

    if compartment is not None:
        mask &= adata.obs[compartment_col] == compartment

    if library is not None:
        mask &= adata.obs[library_col] == library

    return adata[mask].copy()
def create_pseudobulk(
    adata,
    raw_counts,
    donor_col,
    metadata_cols=None
):
    """
    Aggregate cell-level raw counts into donor-level pseudobulk samples.

    Raw counts are summed across all selected cells belonging
    to the same biological donor.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the selected analysis cells.

    raw_counts : array-like or sparse matrix
        Raw cell-by-gene count matrix aligned with adata.

    donor_col : str
        Column in adata.obs identifying biological donors.

    metadata_cols : list of str, optional
        Additional donor-level metadata columns to preserve,
        such as disease status, compartment, or library type.

    Returns
    -------
    pseudobulk_counts : pandas.DataFrame
        Donor-by-gene raw-count matrix.

    pseudobulk_meta : pandas.DataFrame
        Donor-level metadata including number of cells,
        library size, and number of detected genes.
    """

    if metadata_cols is None:
        metadata_cols = []

    # Make sure counts and metadata contain the same cells
    if raw_counts.shape != adata.shape:
        raise ValueError(
            "raw_counts must have the same shape as adata."
        )

    donor_array = (
        adata.obs[donor_col]
        .astype(str)
        .to_numpy()
    )

    donors = sorted(np.unique(donor_array))

    pseudobulk_blocks = []
    metadata_records = []

    for donor in donors:

        # Find cells belonging to this donor
        positions = np.where(donor_array == donor)[0]

        # Sum raw counts across those cells
        donor_counts = np.asarray(
            raw_counts[positions, :].sum(axis=0)
        ).ravel()

        pseudobulk_blocks.append(donor_counts)

        donor_obs = adata.obs.iloc[positions]

        record = {
            donor_col: donor,
            "n_cells": len(positions),
            "library_size": int(donor_counts.sum()),
            "detected_genes": int(
                np.sum(donor_counts > 0)
            )
        }

        # Preserve requested donor-level metadata
        for col in metadata_cols:

            values = donor_obs[col].dropna().unique()

            if len(values) != 1:
                raise ValueError(
                    f"Donor {donor} has multiple values "
                    f"for metadata column '{col}': {values}"
                )

            record[col] = values[0]

        metadata_records.append(record)

    # Donor × gene count matrix
    pseudobulk_counts = pd.DataFrame(
        np.vstack(pseudobulk_blocks),
        index=donors,
        columns=adata.var_names
    )

    pseudobulk_counts.index.name = donor_col

    # Donor-level metadata
    pseudobulk_meta = (
        pd.DataFrame(metadata_records)
        .set_index(donor_col)
    )

    return pseudobulk_counts, pseudobulk_meta

def filter_low_expression(
    pseudobulk_counts,
    min_count=10,
    min_samples=3
):
    """
    Filter low-expression genes from a pseudobulk count matrix.

    A gene is retained if it has at least `min_count` raw counts
    in at least `min_samples` biological samples.

    Parameters
    ----------
    pseudobulk_counts : pandas.DataFrame
        Biological samples × genes raw-count matrix.

    min_count : int, default=10
        Minimum raw count required for a gene to be considered
        expressed in a sample.

    min_samples : int, default=3
        Minimum number of biological samples in which the gene
        must reach `min_count`.

    Returns
    -------
    pandas.DataFrame
        Filtered pseudobulk count matrix.
    """

    samples_with_expression = (
        pseudobulk_counts >= min_count
    ).sum(axis=0)

    keep_genes = (
        samples_with_expression >= min_samples
    )

    return pseudobulk_counts.loc[:, keep_genes].copy()
def run_pseudobulk_pca(
    pseudobulk_counts,
    n_variable_genes=3000,
    n_components=5
):
    """
    Run exploratory PCA on donor-level pseudobulk counts.

    Counts are transformed as:

        raw counts
            ↓
        CPM normalization
            ↓
        log2(CPM + 1)
            ↓
        most variable genes
            ↓
        PCA

    This transformation is intended for exploratory PCA only.
    The original raw counts remain unchanged for count-based
    differential-expression analysis.

    Parameters
    ----------
    pseudobulk_counts : pandas.DataFrame
        Biological samples × genes raw-count matrix.

    n_variable_genes : int, default=3000
        Number of most variable genes used for PCA.

    n_components : int, default=5
        Number of principal components to calculate.

    Returns
    -------
    pca_coordinates : pandas.DataFrame
        PCA coordinates for each biological sample.

    explained_variance : numpy.ndarray
        Fraction of variance explained by each PC.
    """

    # Calculate library sizes
    library_sizes = pseudobulk_counts.sum(axis=1)

    if (library_sizes <= 0).any():
        raise ValueError(
            "All pseudobulk samples must have positive library sizes."
        )

    # CPM normalization
    cpm = (
        pseudobulk_counts
        .div(library_sizes, axis=0)
        * 1_000_000
    )

    # Log transformation
    log_cpm = np.log2(cpm + 1)

    # Select most variable genes
    gene_variance = log_cpm.var(axis=0)

    top_genes = (
        gene_variance
        .sort_values(ascending=False)
        .head(n_variable_genes)
        .index
    )

    pca_input = log_cpm[top_genes]

    # PCA
    pca = PCA(n_components=n_components)

    coordinates = pca.fit_transform(
        pca_input
    )

    # Store coordinates
    pca_coordinates = pd.DataFrame(
        coordinates,
        index=pseudobulk_counts.index,
        columns=[
            f"PC{i + 1}"
            for i in range(n_components)
        ]
    )

    return (
        pca_coordinates,
        pca.explained_variance_ratio_
    )

def fit_deseq2(
    pseudobulk_counts,
    metadata,
    design,
    refit_cooks=True
):
    """
    Fit a PyDESeq2 model to donor-level pseudobulk counts.

    Parameters
    ----------
    pseudobulk_counts : pandas.DataFrame
        Biological samples × genes raw-count matrix.

    metadata : pandas.DataFrame
        Sample-level metadata. Its index must correspond to
        the rows of pseudobulk_counts.

    design : str
        DESeq2 design formula, for example:
        "~ disease_status"

    refit_cooks : bool, default=True
        Whether PyDESeq2 should refit genes affected by
        Cook's-distance outliers when appropriate.

    Returns
    -------
    DeseqDataSet
        Fitted PyDESeq2 dataset.
    """

    counts = pseudobulk_counts.copy()
    metadata = metadata.copy()

    # Standardize sample IDs
    counts.index = counts.index.astype(str)
    metadata.index = metadata.index.astype(str)

    # Check sample alignment
    if not counts.index.equals(metadata.index):
        raise ValueError(
            "Count matrix and metadata sample order do not match."
        )

    # DESeq2 requires non-negative integer counts
    if (counts.values < 0).any():
        raise ValueError(
            "Count matrix contains negative values."
        )

    if not np.allclose(
        counts.values,
        np.round(counts.values)
    ):
        raise ValueError(
            "DESeq2 requires raw integer-like counts."
        )

    counts = counts.astype(int)

    # Construct DESeq2 dataset
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design=design,
        refit_cooks=refit_cooks
    )

    # Fit model
    dds.deseq2()

    return dds

def get_deseq2_results(
    dds,
    factor,
    numerator,
    denominator
):
    """
    Extract differential-expression results for a specified
    DESeq2 contrast.

    Positive log2 fold changes indicate higher expression in
    the numerator group relative to the denominator group.

    Parameters
    ----------
    dds : DeseqDataSet
        Fitted PyDESeq2 dataset.

    factor : str
        Metadata variable defining the comparison.

    numerator : str
        Group represented by positive log2 fold changes.

    denominator : str
        Reference group for the comparison.

    Returns
    -------
    pandas.DataFrame
        Differential-expression results sorted by adjusted
        p-value.
    """

    stats = DeseqStats(
        dds,
        contrast=[
            factor,
            numerator,
            denominator
        ]
    )

    stats.summary()

    results = (
        stats.results_df
        .copy()
        .sort_values(
            "padj",
            na_position="last"
        )
    )

    return results

def compare_de_results(
    primary_results,
    sensitivity_results,
    lfc_col="log2FoldChange",
    padj_col="padj",
    padj_threshold=0.05,
    lfc_threshold=0.5
):
    """
    Compare effect-size robustness between a primary and a
    sensitivity differential-expression analysis.

    Parameters
    ----------
    primary_results : pandas.DataFrame
        DE results from the primary analysis.

    sensitivity_results : pandas.DataFrame
        DE results from the sensitivity analysis.

    lfc_col : str, default="log2FoldChange"
        Column containing log2 fold-change estimates.

    padj_col : str, default="padj"
        Column containing adjusted p-values.

    padj_threshold : float, default=0.05
        Adjusted p-value threshold used to define significant
        genes in the primary analysis.

    lfc_threshold : float, default=0.5
        Absolute log2 fold-change threshold used to define
        significant genes in the primary analysis.

    Returns
    -------
    robustness : pandas.DataFrame
        Gene-level comparison of primary and sensitivity results.

    lfc_correlation : float
        Pearson correlation between log2 fold changes across
        genes with finite estimates.

    primary_sig : pandas.DataFrame
        Primary significant genes with sensitivity results and
        direction-agreement information.
    """

    robustness = pd.DataFrame({
        "log2FC_primary":
            primary_results[lfc_col],

        "padj_primary":
            primary_results[padj_col],

        "log2FC_sensitivity":
            sensitivity_results[lfc_col],

        "padj_sensitivity":
            sensitivity_results[padj_col]
    })

    # Keep genes with finite effect-size estimates
    robustness = robustness[
        np.isfinite(robustness["log2FC_primary"])
        &
        np.isfinite(robustness["log2FC_sensitivity"])
    ].copy()

    # Effect-size correlation across tested genes
    lfc_correlation = robustness[
        [
            "log2FC_primary",
            "log2FC_sensitivity"
        ]
    ].corr().iloc[0, 1]

    # Genes significant in the PRIMARY analysis
    primary_sig = robustness[
        (robustness["padj_primary"] < padj_threshold)
        &
        (
            robustness["log2FC_primary"].abs()
            >= lfc_threshold
        )
    ].copy()

    # Direction agreement
    primary_sig["same_direction"] = (
        np.sign(primary_sig["log2FC_primary"])
        ==
        np.sign(primary_sig["log2FC_sensitivity"])
    )

    return robustness, lfc_correlation, primary_sig

def make_ranked_list(
    de_results,
    ranking_col="stat"
):
    """
    Create a ranked gene list from differential-expression results.

    Genes are ranked from the highest to the lowest value of the
    selected ranking statistic.

    For a DESeq2 Wald statistic:
        positive values -> toward the numerator group
        negative values -> toward the denominator group

    Parameters
    ----------
    de_results : pandas.DataFrame
        Differential-expression results indexed by gene.

    ranking_col : str, default="stat"
        Column used to rank genes.

    Returns
    -------
    pandas.DataFrame
        Two-column DataFrame containing gene identifiers and
        ranking statistics, sorted from highest to lowest.
    """

    if ranking_col not in de_results.columns:
        raise ValueError(
            f"Column '{ranking_col}' not found in DE results."
        )

    ranked_genes = (
        de_results[[ranking_col]]
        .copy()
        .reset_index()
    )

    ranked_genes.columns = [
        "gene",
        "score"
    ]

    # Remove missing or infinite ranking statistics
    ranked_genes = ranked_genes[
        np.isfinite(ranked_genes["score"])
    ].copy()

    # Gene identifiers should be unique for ranked enrichment
    if ranked_genes["gene"].duplicated().any():
        raise ValueError(
            "Duplicated gene identifiers found in ranked list."
        )

    # Highest score -> lowest score
    ranked_genes = (
        ranked_genes
        .sort_values(
            "score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return ranked_genes
def run_preranked_gsea(
    ranked_genes,
    gene_sets,
    min_size=15,
    max_size=500,
    permutation_num=1000,
    seed=42
):
    """
    Run preranked GSEA on a ranked gene list.

    Parameters
    ----------
    ranked_genes : pandas.DataFrame
        Two-column DataFrame containing:
        - gene
        - score

        Genes should be ranked from highest to lowest score.

    gene_sets : str
        Gene-set library used for enrichment analysis.

    min_size : int, default=15
        Minimum gene-set size.

    max_size : int, default=500
        Maximum gene-set size.

    permutation_num : int, default=1000
        Number of permutations used to estimate enrichment
        significance.

    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        GSEA results sorted by FDR q-value.
    """

    required_cols = {"gene", "score"}

    if not required_cols.issubset(ranked_genes.columns):
        raise ValueError(
            "ranked_genes must contain 'gene' and 'score' columns."
        )

    gsea_ranking = ranked_genes[
        ["gene", "score"]
    ].copy()

    pre_res = gp.prerank(
        rnk=gsea_ranking,
        gene_sets=gene_sets,
        min_size=min_size,
        max_size=max_size,
        permutation_num=permutation_num,
        seed=seed,
        verbose=False,
        outdir=None
    )

    results = pre_res.res2d.copy()

    # Ensure numerical columns are numeric
    numeric_cols = [
        "ES",
        "NES",
        "NOM p-val",
        "FDR q-val",
        "FWER p-val"
    ]

    for col in numeric_cols:
        if col in results.columns:
            results[col] = pd.to_numeric(
                results[col],
                errors="coerce"
            )

    results = (
        results
        .sort_values(
            "FDR q-val",
            na_position="last"
        )
        .reset_index(drop=True)
    )

    return results

def compare_pathway_results(
    primary_results,
    comparison_results,
    term_col="Term",
    nes_col="NES",
    fdr_col="FDR q-val"
):
    """
    Compare pathway-enrichment results between two analyses.

    Only pathways present in both analyses are retained.

    Parameters
    ----------
    primary_results : pandas.DataFrame
        Pathway-enrichment results from the primary analysis.

    comparison_results : pandas.DataFrame
        Pathway-enrichment results from a sensitivity,
        validation, or other comparison analysis.

    term_col : str, default="Term"
        Column containing pathway names.

    nes_col : str, default="NES"
        Column containing normalized enrichment scores.

    fdr_col : str, default="FDR q-val"
        Column containing FDR-adjusted p-values.

    Returns
    -------
    comparison : pandas.DataFrame
        Matched pathways with NES and FDR values from both
        analyses, plus an indicator of whether the enrichment
        direction is preserved.

    nes_correlation : float
        Pearson correlation between NES values across all
        matched pathways.
    """

    primary = (
        primary_results[
            [term_col, nes_col, fdr_col]
        ]
        .rename(
            columns={
                nes_col: "NES_primary",
                fdr_col: "FDR_primary"
            }
        )
    )

    comparison = (
        comparison_results[
            [term_col, nes_col, fdr_col]
        ]
        .rename(
            columns={
                nes_col: "NES_comparison",
                fdr_col: "FDR_comparison"
            }
        )
    )

    pathway_comparison = primary.merge(
        comparison,
        on=term_col,
        how="inner"
    )

    # Remove pathways without finite NES values
    pathway_comparison = pathway_comparison[
        np.isfinite(pathway_comparison["NES_primary"])
        & np.isfinite(pathway_comparison["NES_comparison"])
    ].copy()

    # Compare enrichment direction
    pathway_comparison["same_direction"] = (
        np.sign(pathway_comparison["NES_primary"])
        ==
        np.sign(pathway_comparison["NES_comparison"])
    )

    # Overall NES concordance
    nes_correlation = pathway_comparison[
        ["NES_primary", "NES_comparison"]
    ].corr().iloc[0, 1]

    return pathway_comparison, nes_correlation