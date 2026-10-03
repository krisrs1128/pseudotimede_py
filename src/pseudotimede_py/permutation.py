"""PseudotimeDE Permutation Tests

PseudotimeDE computes p-values using a kind of permutation test. To simulate
datasets under the null, we randomly permute the estimated pseudotimes. We then
fit both the alternative and null models on the permuted dataset. This gives a
sense of how much the likelihood of the alternative might increase relative to
the null even when there is no genuine pseudotime structure present. We share
the same permutations across all genes.
"""
import anndata as ad
import numpy as np
import pandas as pd
import torch

from .regression import fit_likelihood_improvement


def subsample_pseudotimes(adata: ad.AnnData, key: str = "pseudotime_subsamples") -> list[pd.Series]:
    """Inferred pseudotimes across subsamples

    As input to PseudotimeDE, we assume that pseudotimes have been estimated
    across many subsamples and saved to the `adata.obsm[key]` slot. The purpose
    of these subsampled pseudotimes is to account for the uncertainty in the
    pseudotime inference step.

    The `adata.obsm[key]` object should have shape cells x replicates. This code
    will split the replicates into separate series for downstream analysis.
    """
    table = adata.obsm[key]
    return [table[column].dropna() for column in table.columns]

RESULT_COLUMNS = [
    "n_cells",
    "loglik_alternative",
    "loglik_null",
    "statistic",
    "converged_alternative",
    "converged_null",
]


def shuffle_pseudotime(pseudotime: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Permute the pseudotimes"""
    shuffled_values = rng.permutation(pseudotime.to_numpy())
    return pd.Series(shuffled_values, index=pseudotime.index, name=pseudotime.name)


def _failed_replicate_records(genes: list[str], n_cells: int) -> list[dict]:
    return [
        {
            "gene": gene,
            "n_cells": n_cells,
            "loglik_alternative": np.nan,
            "loglik_null": np.nan,
            "statistic": np.nan,
            "converged_alternative": False,
            "converged_null": False,
        }
        for gene in genes
    ]


def run_null_fits(
    adata: ad.AnnData,
    replicates: list[pd.Series],
    genes: list[str],
    *,
    seed: int = 0,
    df: int = 5,
    degree: int = 3,
    dispersion_formula: str = "~ 1",
    device: str | torch.device | None = None,
    max_epochs: int = 1000,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    convergence_tol: float = 1e-3,
    convergence_window: int = 10,
) -> pd.DataFrame:
    """Return test statistics on permutation null

    This wraps the overall permutation strategy. We permute each replicate, fit
    the pair of models, and store the null statistics in a DataFrame with shape
    replicate x gene. The statistics are exactly 2 * (ll_alt - ll_null)
    """
    genes = list(genes)
    child_seeds = np.random.SeedSequence(seed).spawn(len(replicates))

    # loop over permutations
    records = []
    for replicate_id, (pseudotime, child_seed) in enumerate(
        zip(replicates, child_seeds), start=1
    ):
        # create the null data
        rng = np.random.default_rng(child_seed)
        permuted = shuffle_pseudotime(pseudotime, rng)

        try:
            # get the test statistic
            fit = fit_likelihood_improvement(
                adata, permuted, genes,
                df=df, degree=degree, dispersion_formula=dispersion_formula,
                device=device, max_epochs=max_epochs, lr=lr, weight_decay=weight_decay,
                convergence_tol=convergence_tol, convergence_window=convergence_window,
            )
            fit = fit.reset_index()
            replicate_records = fit[["gene", *RESULT_COLUMNS]].to_dict("records")

        # don't crash if a permutation failed
        except Exception:
            replicate_records = _failed_replicate_records(genes, len(pseudotime))

        for record in replicate_records:
            record["replicate"] = replicate_id
        records.extend(replicate_records)

    return pd.DataFrame.from_records(records).set_index(["replicate", "gene"])
