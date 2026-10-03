"""Spline fitting and likelihood behavior."""

import anndata as ad
import numpy as np
import pandas as pd
import torch
from formulaic import model_matrix
from scipy.stats import nbinom

from pseudotimede_py.regression import (
    fit_likelihood_improvement,
    fit_marginal,
    mean_formula,
    total_loglik,
)


def _toy_adata(n_cells=25, n_genes=2):
    counts = np.random.default_rng(0).poisson(5.0, (n_cells, n_genes)).astype(float)
    return ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {"pseudotime": np.linspace(0.0, 1.0, n_cells)},
            index=[f"C{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]),
    )


def test_spline_basis_dimension_matches_df():
    obs = pd.DataFrame({"pseudotime": np.linspace(0.0, 1.0, 20)})
    mat = model_matrix(mean_formula(df=5, degree=3), obs)
    assert mat.shape[1] == 1 + 5  # intercept + df basis columns


def test_total_loglik_matches_negative_binomial_reference():
    y = np.array([0.0, 2.0, 5.0, 1.0, 8.0, 3.0, 0.0, 4.0])
    adata = _toy_adata(n_cells=len(y), n_genes=1)
    adata.X = y[:, None]
    model = fit_marginal(
        adata, {"mean": "~ 1", "dispersion": "~ 1"}, max_epochs=0, device="cpu"
    )

    mu, r = 3.5, 2.0
    with torch.no_grad():
        model.predict.coefs["mean"].copy_(torch.log(torch.tensor([[mu]])))
        model.predict.coefs["dispersion"].copy_(torch.log(torch.tensor([[r]])))

    expected = nbinom(n=r, p=r / (r + mu)).logpmf(y).sum()
    np.testing.assert_allclose(total_loglik(model), [expected], rtol=1e-5)


def test_observed_fit_returns_gene_aligned_likelihood_improvement():
    adata = _toy_adata()
    genes = ["G1", "G0"]

    result = fit_likelihood_improvement(
        adata, adata.obs["pseudotime"], genes, max_epochs=20, device="cpu"
    )

    assert list(result.index) == genes
    assert (result["n_cells"] == adata.n_obs).all()
    assert np.isfinite(result[["statistic", "loglik_alternative", "loglik_null"]]).all().all()
    np.testing.assert_array_equal(
        result["statistic"], 2 * (result["loglik_alternative"] - result["loglik_null"])
    )


def test_fitting_does_not_mutate_input():
    adata = _toy_adata()
    original = adata.copy()
    pseudotime = adata.obs["pseudotime"].copy()

    fit_likelihood_improvement(
        adata, pseudotime, list(adata.var_names), max_epochs=5, device="cpu"
    )

    pd.testing.assert_frame_equal(adata.obs, original.obs)
    np.testing.assert_array_equal(adata.X, original.X)
    pd.testing.assert_series_equal(pseudotime, original.obs["pseudotime"])


def test_fitting_works_on_a_reordered_cell_subset():
    adata = _toy_adata(n_cells=30)
    subset_pt = adata.obs["pseudotime"].iloc[[3, 7, 1, 25, 10, 14, 22]]

    result = fit_likelihood_improvement(
        adata, subset_pt, list(adata.var_names), df=4, max_epochs=10, device="cpu"
    )

    assert (result["n_cells"] == len(subset_pt)).all()
    assert np.isfinite(result["statistic"]).all()
