"""Predictions and diagnostic tables from the retained fitted models."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from formulaic import model_matrix
from scipy import sparse
from scipy.stats import nbinom

from pseudotimede_py import FittedMarginalModel, fit_likelihood_improvement
from pseudotimede_py.regression import total_loglik
from pseudotimede_py.diagnostics import model_fit_tables


@pytest.fixture
def fits():
    pt = np.linspace(0, 1, 30)**3  # deliberately nonuniform training knots
    data = ad.AnnData(
        sparse.csr_matrix(np.random.default_rng(0).poisson(5, (30, 2)).astype(float)),
        obs=pd.DataFrame({'pseudotime': pt}, index=[f'C{i}' for i in range(30)]),
        var=pd.DataFrame(index=['A', 'B']),
    )
    # A differently ordered cell subset and gene subset must remain aligned.
    selected = data.obs.pseudotime.iloc[::-1][::2]
    summary, models = fit_likelihood_improvement(
        data, selected, ['B', 'A'], max_epochs=2, device='cpu', return_models=True,
    )
    return summary, models


def test_predictions_use_training_design_and_order(fits):
    _, models = fits
    model = models['alternative']
    assert isinstance(model, FittedMarginalModel)
    assert model.template is model.marginal.adata
    got = model.predict()
    with torch.no_grad():
        expected = model.marginal.predict(model.marginal.loader.dataset.x)
    for name in got:
        assert isinstance(got[name], np.ndarray)
        np.testing.assert_allclose(got[name], expected[name].numpy())
    shuffled = model.template.obs.iloc[::-1]
    np.testing.assert_allclose(model.predict(shuffled)['mean'], got['mean'][::-1])


def test_grid_reuses_training_knots(fits):
    _, models = fits
    model = models['alternative']
    with torch.no_grad():
        model.marginal.predict.coefs['mean'].copy_(torch.arange(12).reshape(6, 2) / 10)
    pt = model.template.obs.pseudotime
    grid = pd.DataFrame({'pseudotime': np.linspace(pt.min(), pt.max(), 50)})
    formula = model.marginal.formula['mean']
    training = model_matrix(formula, model.template.obs)
    design = training.model_spec.get_model_matrix(grid).to_numpy()
    coefs = model.marginal.predict.coefs['mean'].detach().numpy()
    got = model.predict(grid)['mean']
    np.testing.assert_allclose(got, np.exp(design @ coefs), rtol=1e-6)
    wrong = np.exp(model_matrix(formula, grid).to_numpy() @ coefs)
    assert not np.allclose(got, wrong)


def test_tidy_tables_counts_residuals_and_bounds(fits):
    summary, models = fits
    obs, curve, residual = model_fit_tables(models, ['A'], n_grid=12)
    assert list(summary.index) == ['B', 'A']
    assert len(obs) == 15 and len(curve) == 24 and len(residual) == 30
    model = models['alternative']
    assert obs.cell.tolist() == model.template.obs_names.tolist()
    y = model.template[:, 'A'].X.toarray().ravel()
    np.testing.assert_array_equal(obs['count'], y)
    r = residual.query("model == 'alternative'")
    np.testing.assert_allclose(r.pearson, (y - r['mean']) / np.sqrt(r['mean'] + r['mean']**2 / r.dispersion))
    np.testing.assert_allclose(curve.lower, nbinom.ppf(.05, curve.dispersion, curve.dispersion / (curve.dispersion + curve['mean'])))
    assert (curve.lower <= curve.upper).all()
    null = curve.query("model == 'null'")
    np.testing.assert_allclose(null['mean'], null['mean'].iloc[0])


def test_selected_genes_and_cells_stay_aligned(fits):
    _, models = fits
    model = models['alternative']  # original gene order is B, A
    selected = model.select_genes(['A', 'B'])
    assert selected is not model
    assert selected.marginal is model.marginal
    assert list(model.template.var_names) == ['B', 'A']
    assert list(selected.template.var_names) == ['A', 'B']
    np.testing.assert_array_equal(selected.template.X.toarray(), model.template.X.toarray()[:, ::-1])
    obs = model.template.obs.iloc[[4, 1, 0]]
    for name, values in model.predict(obs).items():
        np.testing.assert_allclose(selected.predict(obs)[name], values[:, ::-1])
    subset = model.select_genes(['A'])
    assert list(subset.template.var_names) == ['A']
    np.testing.assert_allclose(subset.predict()['mean'], model.predict()['mean'][:, [1]])


def test_wrappers_retain_exact_likelihood_fits(fits):
    summary, models = fits
    for name, model in models.items():
        np.testing.assert_array_equal(total_loglik(model.marginal), summary[f'loglik_{name}'])
    np.testing.assert_array_equal(
        summary.statistic, 2 * (summary.loglik_alternative - summary.loglik_null),
    )
