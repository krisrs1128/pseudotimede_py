"""Exploratory diagnostics on the NB fits

Our PseudotimeDE test statistics depend on the quality of the fitted NB models.
This module includes helpers for supporting model diagnostics for both the null
and alternative models.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import nbinom


def model_fit_tables(models, genes, *, n_grid=200, interval=0.9):
    """Return summary statistics of the fitted model

    The input `models` should be computed by
    fit_likelihood_improvement(return_models=True).  The returned counts come
    are the training samples used during modeling. The curves are the
    prediction intervals for the fitted NB models.
    """
    genes = list(genes)
    observed, curves, residuals = [], [], []

    for name in ("alternative", "null"):

        # model-level structures
        model = models[name]
        pt = model.template.obs["pseudotime"].to_numpy()
        grid = pd.DataFrame({"pseudotime": np.linspace(pt.min(), pt.max(), n_grid)})
        curve = model.predict(grid)
        at_cells = model.predict()
        tail = (1 - interval) / 2

        # statistics for each gene
        for gene in genes:
            j = model.template.var_names.get_loc(gene)
            counts = model.template.X[:, j]
            y = counts.toarray().ravel() if sparse.issparse(counts) else np.asarray(counts).ravel()

            if name == "alternative":
                observed.append(pd.DataFrame({
                    "cell": model.template.obs_names, "gene": gene,
                    "pseudotime": pt, "count": y,
                }))

            mu, r = curve["mean"][:, j], curve["dispersion"][:, j]
            curves.append(pd.DataFrame({
                "gene": gene, "model": name, "pseudotime": grid.pseudotime,
                "mean": mu, "dispersion": r,
                "lower": nbinom.ppf(tail, r, r / (r + mu)),
                "upper": nbinom.ppf(1 - tail, r, r / (r + mu)),
            }))

            mu_cell, r_cell = at_cells["mean"][:, j], at_cells["dispersion"][:, j]
            residuals.append(pd.DataFrame({
                "cell": model.template.obs_names, "gene": gene, "model": name,
                "pseudotime": pt, "mean": mu_cell, "dispersion": r_cell,
                "pearson": (y - mu_cell) / np.sqrt(mu_cell + mu_cell**2 / r_cell),
            }))
    return tuple(pd.concat(parts, ignore_index=True) for parts in (observed, curves, residuals))
