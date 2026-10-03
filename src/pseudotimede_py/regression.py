"""Models and test statistics used by PseudotimeDE

PseudotimeDE works by fitting two negative binomial models for each gene,
similar to a likelihodo ratio test. One model uses only the intercept, the other
uses a spline basis to enable a nonlinear effect of pseudotime. We use the test
statistic,

`T = 2 * (loglik_alternative - loglik_null)`.

We use the `bs` function in `formulaic` to fit the spline. The complexity of the
fit is controlled by `df` and `degree` hyperparameters, which are arguments to
these fitting functions.
"""
import copy

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scdesigner.distributions import NegBin

from .fitted import FittedMarginalModel
def mean_formula(df: int = 5, degree: int = 3, column: str = "pseudotime") -> str:
    return f"~ bs({column}, df={df}, degree={degree})"


def fit_marginal(
    adata: ad.AnnData,
    formula: dict,
    *,
    device: str | torch.device | None = None,
    max_epochs: int = 1000,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    **fit_kwargs,
) -> NegBin:
    """Fit a NegBin marginal model

    We use all cells in our likelihood ratio statistic, and we attempt to find
    the maximum likelihood (not a regularized version). This means that we
    override the scDesigner default `val_frac` and `weight_decay`
    hyperparameters.
    """
    model = NegBin(formula, device=device)
    model.setup_data(adata, batch_size=adata.n_obs, device=model.device)
    model.fit(
        max_epochs=max_epochs, verbose=False, val_frac=0.0,
        lr=lr, weight_decay=weight_decay, **fit_kwargs,
    )
    return model


def total_loglik(model: NegBin) -> np.ndarray:
    """Per-gene log-likelihood

    One important detail is that we use float64 rather than float32. The
    `lgamma` terms in the NB loglikelihood can become quite large when the
    counts are large, and if we don't use high enough precision, this causes
    substantitve (not minor numerical) changes in the test statistic.
    """
    model64 = copy.copy(model)
    model64.predict = copy.deepcopy(model.predict).to("cpu").double().eval()
    total = None
    with torch.no_grad():
        for y, x in model.loader:
            y = y.to("cpu").double()
            x = {name: design.to("cpu").double() for name, design in x.items()}
            ll = model64.likelihood((y, x)).sum(dim=0)
            total = ll if total is None else total + ll
    return total.numpy()


def _converged(fit_history: list[dict], tol: float = 1e-3, window: int = 10) -> bool:
    """Has the model converged in the last `window` epochs?

    val_frac=0 disables scDesigner's validation-based early stopping (it only
    triggers when a validation loss exists), so this is the only convergence
    signal available for likelihood-comparison fits. This is full-batch Adam
    with no learning-rate decay, so a genuinely converged fit still oscillates
    at a noise floor of a few 1e-4 relative; tol is set above that floor
    (checked empirically against the LPS CCL5/CXCL10 fits) rather than at a
    stricter value that would flag every converged fit as not converged.
    """
    if len(fit_history) <= window:
        return False
    losses = [h["train_loss"] for h in fit_history[-(window + 1):]]
    if not all(np.isfinite(loss) for loss in losses):
        return False
    rel_changes = [
        abs(losses[i + 1] - losses[i]) / (abs(losses[i]) + 1e-12)
        for i in range(len(losses) - 1)
    ]
    return max(rel_changes) < tol


def fit_likelihood_improvement(
    adata: ad.AnnData,
    pseudotime: pd.Series,
    genes: list[str],
    *,
    df: int = 5,
    degree: int = 3,
    dispersion_formula: str = "~ 1",
    device: str | torch.device | None = None,
    max_epochs: int = 1000,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    convergence_tol: float = 1e-3,
    convergence_window: int = 10,
    return_models: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, FittedMarginalModel]]:
    """Fit the null and alternative models, return the LRT statistic

    This calls scDesigner to fit negative binomial models under both the null
    and alternative models (without and with pseudotime effects, respectively).
    It returns a DataFrame with genes along rows. Each row includes the
    likelihoods from both models, whether the models converged, and the LRT
    statistic. If called with `return_models=True`, we return the actual models
    (but don't call this during permutation testing, since that saves lots of
    unnecessary models) as a class `FittedMarginalModel`, which has the same
    interface as scDesigner's simulators, but without the copula component.
    """
    # specify the models
    alt_mean_formula = mean_formula(df=df, degree=degree)
    sub = adata[pseudotime.index, list(genes)].copy()
    sub.obs["pseudotime"] = pseudotime.loc[sub.obs_names].to_numpy()

    fit_kwargs = dict(
        device=device, max_epochs=max_epochs, lr=lr, weight_decay=weight_decay
    )
    alt_formula = {"mean": alt_mean_formula, "dispersion": dispersion_formula}
    null_formula = {"mean": "~ 1", "dispersion": dispersion_formula}

    # fit the models
    alt_model = fit_marginal(sub, alt_formula, **fit_kwargs)
    null_model = fit_marginal(sub, null_formula, **fit_kwargs)

    # compute the test statistic
    loglik_alt = total_loglik(alt_model)
    loglik_null = total_loglik(null_model)
    statistic = 2.0 * (loglik_alt - loglik_null)

    # gather the results
    summary = pd.DataFrame(
        {
            "gene": list(sub.var_names),
            "n_cells": sub.n_obs,
            "loglik_alternative": loglik_alt,
            "loglik_null": loglik_null,
            "statistic": statistic,
            "converged_alternative": _converged(
                alt_model.fit_history, convergence_tol, convergence_window
            ),
            "converged_null": _converged(
                null_model.fit_history, convergence_tol, convergence_window
            ),
        }
    ).set_index("gene")
    if return_models:
        return summary, {
            "alternative": FittedMarginalModel(alt_model),
            "null": FittedMarginalModel(null_model),
        }
    return summary
