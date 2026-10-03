"""scDesigner interface for NB marginals."""
import copy
import numpy as np
import pandas as pd
import torch
from collections.abc import Sequence
from formulaic import model_matrix
from scdesigner.distributions import NegBin


class FittedMarginalModel:
    """Interface to an NB marginal model

    This adds a `.predict` method to an NB model, so that it has a similar
    interface to the full simulator objects in scDesigner. The issue is that
    ``NegBin`` as used throughout this notebook is a lower-level simulation
    object and doesn't take anndata as inputs. For convenience in downstream
    diagnostic assessment, though, we would like the scDesigner simulation
    interface. This class is a small wrapper to support that interface.
    """

    def __init__(self, marginal: NegBin):
        self.marginal = marginal
        self.template = marginal.adata
        self._columns = np.arange(self.template.n_vars)

        pt = self.template.obs["pseudotime"].to_numpy(dtype=float)
        self._pseudotime_range = (pt.min(), pt.max())
        self._model_specs = {}

        for name, formula in marginal.formula.items():
            training = model_matrix(formula, self.template.obs, na_action="raise")
            self._model_specs[name] = training.model_spec

    def predict(self, obs: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
        """Predicted (cells, selected genes) arrays

        Setting ``obs=None`` will default to ``template.obs``.
        """
        if obs is None:
            obs = self.template.obs
        coef = next(self.marginal.predict.parameters())
        if len(obs) == 0:
            return {
                name: torch.empty((0, len(self._columns)), dtype=coef.dtype).numpy()
                for name in self._model_specs
            }
        design = {
            name: torch.as_tensor(
                np.asarray(spec.get_model_matrix(obs, na_action="raise")).copy(),
                dtype=coef.dtype, device=coef.device,
            )
            for name, spec in self._model_specs.items()
        }
        with torch.no_grad():
            return {
                name: values.detach().cpu().numpy()[:, self._columns]
                for name, values in self.marginal.predict(design).items()
            }

    def select_genes(self, genes: Sequence[str]) -> FittedMarginalModel:
        """View a subset of genes

        In our testing, we may want to calibrate with a subset of genes. This is
        a convenience function to replace the full marginal model with a version
        that's focused on a subset of interest.
        """
        if isinstance(genes, str):
            genes = [genes]
        genes = pd.Index(genes)
        positions = self.template.var_names.get_indexer(genes)
        selected = copy.copy(self)
        selected.template = self.template[:, positions]
        selected._columns = self._columns[positions]
        return selected
