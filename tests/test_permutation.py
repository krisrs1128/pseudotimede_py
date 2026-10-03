"""Pseudotime shuffling and reproducible permutation-null fitting."""

import anndata as ad
import numpy as np
import pandas as pd

from pseudotimede_py.calibration import summarize_calibration
from pseudotimede_py.permutation import run_null_fits, shuffle_pseudotime


def _toy_adata(n_cells=20, n_genes=2, seed=0):
    rng = np.random.default_rng(seed)
    pseudotime = np.linspace(0.0, 1.0, n_cells)
    counts = rng.poisson(lam=5.0, size=(n_cells, n_genes)).astype(float)
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {"pseudotime": pseudotime}, index=[f"C{i}" for i in range(n_cells)]
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]),
    )
    return adata


class TestShufflePseudotime:
    def test_shuffles_pairings_preserving_cells_and_values(self):
        pseudotime = pd.Series(
            [0.1, 0.4, 0.2, 0.9, 0.5], index=["A", "B", "C", "D", "E"]
        )
        rng = np.random.default_rng(0)
        shuffled = shuffle_pseudotime(pseudotime, rng)

        assert list(shuffled.index) == list(pseudotime.index)
        assert not np.array_equal(shuffled.to_numpy(), pseudotime.to_numpy())
        np.testing.assert_array_equal(
            np.sort(shuffled.to_numpy()), np.sort(pseudotime.to_numpy())
        )


class TestRunNullFits:
    def test_shape_and_replicate_numbering(self):
        adata = _toy_adata(n_cells=20, n_genes=2)
        pseudotime = adata.obs["pseudotime"]
        replicates = [pseudotime, pseudotime, pseudotime]
        genes = list(adata.var_names)

        result = run_null_fits(adata, replicates, genes, seed=0, max_epochs=20)

        assert len(result) == len(replicates) * len(genes)
        assert list(result.index.get_level_values("replicate").unique()) == [1, 2, 3]
        assert set(result.index.get_level_values("gene")) == set(genes)
        assert np.isfinite(result["statistic"]).all()
        assert list(result.columns) == [
            "n_cells", "loglik_alternative", "loglik_null", "statistic",
            "converged_alternative", "converged_null",
        ]
        summary = summarize_calibration(pd.Series(0.0, index=genes), result)
        assert summary["p_empirical"].between(0, 1).all()

    def test_reproducible_with_same_seed(self):
        adata = _toy_adata(n_cells=25, n_genes=2)
        pseudotime = adata.obs["pseudotime"]
        replicates = [pseudotime.iloc[:20], pseudotime.iloc[5:]]
        genes = list(adata.var_names)

        r1 = run_null_fits(adata, replicates, genes, seed=42, max_epochs=20)
        r2 = run_null_fits(adata, replicates, genes, seed=42, max_epochs=20)

        pd.testing.assert_frame_equal(r1, r2)

    def test_no_input_mutation(self):
        adata = _toy_adata(n_cells=20, n_genes=2)
        pseudotime = adata.obs["pseudotime"].copy()
        replicate = pseudotime.copy()
        original_obs_columns = list(adata.obs.columns)

        run_null_fits(adata, [replicate], list(adata.var_names), seed=0, max_epochs=10)

        assert list(adata.obs.columns) == original_obs_columns
        pd.testing.assert_series_equal(replicate, pseudotime)
