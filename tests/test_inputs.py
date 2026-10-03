"""Input data structures and the LPS dataset loading workflow."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scdesigner import datasets

from pseudotimede_py import subsample_pseudotimes
from pseudotimede_py.validation import align_pseudotime


def test_pseudotime_is_aligned_by_cell_id():
    pt = pd.Series([0.9, 0.1, 0.5], index=["C3", "C1", "C2"])
    cell_order = pd.Index(["C1", "C2", "C3"])

    aligned = align_pseudotime(pt, cell_order, require_complete=True)

    assert list(aligned.index) == list(cell_order)
    assert aligned.tolist() == [0.1, 0.5, 0.9]


def test_subsample_pseudotime_can_cover_fewer_cells():
    pt = pd.Series([0.5, 0.1], index=["C3", "C1"])
    cell_order = pd.Index(["C1", "C2", "C3"])

    aligned = align_pseudotime(pt, cell_order, require_complete=False)

    pd.testing.assert_series_equal(aligned, pt)


def test_subsample_table_is_unpacked_in_replicate_order():
    adata = ad.AnnData(
        X=np.zeros((4, 1)), obs=pd.DataFrame(index=["C1", "C2", "C3", "C4"])
    )
    adata.obsm["pseudotime_subsamples"] = pd.DataFrame(
        {
            "replicate_1": [0.1, 0.4, 0.2, np.nan],
            "replicate_2": [0.1, 0.4, np.nan, 0.9],
        },
        index=adata.obs_names,
    )

    subs = subsample_pseudotimes(adata)

    assert len(subs) == 2
    pd.testing.assert_series_equal(
        subs[0], pd.Series([0.1, 0.4, 0.2], index=["C1", "C2", "C3"], name="replicate_1")
    )
    pd.testing.assert_series_equal(
        subs[1], pd.Series([0.1, 0.4, 0.9], index=["C1", "C2", "C4"], name="replicate_2")
    )


def test_lps_dataset_loads_with_usable_subsamples():
    try:
        lps = datasets.lps()
    except RuntimeError as e:  # Figshare download failed (e.g. offline)
        pytest.skip(f"scdesigner LPS dataset unavailable: {e}")

    assert lps.shape == (390, 100)
    subs = subsample_pseudotimes(lps)
    assert len(subs) == 1000
    for pt in subs:
        assert len(pt) == 312
        assert pt.index.isin(lps.obs_names).all()
