"""Empirical p-values and per-gene calibration."""

import numpy as np
import pandas as pd
import pytest

from pseudotimede_py.calibration import (
    empirical_pvalue,
    summarize_calibration,
)


def _null_statistics(gene_to_stats: dict) -> pd.DataFrame:
    """Build a (replicate, gene)-indexed frame like run_null_fits' output."""
    records = []
    for gene, stats in gene_to_stats.items():
        for replicate, stat in enumerate(stats, start=1):
            records.append(
                {"replicate": replicate, "gene": gene, "statistic": stat}
            )
    return pd.DataFrame.from_records(records).set_index(["replicate", "gene"])


class TestEmpiricalPvalue:
    def test_hand_computed_no_ties(self):
        # observed=5, null=[1,2,3,4,10]: only 10 >= 5 -> (1+1)/(5+1)
        p = empirical_pvalue(5.0, np.array([1.0, 2.0, 3.0, 4.0, 10.0]))
        assert p == pytest.approx(2 / 6)

    def test_hand_computed_with_ties(self):
        # observed=5, null=[5,5,3,2,1]: two ties count as exceedances (>=)
        p = empirical_pvalue(5.0, np.array([5.0, 5.0, 3.0, 2.0, 1.0]))
        assert p == pytest.approx(3 / 6)

    def test_minimum_attainable_value_when_no_exceedances(self):
        # observed larger than every null draw -> p hits the 1/(B+1) floor
        p = empirical_pvalue(100.0, np.array([1.0, 2.0, 3.0]))
        assert p == pytest.approx(1 / 4)


class TestSummarizeCalibration:
    def test_matches_direct_computation_when_all_succeed(self):
        observed = pd.Series({"G0": 5.0, "G1": 10.0})
        null_statistics = _null_statistics(
            {"G0": [1.0, 2.0, 3.0, 4.0, 10.0], "G1": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )

        summary = summarize_calibration(observed, null_statistics)

        assert summary.loc["G0", "p_empirical"] == pytest.approx(
            empirical_pvalue(5.0, np.array([1.0, 2.0, 3.0, 4.0, 10.0]))
        )
        assert summary.loc["G1", "p_empirical"] == pytest.approx(1 / 6)

    def test_genes_parameter_restricts_output(self):
        observed = pd.Series({"G0": 5.0, "G1": 10.0})
        null_statistics = _null_statistics(
            {"G0": [1.0, 2.0], "G1": [1.0, 2.0]}
        )
        summary = summarize_calibration(observed, null_statistics, genes=["G1"])
        assert list(summary.index) == ["G1"]
