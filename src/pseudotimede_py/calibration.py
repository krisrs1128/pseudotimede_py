"""Permutation tests

We compute our p-values using,

p = (1 + #{T_null >= T_observed}) / (B + 1)

We don't support a gamma mixture tail approximation like in the original
ClusterDE package. The minimum p-value is bounded below by 1/(B + 1).
"""
import numpy as np
import pandas as pd


def empirical_pvalue(observed: float, null_statistics: np.ndarray) -> float:
    """`(1 + #{T_null >= T_observed}) / (B + 1)` for one gene

    `null_statistics` is an array of test statisitcs under permutation.
    """
    values = np.asarray(null_statistics, dtype=float)
    exceedances = int(np.sum(values >= observed))
    return (1 + exceedances) / (values.size + 1)


def summarize_calibration(
    observed: pd.Series, null_statistics: pd.DataFrame, genes: list[str] | None = None
) -> pd.DataFrame:
    """Permutation p-values from an observed fit and a matrix of null statistics

    `observed` should have length equal to the number of genes. See the
    `statistic` column in `fit_likelihood_improvement`. `null_statistics` should
    be indexed by both permutation replicate and gene. It corresopnds to the
    `statistic` column in `run_null_fits`.
    """
    if genes is None:
        genes = list(observed.index)

    rows = []
    for gene in genes:
        statistic = float(observed.loc[gene])
        gene_null = null_statistics.xs(gene, level="gene")["statistic"].to_numpy()
        rows.append(
            {
                "gene": gene,
                "statistic_observed": statistic,
                "p_empirical": empirical_pvalue(statistic, gene_null),
            }
        )

    return pd.DataFrame.from_records(rows).set_index("gene")
