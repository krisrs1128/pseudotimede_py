"""Helpers to validate data inputs."""

import pandas as pd


def align_pseudotime(
    pseudotime: pd.Series, cell_order: pd.Index, *, require_complete: bool
) -> pd.Series:
    """Cell alignment helper

    This function was used when creating the lps dataset in our notebook
    starting from the R vignette's output. It makes sure that we align cells
    according to cell ID, not just the row index.
    """
    unknown = pseudotime.index.difference(cell_order)
    if len(unknown):
        raise ValueError(f"pseudotime references unknown cells: {list(unknown[:5])}")
    if not require_complete:
        return pseudotime
    return pseudotime.reindex(cell_order)
