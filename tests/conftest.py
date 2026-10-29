"""
A small synthetic cell type: two groups of cells, each enriched in its own
hotspots, on a genome of chr1, chr2, chrX and chrM (which ChromaFold leaves
out), written as a GATAC fragment file. Builds in seconds.
"""
from __future__ import annotations

import gzip

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import gatac as ga

CHROM_SIZES = {"chr1": 2_000_000, "chr2": 1_500_000, "chrX": 1_000_000, "chrM": 16_500}
N_GROUPS, CELLS_PER_GROUP = 2, 60


def _fragments(rng) -> pd.DataFrame:
    chroms = [c for c in CHROM_SIZES if c != "chrM"]
    hot = {g: [(c, int(s)) for c in chroms
               for s in rng.choice(np.arange(20_000, CHROM_SIZES[c] - 20_000, 5_000), 30, replace=False)]
           for g in range(N_GROUPS)}
    rows = []
    for g in range(N_GROUPS):
        for i in range(CELLS_PER_GROUP):
            n = int(rng.integers(1_500, 3_000))
            in_hot = rng.random(n) < 0.5
            pick = [hot[g][j] for j in rng.integers(len(hot[g]), size=n)]
            chrom = np.where(in_hot, [c for c, _ in pick], rng.choice(list(CHROM_SIZES), n, p=[0.4, 0.3, 0.2, 0.1]))
            size = np.array([CHROM_SIZES[c] for c in chrom])
            length = rng.integers(40, 400, n)
            start = np.where(in_hot, np.array([s for _, s in pick]) + rng.integers(-150, 150, n),
                             (rng.random(n) * (size - 1_000)).astype(np.int64))
            rows.append(pd.DataFrame({"chrom": chrom, "start": start, "end": start + length,
                                      "barcode": f"g{g}c{i:03d}-1", "count": 1}))
    df = pd.concat(rows)
    return df.sort_values(["chrom", "start", "end", "barcode"]).drop_duplicates(
        ["chrom", "start", "end", "barcode"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def celltype(tmp_path_factory):
    """``parquet``, the ``fragments`` table, and ``cells``: an AnnData of the cells with a
    grouped embedding in ``obsm["X_emb"]``."""
    d = tmp_path_factory.mktemp("celltype")
    rng = np.random.default_rng(0)
    df = _fragments(rng)
    tsv = d / "frags.tsv.gz"
    with gzip.open(tsv, "wt") as fh:
        df.to_csv(fh, sep="\t", header=False, index=False)
    parquet = ga.io.make_parquet(tsv, d / "frags.parquet")
    names = sorted(df.barcode.unique())
    group = np.array([int(n[1]) for n in names])
    cells = ad.AnnData(obs=pd.DataFrame(index=names))
    cells.obsm["X_emb"] = rng.normal(size=(len(names), 6)) + 3 * group[:, None]
    return {"dir": d, "parquet": parquet, "fragments": df, "cells": cells}
