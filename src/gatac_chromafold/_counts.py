"""
Tn5 insertion counts, as ChromaFold takes them: per cell in 500 bp tiles,
and summed over cells in 50 bp bins.

Each end of each unique fragment is one insertion, at ``start`` and at
``end - 1`` (GATAC's and SnapATAC2's convention; ChromaFold's pandas code puts
the second one at ``end``), so a fragment with both ends in a tile counts twice
there: SnapATAC2's ``counting_strategy="insertion"``, not GATAC's default
paired-insertion counting. Insertions past a chromosome's end are not counted.
Both stream the fragment file on the GPU with :meth:`gatac.io.FragmentFile.iter_batches`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import cudf
import cupy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from gatac.io import FragmentFile

logger = logging.getLogger(__name__)


def _layout(chrom_sizes: Mapping[str, int], bin_size: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Chromosomes, their numbers of bins (the last one partial) and their first bin's global index."""
    chroms = list(chrom_sizes)
    n_bins = np.array([-(-int(chrom_sizes[c]) // bin_size) for c in chroms], np.int64)
    return chroms, n_bins, np.concatenate([[0], np.cumsum(n_bins)])


def insertion_tiles(
    fragments: str | Path,
    chrom_sizes: Mapping[str, int],
    cells: Sequence[str],
    tile_size: int = 500,
    row_groups_per_batch: int = 64,
) -> ad.AnnData:
    """
    Cells × tiles insertion counts on the chromosomes of ``chrom_sizes``.

    Cells are in sorted order, as ChromaFold's ``np.unique`` of the barcodes
    orders them, and those without a fragment on these chromosomes are left out.
    ``obs["n_unique"]`` is each cell's number of fragments there (the depth
    :func:`gatac.dr.iterative_lsi` correlates components with); ``var`` has
    ``chrom``, ``start`` and ``end``, named ``chrom:start-end``, every tile of
    each chromosome from its start.
    """
    t0 = time.perf_counter()
    ff = FragmentFile(fragments)
    chroms, n_tiles, offsets = _layout(chrom_sizes, tile_size)
    total = int(offsets[-1])

    counts = ff.barcode_counts(chroms=chroms).set_index("barcode")["n_unique"]
    wanted = pd.Index(np.unique(np.asarray(cells, dtype=str)))
    n_unique = counts.reindex(wanted).fillna(0).astype(np.int64)
    cells = list(wanted[n_unique.to_numpy() > 0])
    if not cells:
        raise ValueError("none of the cells has a fragment on these chromosomes")

    offsets_d, n_tiles_d = cp.asarray(offsets[:-1]), cp.asarray(n_tiles)
    parts = []
    for b in ff.iter_batches(["chrom", "start", "end"], row_groups_per_batch, chroms=chroms, cells=cells):
        chrom = b["chrom"].values
        row = b["cell"].values.astype(cp.int64) * total
        keys = []
        for pos in (b["start"].values, b["end"].values - 1):
            i = pos.astype(cp.int64) // tile_size
            ok = (i >= 0) & (i < n_tiles_d[chrom])
            keys.append(row[ok] + offsets_d[chrom][ok] + i[ok])
        key, n = cp.unique(cp.concatenate(keys), return_counts=True)
        parts.append(cudf.DataFrame({"key": key, "n": n.astype(cp.int32)}))
        del b, chrom, row, keys
    agg = cudf.concat(parts).groupby("key", sort=True)["n"].sum()
    del parts
    key, data = agg.index.values, agg.values.astype(cp.float32)
    rows = (key // total).astype(cp.int64)
    indptr = cp.zeros(len(cells) + 1, cp.int64)
    indptr[1:] = cp.cumsum(cp.bincount(rows, minlength=len(cells)))
    X = sp.csr_matrix(
        (data.get(), (key % total).astype(cp.int32).get(), indptr.get()), shape=(len(cells), total)
    )
    del agg, key, data, rows, indptr
    cp.get_default_memory_pool().free_all_blocks()

    starts = np.concatenate([np.arange(n, dtype=np.int64) * tile_size for n in n_tiles])
    chrom_of = np.repeat(np.asarray(chroms, dtype=object), n_tiles)
    sizes = np.repeat([int(chrom_sizes[c]) for c in chroms], n_tiles)
    var = pd.DataFrame({"chrom": chrom_of, "start": starts, "end": np.minimum(starts + tile_size, sizes)})
    var.index = var["chrom"].astype(str) + ":" + var["start"].astype(str) + "-" + var["end"].astype(str)
    obs = pd.DataFrame({"n_unique": n_unique.loc[cells].to_numpy()}, index=pd.Index(cells))
    logger.info(
        f"Counted insertions of {len(cells):,} cells in {total:,} tiles of {tile_size} bp "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    return ad.AnnData(X=X, obs=obs, var=var)


@dataclass
class InsertionTracks:
    """
    Tn5 insertions per group of cells in fixed bins, returned by :func:`insertion_tracks`.

    Attributes
    ----------
    groups : list of str
        Group names, in the order of the rows of each array.
    bin_size : int
        Bin width in base pairs: bin ``i`` covers ``[i * bin_size, (i + 1) * bin_size)``.
    chrom_sizes : dict
        Lengths of the chromosomes counted.
    counts : dict
        Per chromosome, an ``int32`` array ``(n_groups, ceil(length / bin_size))``.
    """

    groups: list[str]
    bin_size: int
    chrom_sizes: dict[str, int]
    counts: dict[str, np.ndarray]

    def __repr__(self) -> str:
        n_bins = sum(a.shape[1] for a in self.counts.values())
        return (
            f"InsertionTracks({len(self.groups)} groups × {n_bins:,} bins of "
            f"{self.bin_size} bp on {len(self.counts)} chromosomes)"
        )

    @property
    def library_size(self) -> pd.Series:
        """Insertions counted per group, over all chromosomes."""
        total = sum(a.sum(axis=1, dtype=np.int64) for a in self.counts.values())
        return pd.Series(total, index=self.groups, name="library_size")

    def track(self, group: str, chrom: str) -> np.ndarray:
        """Counts of one group along one chromosome."""
        return self.counts[chrom][self.groups.index(group)]


def insertion_tracks(
    fragments: str | Path,
    chrom_sizes: Mapping[str, int],
    bin_size: int = 50,
    groups: pd.Series | None = None,
    row_groups_per_batch: int = 64,
    max_gpu_bytes: int = 8 << 30,
) -> InsertionTracks:
    """
    Insertions per group of cells in ``bin_size`` bins on the chromosomes of
    ``chrom_sizes``: ``groups`` maps cells to groups (cells it leaves out, or
    maps to a missing value, are not counted); by default one group, ``"all"``,
    of every fragment. Groups that do not fit in ``max_gpu_bytes`` at once are
    counted over several passes of the file.
    """
    chroms, n_bins, offsets = _layout(chrom_sizes, bin_size)
    total = int(offsets[-1])
    if groups is None:
        names, cells, cell_group = ["all"], None, None
    else:
        groups = groups[groups.notna()]
        cat = pd.Categorical(groups.astype(str))
        names = [str(c) for c in cat.categories]
        cells = list(groups.index.astype(str))
        cell_group = np.asarray(cat.codes, np.int64)
    n_groups = len(names)
    per_pass = max(1, int(max_gpu_bytes // (4 * total)))

    t0 = time.perf_counter()
    ff = FragmentFile(fragments)
    out = {c: np.zeros((n_groups, n), np.int32) for c, n in zip(chroms, n_bins)}
    offsets_d, n_bins_d = cp.asarray(offsets[:-1]), cp.asarray(n_bins)
    for g0 in range(0, n_groups, per_pass):
        g1 = min(g0 + per_pass, n_groups)
        acc = cp.zeros((g1 - g0) * total, cp.int32)
        if cells is None:
            pass_cells, group_d = None, None
        else:
            keep = (cell_group >= g0) & (cell_group < g1)
            pass_cells = [c for c, k in zip(cells, keep) if k]
            group_d = cp.asarray(cell_group[keep] - g0)
        for b in ff.iter_batches(["chrom", "start", "end"], row_groups_per_batch, chroms=chroms, cells=pass_cells):
            chrom = b["chrom"].values
            base, last = offsets_d[chrom], n_bins_d[chrom]
            if group_d is not None:
                base = base + group_d[b["cell"].values] * total
            for pos in (b["start"].values, b["end"].values - 1):
                i = pos.astype(cp.int64) // bin_size
                ok = (i >= 0) & (i < last)
                cp.add.at(acc, base[ok] + i[ok], 1)
            del b, chrom, base, last
        acc = acc.reshape(g1 - g0, total).get()
        for i, c in enumerate(chroms):
            out[c][g0:g1] = acc[:, offsets[i] : offsets[i + 1]]
        del acc
    cp.get_default_memory_pool().free_all_blocks()
    logger.info(
        f"Counted insertions of {n_groups} groups in {total:,} bins of {bin_size} bp "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    return InsertionTracks(
        groups=names, bin_size=int(bin_size), chrom_sizes={c: int(chrom_sizes[c]) for c in chroms}, counts=out
    )
