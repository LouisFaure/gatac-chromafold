"""
ChromaFold's scATAC inputs, built with GATAC.

ChromaFold {Gao et al. 2024, Nat Commun 15:9432} takes, per cell type:

* the 50 bp Tn5 insertion track of all its cells, and their library size;
* the 500 bp insertion counts of metacells, Cicero-style overlapping
  neighbourhoods of 100 cells on ArchR's iterative LSI, from which it
  correlates tiles (co-accessibility);
* a CTCF motif (or ChIP) track, which depends on the genome only.

Its preprocessing (preprocessing_pipeline/ in the ChromaFold repository) runs
ArchR for the cell filters and the LSI, then pandas for the counts.
:func:`prepare` builds the same inputs on the GPU, with GATAC's fragment files,
iterative LSI and this package's counts and metacells, and
:meth:`ChromaFoldInputs.write` writes them in ChromaFold's file layout, which
its ``inference.py`` reads.

Insertions are counted as GATAC and SnapATAC2 place them: a fragment's second
insertion is at ``end - 1``, where ChromaFold's pandas code puts it at ``end``. That moves
the second insertion of the fragments whose ``end`` is a multiple of the bin
size into the previous bin (0.1 % of insertions at 500 bp, 1 % at 50 bp);
everything else is identical (tests/reproducibility).
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import gatac as ga

from ._counts import InsertionTracks, insertion_tiles, insertion_tracks
from ._metacells import metacells as _metacells
from ._metacells import profiles as _profiles

logger = logging.getLogger(__name__)

#: ChromaFold's preprocessing parameters: cell filters of ``createArrowFiles``,
#: ``addIterativeLSI`` of ArchR_preparation.R, metacells and resolutions of
#: scATAC_preparation.py.
PARAMS = {
    "min_tsse": 4,
    "min_fragments": 1000,
    "lsi": {"n_comps": 30, "iterations": 2, "n_features": 25000,
            "cluster_params": {"resolution": 0.2}},
    "n_neighbors": 100,
    "max_overlap": 33,
    "random_state": 10,
    "tile_size": 500,
    "bin_size": 50,
}

#: Genomes whose chromosome sizes gatac.pp has (gatac.pp.HG38, ...)
GENOMES = ("hg38", "hg19", "mm10", "mm39")


def chromosomes(chrom_sizes: Mapping[str, int]) -> dict[str, int]:
    """ChromaFold's chromosomes, chr1-22 and chrX, those of ``chrom_sizes``."""
    names = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    return {c: int(chrom_sizes[c]) for c in names if c in chrom_sizes}


@dataclass
class ChromaFoldInputs:
    """
    ChromaFold's scATAC inputs of one cell type, returned by :func:`prepare`.

    Attributes
    ----------
    tiles : AnnData
        Cells × 500 bp tiles, insertion counts, with the embedding, the
        metacells in ``obsm["chromafold_metacells"]`` and their seed cells in
        ``uns["chromafold_metacells"]``.
    metacells : AnnData
        Metacells × 500 bp tiles, the sums of their cells' counts.
    pseudobulk : InsertionTracks
        The 50 bp insertion track of all the cells (group ``"all"``).
    chrom_sizes : dict
        The chromosomes used, chr1-22 and chrX.
    """

    tiles: ad.AnnData
    metacells: ad.AnnData
    pseudobulk: InsertionTracks
    chrom_sizes: dict[str, int]

    def __repr__(self) -> str:
        return (f"ChromaFoldInputs({self.tiles.n_obs:,} cells, {self.metacells.n_obs:,} metacells, "
                f"{len(self.chrom_sizes)} chromosomes)")

    @property
    def library_size(self) -> int:
        """Insertions of all the cells, which ChromaFold normalises the track by."""
        return int(self.pseudobulk.library_size["all"])

    def write(self, data_path: Union[str, Path], cell_type: str) -> dict[str, Path]:
        """
        Write the inputs as ChromaFold's preprocessing does, under
        ``data_path/atac/``, for ``inference.py --data-path data_path -ct cell_type``:

        * ``{cell_type}_tile_pbulk_50bp_dict.p``: per chromosome, the 50 bp
          track as a ``(length // 50 + 1, 1)`` float64 array;
        * ``{cell_type}_tile_500bp_dict.p``: per chromosome, the cells ×
          ``(length // 500 + 1)`` float64 CSR insertion counts;
        * ``{cell_type}_tile_500bp_barcode.npy``: the cells, in the rows' order;
        * ``{cell_type}_metacell_mask.csv``: metacells × cells, 0/1, with the
          cells as columns in that same order.

        ChromaFold's training also reads per-cell 50 bp tiles, which are not
        written: inference uses the track of all cells.

        Returns
        -------
        dict
            The paths written.
        """
        out = Path(data_path) / "atac"
        os.makedirs(out, exist_ok=True)
        cells = np.asarray(self.tiles.obs_names, dtype=str)
        chrom_of_tile = self.tiles.var["chrom"].to_numpy()
        tiles = {}
        for c, length in self.chrom_sizes.items():
            i = np.flatnonzero(chrom_of_tile == c)
            m = self.tiles.X[:, i[0]:i[-1] + 1].astype(np.float64)
            tiles[c] = _pad_columns(sp.csr_matrix(m), length // PARAMS["tile_size"] + 1)
        bulk = {}
        for c, length in self.chrom_sizes.items():
            v = np.zeros(length // PARAMS["bin_size"] + 1)
            t = self.pseudobulk.track("all", c)
            v[:len(t)] = t
            bulk[c] = v[:, None]
        mask = pd.DataFrame(self.tiles.obsm["chromafold_metacells"].T.toarray().astype(np.int64),
                            columns=cells)
        paths = {
            "pseudobulk": out / f"{cell_type}_tile_pbulk_50bp_dict.p",
            "tiles": out / f"{cell_type}_tile_500bp_dict.p",
            "barcodes": out / f"{cell_type}_tile_500bp_barcode.npy",
            "metacell_mask": out / f"{cell_type}_metacell_mask.csv",
        }
        with open(paths["pseudobulk"], "wb") as f:
            pickle.dump(bulk, f)
        with open(paths["tiles"], "wb") as f:
            pickle.dump(tiles, f)
        np.save(paths["barcodes"], cells)
        mask.to_csv(paths["metacell_mask"])
        logger.info(f"Wrote ChromaFold inputs of {cell_type!r} to {out}")
        return paths


def _pad_columns(m: sp.csr_matrix, n: int) -> sp.csr_matrix:
    return sp.csr_matrix((m.data, m.indices, m.indptr), shape=(m.shape[0], n))


def prepare(
    fragments: Union[str, Path],
    genome: Union[str, Mapping[str, int]] = "hg38",
    cells: Optional[Union[Sequence[str], ad.AnnData]] = None,
    use_rep: Optional[str] = None,
    metrics=None,
    params: Optional[dict] = None,
) -> ChromaFoldInputs:
    """
    Build ChromaFold's scATAC inputs of one cell type from its fragments.

    Parameters
    ----------
    fragments
        GATAC fragment parquet file of the cell type (:func:`gatac.io.make_parquet`).
    genome
        ``"hg38"``, ``"hg19"``, ``"mm10"``, ``"mm39"``, or chromosome sizes.
        Only chr1-22 and chrX are used, as ChromaFold does.
    cells
        The cells to use, as names or an AnnData (its ``obs_names``), in
        ChromaFold's order whatever the order given: sorted, as its
        ``np.unique`` of the barcodes sorts them, which the metacells depend
        on. Cells without a fragment on chr1-22 and chrX are left out. By
        default, the cells passing ChromaFold's filters, TSS enrichment of at
        least ``params["min_tsse"]`` and at least ``params["min_fragments"]``
        unique fragments, read from ``metrics``
        (:func:`gatac.pp.compute_metrics`), which is then required.
    use_rep
        With ``cells`` an AnnData, the embedding of its ``obsm`` to build the
        metacells on. By default an iterative LSI with ChromaFold's parameters
        (``params["lsi"]``) is computed, as ChromaFold computes it with ArchR.
        The published models were trained on that LSI.
    metrics
        Cell metrics with ``barcode``, ``n_unique`` and ``tsse_score``, for
        the default cell filters.
    params
        Overrides of :data:`PARAMS`.

    Returns
    -------
    ChromaFoldInputs

    Examples
    --------
    >>> metrics = ga.pp.compute_metrics("tcell.parquet", "gencode.gtf")
    >>> inputs = ga.chromafold.prepare("tcell.parquet", "hg38", metrics=metrics)
    >>> inputs.write("chromafold_data", "tcell")
    """
    p = {**PARAMS, **(params or {})}
    if isinstance(genome, str):
        if genome not in GENOMES:
            raise ValueError(f"unknown genome {genome!r}; one of {list(GENOMES)}, or chromosome sizes")
        genome = getattr(ga.pp, genome.upper())
    sizes = chromosomes(genome)
    if not sizes:
        raise ValueError("none of chr1-22 and chrX are in the genome")

    embedding = None
    if isinstance(cells, ad.AnnData):
        if use_rep is not None:
            embedding = pd.DataFrame(np.asarray(cells.obsm[use_rep]), index=cells.obs_names)
        cells = list(cells.obs_names)
    elif use_rep is not None:
        raise ValueError("use_rep needs cells as an AnnData")
    if cells is None:
        if metrics is None:
            raise ValueError("give the cells, or the metrics to filter them on")
        m = metrics.to_pandas() if hasattr(metrics, "to_pandas") else pd.DataFrame(metrics)
        m = m[(m["tsse_score"] >= p["min_tsse"]) & (m["n_unique"] >= p["min_fragments"])]
        cells = list(m["barcode"].astype(str))
    if len(cells) == 0:
        raise ValueError("no cells")

    tiles = insertion_tiles(fragments, sizes, cells, tile_size=p["tile_size"])
    if embedding is not None:
        missing = tiles.obs_names.difference(embedding.index)
        if len(missing):
            raise ValueError(f"{len(missing)} cells have no {use_rep!r}, e.g. {missing[0]!r}")
        tiles.obsm[use_rep] = embedding.loc[tiles.obs_names].to_numpy()
    else:
        use_rep = "X_iterative_lsi"
        ga.dr.iterative_lsi(tiles, **p["lsi"])
    _metacells(tiles, use_rep=use_rep, n_neighbors=p["n_neighbors"], max_overlap=p["max_overlap"],
               random_state=p["random_state"], key_added="chromafold_metacells")
    metacells = _profiles(tiles, "chromafold_metacells")
    pseudobulk = insertion_tracks(fragments, sizes, bin_size=p["bin_size"],
                                  groups=pd.Series("all", index=tiles.obs_names))
    tiles.uns["chromafold"] = {"params": p, "use_rep": use_rep}
    return ChromaFoldInputs(tiles=tiles, metacells=metacells, pseudobulk=pseudobulk, chrom_sizes=sizes)
