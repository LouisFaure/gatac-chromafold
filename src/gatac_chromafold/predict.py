"""
ChromaFold's contact-map prediction, on the GPU.

For each 10 kb step along a chromosome, ChromaFold predicts the contacts of
the centre bin of a 4.01 Mb window with the 200 bins (2 Mb) on either side, a
"V-stripe", from three inputs over the window:

* the 50 bp track of all cells, divided by ``library_size * 150 /
  effective_genome_size`` and log-transformed;
* the CTCF motif track at 50 bp;
* co-accessibility: the Jaccard index between the binarised metacell profiles
  of the 20 centre 500 bp tiles and of the other 8,000 tiles of the window.

:func:`predict` computes the windows as ChromaFold's ``test_Dataset`` and
``test`` (chromafold/dataloader.py, testing.py, utils.py) do, batched on the
GPU, and runs the published model (_model.py). Stripes overlap: each pair of
bins is predicted from both of its bins, and :meth:`ContactPrediction.matrix`
averages the two, as ChromaFold's evaluation notebook does
(``get_combined_yhat(avg_stripe=True)``). The values are on the scale of the
HiC-DC+ z-scores the model was trained on.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import _ArrayUnpickler, checkpoint, ctcf_track

logger = logging.getLogger(__name__)

INPUT_SIZE = 4_010_000
RESOLUTION = 10_000
PBULK_RES, SC_RES = 50, 500
PRED_LEN = 200
EFFECTIVE_GENOME_SIZE = {"hg": 2_913_022_398, "mm": 2_652_783_500}

#: ChromaFold's last window ends (chromafold/util_chrom_start.py): windows start
#: from ``offset`` up to ``end - 5 Mb``. Not the chromosome lengths: some hg38
#: entries are hg19's.
ENDS = {
    "hg38": {"chr1": 248_950_000, "chr2": 242_190_000, "chr3": 198_020_000, "chr4": 190_210_000,
             "chr5": 180_910_000, "chr6": 170_800_000, "chr7": 159_130_000, "chr8": 145_130_000,
             "chr9": 138_390_000, "chr10": 133_790_000, "chr11": 135_000_000, "chr12": 133_270_000,
             "chr13": 114_360_000, "chr14": 107_040_000, "chr15": 101_990_000, "chr16": 90_330_000,
             "chr17": 81_190_000, "chr18": 78_070_000, "chr19": 58_610_000, "chr20": 63_020_000,
             "chr21": 46_700_000, "chr22": 51_300_000, "chrX": 155_270_000},
    "hg19": {"chr1": 249_250_000, "chr2": 243_190_000, "chr3": 198_020_000, "chr4": 191_150_000,
             "chr5": 180_910_000, "chr6": 171_110_000, "chr7": 159_130_000, "chr8": 146_360_000,
             "chr9": 141_210_000, "chr10": 135_530_000, "chr11": 135_000_000, "chr12": 133_850_000,
             "chr13": 115_160_000, "chr14": 107_340_000, "chr15": 102_530_000, "chr16": 90_350_000,
             "chr17": 81_190_000, "chr18": 78_070_000, "chr19": 59_120_000, "chr20": 63_020_000,
             "chr21": 48_120_000, "chr22": 51_300_000, "chrX": 155_270_000},
    "mm10": {"chr1": 195_470_000, "chr2": 182_110_000, "chr3": 160_030_000, "chr4": 156_500_000,
             "chr5": 151_830_000, "chr6": 149_730_000, "chr7": 145_440_000, "chr8": 129_400_000,
             "chr9": 124_590_000, "chr10": 130_690_000, "chr11": 122_080_000, "chr12": 120_120_000,
             "chr13": 120_420_000, "chr14": 124_900_000, "chr15": 104_040_000, "chr16": 98_200_000,
             "chr17": 94_980_000, "chr18": 90_700_000, "chr19": 61_430_000, "chrX": 171_030_000},
}


@dataclass
class ContactPrediction:
    """
    ChromaFold's predicted contacts, returned by :func:`predict`.

    Attributes
    ----------
    stripes : dict
        Per chromosome, a float32 array ``(n_windows, 400)``: for the window
        centred on 10 kb bin ``c``, its contacts with bins ``c - 200 ... c - 1``
        then ``c + 1 ... c + 200``.
    centers : dict
        Per chromosome, the centre bin of each window.
    chrom_sizes : dict
        Chromosome lengths.
    resolution : int
        Bin size, 10 kb.
    """

    stripes: dict[str, np.ndarray]
    centers: dict[str, np.ndarray]
    chrom_sizes: dict[str, int]
    resolution: int = RESOLUTION

    def __repr__(self) -> str:
        n = sum(len(v) for v in self.centers.values())
        return f"ContactPrediction({len(self.stripes)} chromosomes, {n:,} windows at {self.resolution // 1000} kb)"

    def matrix(self, chrom: str) -> sp.csr_matrix:
        """
        Upper triangle of the predicted contact map of ``chrom``, 10 kb bins
        from the chromosome start, up to 200 bins off the diagonal; the two
        predictions of each pair averaged.
        """
        y, c = self.stripes[chrom], self.centers[chrom]
        n = -(-self.chrom_sizes[chrom] // self.resolution)
        j = np.arange(PRED_LEN)
        rows = np.concatenate([c[:, None] - PRED_LEN + j, np.repeat(c[:, None], PRED_LEN, 1)], axis=1)
        cols = np.concatenate([np.repeat(c[:, None], PRED_LEN, 1), c[:, None] + 1 + j], axis=1)
        ok = (rows >= 0) & (cols < n)
        rows, cols, vals = rows[ok], cols[ok], y[ok].astype(np.float64)
        total = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        count = sp.coo_matrix((np.ones_like(vals), (rows, cols)), shape=(n, n)).tocsr()
        total.data /= count.data
        return total

    def to_dataframe(self, chrom: str) -> pd.DataFrame:
        """:meth:`matrix` as a table of ``start1``, ``start2`` and ``score``."""
        m = self.matrix(chrom).tocoo()
        return pd.DataFrame({"start1": m.row.astype(np.int64) * self.resolution,
                             "start2": m.col.astype(np.int64) * self.resolution, "score": m.data})


def _read_layout(data_path: Union[str, Path], cell_type: str):
    """The 50 bp tracks and tiles × metacells profiles of a ChromaFold data directory,
    as its load_atac_eval reads them."""
    atac = Path(data_path) / "atac"

    def load(name):
        with open(atac / name, "rb") as f:
            return _ArrayUnpickler(f).load()

    bulk = {c: np.asarray(v).ravel().astype(np.float64) for c, v in load(f"{cell_type}_tile_pbulk_50bp_dict.p").items()}
    tiles = load(f"{cell_type}_tile_500bp_dict.p")
    mask = sp.csr_matrix(pd.read_csv(atac / f"{cell_type}_metacell_mask.csv", index_col=0).values)
    profiles = {c: sp.csr_matrix((mask * tiles[c]).T) for c in tiles}
    return bulk, profiles


def _from_inputs(inputs):
    """The same, from :class:`~gatac_chromafold.ChromaFoldInputs`."""
    bulk, profiles = {}, {}
    chrom_of_tile = inputs.metacells.var["chrom"].to_numpy()
    for c, length in inputs.chrom_sizes.items():
        v = np.zeros(length // PBULK_RES + 1)
        t = inputs.pseudobulk.track("all", c)
        v[:len(t)] = t
        bulk[c] = v
        i = np.flatnonzero(chrom_of_tile == c)
        profiles[c] = sp.csr_matrix(inputs.metacells.X[:, i[0]:i[-1] + 1].T)
    return bulk, profiles


def _load_model(path, device):
    import torch

    from ._model import branch_cov, branch_pbulk, trunk

    model = trunk(branch_pbulk(), branch_cov())
    state = torch.load(path, map_location="cpu", weights_only=True)
    # saved from nn.DataParallel wrappers around the trunk and both branches
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    return model.to(device).eval()


def _windows(x, starts, res, size, n_rows=None):
    """``x[start // res : start // res + size]`` for each start, zero-padded on the left
    where the window starts before the chromosome (ChromaFold's start < 0)."""
    import torch

    idx = torch.as_tensor(np.asarray(starts) // res, device=x.device)[:, None] + torch.arange(size, device=x.device)
    ok = (idx >= 0) & (idx < x.shape[0])
    out = x[idx.clamp(0, x.shape[0] - 1)]
    return out * ok.reshape(ok.shape + (1,) * (out.dim() - 2)).to(out.dtype)


def _jaccard(z):
    """ChromaFold's cpu_batch_corcoeff_vstripe on a batch: ``z`` (B, 8020 tiles, metacells),
    binarised; Jaccard of the 20 centre tiles against the other 8,000 -> (B, 20, 8000)."""
    import torch

    z = (z > 0).float()
    n = z.shape[2]
    centre = z[:, 4000:4020]
    others = torch.cat([z[:, :4000], z[:, 4020:]], dim=1)
    num = centre @ others.transpose(1, 2)
    both_zero = (1 - centre) @ (1 - others).transpose(1, 2)
    denom = torch.clamp(n - both_zero, min=1e-8)
    out = num / denom
    out[out != out] = 0
    return out


def predict(
    inputs=None,
    genome: str = "hg38",
    chroms: Optional[Sequence[str]] = None,
    model: Union[str, os.PathLike] = "CTCFmotif",
    ctcf: Optional[Mapping[str, np.ndarray]] = None,
    offset: int = -2_000_000,
    batch_size: int = 64,
    device: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    cell_type: Optional[str] = None,
) -> ContactPrediction:
    """
    Predict contact maps with ChromaFold.

    Parameters
    ----------
    inputs
        :class:`~gatac_chromafold.ChromaFoldInputs` from :func:`prepare`. Or
        give ``data_path`` and ``cell_type`` instead, to read a ChromaFold data
        directory, written by :meth:`ChromaFoldInputs.write` or by ChromaFold's
        own preprocessing.
    genome
        ``"hg38"``, ``"hg19"`` or ``"mm10"``: the CTCF motif track, the
        effective genome size and ChromaFold's window ends.
    chroms
        Chromosomes to predict (default: all of the inputs').
    model
        A checkpoint of :data:`~gatac_chromafold.data.CHECKPOINTS`
        (``"CTCFmotif"``, the published model, or ``"CTCFChIP"``, which takes a
        CTCF ChIP-seq track as ``ctcf``), or the path of a checkpoint.
    ctcf
        The CTCF track per chromosome in 50 bp bins (default: ChromaFold's
        motif track of ``genome``, downloaded on first use).
    offset
        Start of the first window relative to the chromosome start.
        ``-2_000_000``, the default, puts the first centre at the first bin,
        the missing input left of the chromosome as zeros; ``0`` is
        ``inference.py``'s default, every window with its full input.
    batch_size
        Windows per batch.
    device
        Torch device (default: ``"cuda"`` if available).

    Returns
    -------
    ContactPrediction
    """
    import torch

    if inputs is not None:
        bulk, profiles = _from_inputs(inputs)
        lengths = inputs.chrom_sizes
    elif data_path is not None and cell_type is not None:
        bulk, profiles = _read_layout(data_path, cell_type)
        lengths = {c: len(v) * PBULK_RES for c, v in bulk.items()}   # length // 50 + 1 bins
    else:
        raise ValueError("give inputs, or data_path and cell_type")
    if genome not in ENDS:
        raise ValueError(f"genome must be one of {list(ENDS)}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    chroms = list(chroms) if chroms is not None else [c for c in ENDS[genome] if c in bulk]
    ctcf = ctcf if ctcf is not None else ctcf_track(genome)
    net = _load_model(model if os.path.exists(str(model)) else checkpoint(str(model)), device)

    # normalize_bulk_dict: the library size is the track's total over all chromosomes
    scale = sum(v.sum() for v in bulk.values()) * 150 / EFFECTIVE_GENOME_SIZE[genome[:2]]
    n50, n500 = INPUT_SIZE // PBULK_RES, INPUT_SIZE // SC_RES
    stripes, centers = {}, {}
    with torch.no_grad():
        for c in chroms:
            # utils.normalize: log(x + 1) of the float32 normalised track
            x1 = torch.log(torch.as_tensor((bulk[c] / scale).astype(np.float32), device=device) + 1)
            x2 = torch.as_tensor(np.asarray(ctcf[c], dtype=np.float32), device=device)
            z = torch.as_tensor(profiles[c].toarray(), dtype=torch.float32, device=device)
            starts = np.arange(offset, ENDS[genome][c] - 5_000_000, RESOLUTION).astype(int)
            out = []
            for b in range(0, len(starts), batch_size):
                s = starts[b:b + batch_size]
                X = torch.stack([_windows(x1, s, PBULK_RES, n50), _windows(x2, s, PBULK_RES, n50)], dim=1)
                Z = _jaccard(_windows(z, s, SC_RES, n500))
                y = net(Z, X)
                y_rev = net(torch.flip(Z, [2]), torch.flip(X, [2]))
                out.append(torch.cat([y, torch.flip(y_rev, [1])], 1).cpu().numpy())
            stripes[c] = np.concatenate(out) if out else np.zeros((0, 2 * PRED_LEN), np.float32)
            centers[c] = (starts + INPUT_SIZE // 2 - RESOLUTION // 2) // RESOLUTION
            logger.info(f"{c}: {len(starts):,} windows")
    return ContactPrediction(stripes=stripes, centers=centers, chrom_sizes={c: int(lengths[c]) for c in chroms})
