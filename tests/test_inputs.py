"""``prepare`` and ``ChromaFoldInputs.write`` on the synthetic cell type of conftest.py."""
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import gatac as ga
import gatac_chromafold as gcf

CHROM_SIZES = {"chr1": 2_000_000, "chr2": 1_500_000, "chrX": 1_000_000, "chrM": 16_500}  # as in conftest.py
SMALL = {"n_neighbors": 15, "max_overlap": 5}


def _chromafold_metacells(lsi, n_neighbors, max_overlap):
    """ChromaFold's metacell_computing (preprocessing_pipeline/scATAC_preparation.py), but for I/O."""
    from sklearn.neighbors import NearestNeighbors

    def get_max_overlap(y, x):
        return np.max(x.dot(y))

    def generate_cicero_metacell(nbrs, max_overlap, sampled_id=[0]):
        order = np.arange(nbrs.shape[0])
        np.random.seed(10)
        np.random.shuffle(order)
        selected_idx = [False] * nbrs.shape[0]
        for i in sampled_id:
            selected_idx[i] = True
        for idx in order:
            if get_max_overlap(nbrs[idx], nbrs[selected_idx]) < max_overlap:
                selected_idx[idx] = True
        return selected_idx

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(lsi)
    nbrs = nbrs.kneighbors_graph(lsi).toarray()
    return nbrs[generate_cicero_metacell(nbrs, max_overlap=max_overlap), :]


@pytest.fixture(scope="module")
def inputs(celltype):
    return gcf.prepare(celltype["parquet"], CHROM_SIZES, cells=celltype["cells"], use_rep="X_emb",
                       params=SMALL)


def test_prepare_builds_chromafolds_inputs(inputs, celltype):
    """ChromaFold's chromosomes, its metacells, their summed tiles and the track of all insertions."""
    assert list(inputs.chrom_sizes) == ["chr1", "chr2", "chrX"]
    t = inputs.tiles
    assert set(t.var["chrom"]) == {"chr1", "chr2", "chrX"}
    assert list(t.obs_names) == list(celltype["cells"].obs_names)

    M = t.obsm["chromafold_metacells"]
    want = _chromafold_metacells(t.obsm["X_emb"], 15, 5)
    assert np.array_equal(M.T.toarray(), want)
    assert np.allclose(inputs.metacells.X.toarray(), (M.T @ t.X.astype(np.float64)).toarray())
    assert list(inputs.metacells.obs_names) == list(t.uns["chromafold_metacells"]["names"])

    f = celltype["fragments"]
    n_ins = 2 * int((f.chrom != "chrM").sum())
    assert t.X.sum() == n_ins
    # each end of each fragment in its 500 bp tile, both ends in one tile counting twice
    s = f[f.chrom != "chrM"]
    ends = pd.DataFrame({"cell": np.r_[s.barcode, s.barcode], "chrom": np.r_[s.chrom, s.chrom],
                         "start": np.r_[s.start, s.end - 1] // 500 * 500})
    n = ends.groupby(["cell", "chrom", "start"]).size()
    col = pd.Series(np.arange(t.n_vars), index=pd.MultiIndex.from_arrays([t.var["chrom"], t.var["start"]]))
    want = sp.csr_matrix((n.to_numpy(), (t.obs_names.get_indexer(n.index.get_level_values(0)),
                                         col.loc[list(zip(n.index.get_level_values(1), n.index.get_level_values(2)))])),
                         shape=t.shape)
    assert (t.X != want).nnz == 0
    assert np.array_equal(t.obs["n_unique"], s.groupby("barcode").size().loc[t.obs_names])
    assert inputs.library_size == n_ins
    for c in inputs.chrom_sizes:
        tr = inputs.pseudobulk.track("all", c)
        s = f[f.chrom == c]
        want = np.bincount(np.r_[s.start, s.end - 1] // 50, minlength=len(tr))
        assert np.array_equal(tr, want)


def test_write_is_read_as_chromafold_reads_it(inputs, tmp_path):
    """The files load as ChromaFold's load_atac_eval loads them, and give the same metacell profiles."""
    paths = inputs.write(tmp_path, "ct")
    assert {p.name for p in (tmp_path / "atac").iterdir()} == {
        "ct_tile_pbulk_50bp_dict.p", "ct_tile_500bp_dict.p", "ct_tile_500bp_barcode.npy", "ct_metacell_mask.csv"}

    bulk = pickle.load(open(paths["pseudobulk"], "rb"))
    tiles = pickle.load(open(paths["tiles"], "rb"))
    mask = pd.read_csv(paths["metacell_mask"], index_col=0)
    barcodes = np.load(paths["barcodes"])
    assert list(barcodes) == list(inputs.tiles.obs_names) == list(mask.columns)
    for c, n in inputs.chrom_sizes.items():
        assert bulk[c].shape == (n // 50 + 1, 1)
        assert bulk[c].sum() == inputs.pseudobulk.track("all", c).sum()
        assert tiles[c].shape == (inputs.tiles.n_obs, n // 500 + 1)
        # chromafold/utils.py get_metacell_profile
        profile = sp.csr_matrix((sp.csr_matrix(mask.values) * tiles[c]).toarray().transpose())
        i = np.flatnonzero(inputs.metacells.var["chrom"] == c)
        ours = inputs.metacells.X[:, i[0]:i[-1] + 1].toarray().T
        assert np.array_equal(profile.toarray()[:len(ours)], ours)


def test_prepare_filters_cells_on_metrics(celltype):
    """Without cells, the cells passing ChromaFold's TSS and fragment thresholds, and an iterative LSI."""
    names = list(celltype["cells"].obs_names)
    metrics = pd.DataFrame({"barcode": names, "n_unique": 2_000, "tsse_score": 5.0})
    metrics.loc[::4, "tsse_score"] = 3.0
    metrics.loc[1::4, "n_unique"] = 900
    lsi = {"n_comps": 5, "iterations": 2, "n_features": 2_000, "total_features": 5_000,
           "cluster_params": {"resolution": 0.2}}
    got = gcf.prepare(celltype["parquet"], CHROM_SIZES, metrics=metrics, params={**SMALL, "lsi": lsi})
    assert list(got.tiles.obs_names) == [n for i, n in enumerate(names) if i % 4 >= 2]
    assert got.tiles.obsm["X_iterative_lsi"].shape[1] == 5
    assert got.tiles.uns["chromafold"]["use_rep"] == "X_iterative_lsi"
    with pytest.raises(ValueError, match="metrics"):
        gcf.prepare(celltype["parquet"], CHROM_SIZES)


def test_extension_and_command(celltype, tmp_path):
    """Reached as gatac.chromafold, and `gatac chromafold prepare` writes the inputs."""
    assert "chromafold" in ga.extensions()
    assert ga.chromafold.prepare is gcf.prepare
    # the lazy names stay the functions once their modules are imported
    assert callable(gcf.predict) and callable(gcf.predict) and ga.chromafold.predict is gcf.predict
    h5ad = tmp_path / "cells.h5ad"
    celltype["cells"].write_h5ad(h5ad)
    gatac = Path(sys.executable).parent / "gatac"
    r = subprocess.run([str(gatac), "chromafold", "prepare", str(celltype["parquet"]), "-o", str(tmp_path / "out"),
                        "-c", "ct", "-g", "hg38", "--cells", str(h5ad), "--use-rep", "X_emb"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out" / "atac" / "ct_metacell_mask.csv").exists()
