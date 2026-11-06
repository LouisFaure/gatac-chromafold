"""
ChromaFold's scATAC inputs: GATAC vs ChromaFold's preprocessing, from the
fragments.

ChromaFold (Gao et al. 2024) predicts Hi-C contacts from three scATAC inputs,
built by preprocessing_pipeline/scATAC_preparation.py:

* metacells: Cicero-style overlapping neighbourhoods (100 cells, fewer than 33
  shared with any other) on an iterative LSI (``metacell_computing``);
* per-cell Tn5 insertion counts in 500 bp tiles, both ends of each fragment
  (``atac_processing`` step 2), summed into metacells (``get_metacell_profile``
  in chromafold/utils.py), from which it correlates tiles;
* the 50 bp insertion track of all cells (``atac_processing`` step 3).

The test starts from the 10x PBMC 5k fragments: ``gatac.io.make_parquet``,
``gatac.pp.compute_metrics``, then ``gatac_chromafold.prepare``, which

1. keeps ChromaFold's cells, TSS enrichment >= 4 and >= 1,000 fragments
   (ChromaFold applies them with ArchR's ``createArrowFiles``, whose TSS
   enrichment is computed differently, so the cells are not ArchR's);
2. counts insertions in 500 bp tiles on chr1-22 and chrX (``_counts.insertion_tiles``);
3. computes the iterative LSI with ChromaFold's ``addIterativeLSI``
   parameters (``dr.iterative_lsi``);
4. builds the inputs: ``_metacells.metacells``, ``_metacells.profiles`` and
   ``_counts.insertion_tracks``.

ChromaFold's functions get the same cells, fragments and LSI. They are copied
below from commit 1c5dd9b (2024-10-04), with the file reads and writes
replaced by in-memory inputs and outputs. Step 4 of ``atac_processing``
(per-cell 50 bp tiles) is left out: it only serves ChromaFold's training,
which samples cells; inference uses the track of all cells.

It then predicts chr19 with the published model three ways: ChromaFold's
inference code (``inference.py``'s ``main``, ``test_Dataset`` and ``test``,
copied below) on ChromaFold's preprocessing; ``gatac_chromafold.predict`` on
the same preprocessing, read from ChromaFold's file layout, which must give
the same predictions; and ``predict`` on GATAC's inputs, end to end.

ChromaFold puts a fragment's second insertion at ``end``, GATAC (and SnapATAC2)
at ``end - 1``, the last base of the fragment. The two differ only for the
fragments with ``end`` a multiple of the tile size, whose insertion ChromaFold
counts one tile further; the test computes those moves and asserts that they
are the only differences. ChromaFold also has one more bin per chromosome when
its length is a multiple of the tile size.

Downloads and results are kept in $GATAC_CHROMAFOLD_DATA (default
~/.cache/gatac-chromafold). ChromaFold's metacells and tiles (about 5 minutes)
are cached there, keyed on their inputs; ``--rerun-reference`` recomputes
them. Run with ``pytest -m reproducibility tests/reproducibility`` or as a script.
"""

import argparse
import hashlib
import os
import pickle
import time
import urllib.request

import anndata as ad
import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import gatac as ga
import gatac_chromafold as gcf
from gatac.io import FragmentFile
from gatac_chromafold._counts import insertion_tiles, insertion_tracks
from gatac_chromafold._metacells import metacells, profiles

CHROMAFOLD_COMMIT = "1c5dd9b07d851d196c31c26024d90c0335f1bd75"
VALID_CHROM = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
PREDICT_CHROM = "chr19"
SAMPLE = "pbmc5k"

# The SnapATAC2 copies of the 10x PBMC 5k fragments and of GENCODE v41
# (snapatac2.datasets.pbmc5k, snapatac2.genome.hg38)
DOWNLOADS = {
    "atac_pbmc_5k.tsv.gz": "5fe44c0f8f76ce1534c1ae418cf0707ca5ef712004eee77c3d98d2d4b35ceaec",
    "gencode_v41_GRCh38.gff3.gz": "b82a655bdb736ca0e463a8f5d00242bedf10fa88ce9d651a017f135c7c4e9285",
}
BASE_URL = "http://renlab.sdsc.edu/kai/public_datasets/"


# ---------------------------------------------------------------------------
# ChromaFold, preprocessing_pipeline/scATAC_preparation.py and chromafold/utils.py
# ---------------------------------------------------------------------------

def atac_processing(frag, valid_barcode, chrom_size, valid_chrom=VALID_CHROM):
    """Steps 1-3 of ``atac_processing``: the 500 bp tile matrices and the 50 bp pseudobulk."""
    #1. read fragment file and filter for valid cell barcode
    dist = frag[2]-frag[1]
    valid_barcode = [str.split(i,"#")[1] for i in valid_barcode]
    frag = frag[frag[3].isin(valid_barcode)]

    ########### Create aggregated tile file for 500bp tiles ###########

    #2a. aggregate for each tile
    frag_stacked = pd.concat([frag[[0,1,3,4]],frag[[0,2,3,4]].rename(columns={2:1})], ignore_index=True)
    frag_stacked[4] = 1 #binarize count
    frag_stacked[1] = frag_stacked[1]//500 # aggregate counts per 500bp tile
    barcodes, barcode_id = np.unique(frag_stacked[3], return_inverse=True)
    frag_stacked[5] = barcode_id
    #2b. create tile files
    tile_dict = {}
    for chrom in list(set(frag_stacked[0])):
        if chrom in valid_chrom:
            frag_grouped = frag_stacked[frag_stacked[0]==chrom][[1,5,4]].groupby([5,1]).sum().reset_index()
            frag_grouped = csr_matrix((frag_grouped[4].astype(np.float64),
                                       (frag_grouped[5].values, frag_grouped[1].values)),
                                       shape=(barcodes.shape[0], chrom_size.loc[chrom].values[0]//500+1))
            tile_dict[chrom] = frag_grouped
    tile_500bp = tile_dict
    del frag_stacked, tile_dict

    ########### Create pseudo-bulk data for evaluation ###########

    #3a. aggregate for 50bp tiles
    frag_stacked = pd.concat([frag[[0,1,4]],frag[[0,2,4]].rename(columns={2:1})], ignore_index=True)
    frag_stacked[4] = 1 #binarize counte
    frag_stacked[1] = frag_stacked[1]//50 # aggregate counts per 50bp tile
    tile_dict = {}
    for chrom in list(set(frag_stacked[0])):
        if chrom in valid_chrom:
            frag_grouped = frag_stacked[frag_stacked[0]==chrom][[1,4]].groupby([1]).sum().reset_index()
            frag_grouped = pd.DataFrame(frag_grouped[4].values,
                                        index = frag_grouped[1].values).reindex(np.arange(0,
                                                                                          chrom_size.loc[chrom].values[0]//50+1)).fillna(0).values
            tile_dict[chrom] = frag_grouped
    del frag_stacked
    return tile_500bp, barcodes.astype(str), tile_dict


def get_max_overlap(y,x):
    return np.max(x.dot(y))


def generate_cicero_metacell(nbrs, max_overlap, sampled_id = [0]):
    order = np.arange(nbrs.shape[0])
    np.random.seed(10)
    np.random.shuffle(order)
    selected_idx = [False]*nbrs.shape[0]
    for i in sampled_id:
        selected_idx[i] = True
    for idx in order:
        selected_cells = nbrs[selected_idx]
        candidate = nbrs[idx]
        overlap = get_max_overlap(candidate,selected_cells)
        if overlap < max_overlap:
            selected_idx[idx] = True
    return selected_idx


def metacell_computing(lsi, n_neighbors = 100, max_overlap = 33):
    lsi.index = [x.split('#')[1] for x in lsi.index]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(lsi.values)
    nbrs = nbrs.kneighbors_graph(lsi.values).toarray()
    selected_idx = generate_cicero_metacell(nbrs, max_overlap = max_overlap)
    metacell_assignment = nbrs[selected_idx,:]
    return metacell_assignment


def get_metacell_profile(sc_atac_dict, nbrs):
    from scipy import sparse
    metacell_sc_atac_dict = {}
    metacell = nbrs
    for chrom in list(sc_atac_dict.keys()):
        metacell_sc_atac_dict[chrom] = (
            sparse.csr_matrix(metacell) * sc_atac_dict[chrom]
        )
        metacell_sc_atac_dict[chrom] = sparse.csr_matrix(
            metacell_sc_atac_dict[chrom].toarray().transpose()
        )

    return metacell_sc_atac_dict


# chromafold/utils.py, testing.py and inference.py, for the predictions

def normalize(tensor, effective_genome_size=0, libsize=0, training=False):
    tensor = torch.tensor(tensor)
    if training:
        assert libsize != 0, "Libsize cannot be 0 during training."
        scale_factor = (libsize * 150) / effective_genome_size
        tensor.div_(torch.as_tensor(scale_factor))
    tensor = torch.log(tensor + 1)
    return tensor


def normalize_bulk_dict(pbulk_dict, effective_genome_size):
    from scipy import sparse
    normalized_pbulk_dict = {}
    libsize = 0
    for chrom in pbulk_dict.keys():
        libsize += pbulk_dict[chrom].toarray().sum()
    scale_factor = (libsize * 150) / (effective_genome_size)
    for chrom in pbulk_dict.keys():
        y = pbulk_dict[chrom].toarray()
        y_t = y / scale_factor
        normalized_pbulk_dict[chrom] = sparse.csr_matrix(y_t)
    return normalized_pbulk_dict


def cpu_jaccard_vstripe(x):
    # calculate jaccard similarity of rows
    scatac_res = 500
    size = x.shape[1]
    eps = 1e-8
    i = np.int16(1000 / scatac_res)

    x = torch.where(x > 0.0, torch.tensor([1.0]), torch.tensor([0.0]))
    num = torch.mm(
        x[2000 * i : 2010 * i, :],
        x[np.r_[:, 0 : 2000 * i, 2010 * i : 4010 * i]].transpose(0, 1),
    )

    x = torch.where(x == 0.0, torch.tensor([1.0]), torch.tensor([0.0]))
    denom = torch.mm(
        x[2000 * i : 2010 * i, :],
        x[np.r_[:, 0 : 2000 * i, 2010 * i : 4010 * i]].transpose(0, 1),
    )
    denom = size - denom

    num = torch.div(num, torch.max(denom, eps * torch.ones_like(denom)))

    return num


def cpu_batch_corcoeff_vstripe(x):
    c = cpu_jaccard_vstripe(x.permute(1, 0))
    c[c != c] = 0
    return c


def get_starts(chrom_list, start_dict, end_dict, chrom_start_offset, chrom_end_offset,
               multi_ct=False, ct_list=[], step=5e4):
    ctl = []
    startl = []
    chroml = []
    for chrom in chrom_list:
        chrom = "chr{}".format(chrom)
        cur_starts = list(
            np.arange(
                start_dict[chrom] + chrom_start_offset,
                end_dict[chrom] - chrom_end_offset,
                step,
            ).astype(int)
        )
        startl = startl + cur_starts
        chroml = chroml + list(np.repeat(chrom, len(cur_starts)))
    return ctl, startl, chroml


class test_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_size, pbulk_dict, scatac_dict, ctcf_dict, list_cts, list_starts,
                 list_chroms, transform, is_training=False):
        self.input_size = input_size
        self.pbulk_dict = pbulk_dict
        self.scatac_dict = scatac_dict
        self.ctcf_dict = ctcf_dict
        self.list_cts = list_cts
        self.list_starts = list_starts
        self.list_chroms = list_chroms
        self.is_training = is_training
        self.transform = transform

    def __len__(self):
        return len(self.list_starts)

    def __getitem__(self, index):
        chrom = self.list_chroms[index]
        start = self.list_starts[index]

        shift_input_by = 0
        pbulk_res = 50
        scatac_res = 500

        input_window = int(self.input_size / pbulk_res)
        atac_start_ind = int(start / pbulk_res)
        sc_input_window = int(self.input_size / scatac_res)
        sc_atac_start_ind = int(start / scatac_res)

        X1 = self.pbulk_dict[chrom]
        X1 = X1[
            max(0, atac_start_ind + shift_input_by) : atac_start_ind
            + shift_input_by
            + input_window
        ]
        X1 = self.transform(X1.toarray().astype(np.float32).T)

        Z1 = self.scatac_dict[chrom]
        Z1 = Z1[
            max(0, sc_atac_start_ind + shift_input_by) : sc_atac_start_ind
            + shift_input_by
            + sc_input_window,
            :,
        ]

        Z1 = Z1.toarray().astype(np.float32).T
        Z1 = torch.tensor(Z1)

        X2 = self.ctcf_dict[chrom]
        X2 = X2[max(0, atac_start_ind) : atac_start_ind + input_window]
        X2 = torch.tensor(X2.toarray().astype(np.float32).T)

        if start < 0:
            desired_shape = (1, 80200)
            # Determine the number of zeros to add on the left side
            num_zeros = desired_shape[1] - X1.shape[1]
            # Pad the tensor with zeros on the left side
            X1 = torch.cat([torch.zeros((X1.shape[0], num_zeros)), X1], dim=1)
            X2 = torch.cat([torch.zeros((X2.shape[0], num_zeros)), X2], dim=1)

            desired_shape = (Z1.shape[0], 8020)
            num_zeros = desired_shape[1] - Z1.shape[1]
            Z1 = torch.cat([torch.zeros((Z1.shape[0], num_zeros)), Z1], dim=1)

        Z1 = cpu_batch_corcoeff_vstripe(Z1)

        X1_rev = torch.empty_like(X1).copy_(X1)
        X1_rev = torch.flip(X1_rev, [1])
        X2_rev = torch.empty_like(X2).copy_(X2)
        X2_rev = torch.flip(X2_rev, [1])
        Z1_rev = torch.empty_like(Z1).copy_(Z1)
        Z1_rev = torch.flip(Z1_rev, [1])

        return Z1, Z1_rev, X1, X1_rev, X2, X2_rev


def test(test_loader, model, device):
    y_hat_list = []
    model.eval()
    for Z2, Z2_rev, X1, X1_rev, X2, X2_rev in test_loader:
        # left interactions
        X1 = X1.to(device)
        X2 = X2.to(device)
        X = torch.cat((X1, X2), axis = 1)
        Z2 = Z2.to(device)
        y_hat = model(Z2, X)

        # right interactions
        X1_rev = X1_rev.to(device)
        X2_rev = X2_rev.to(device)
        X_rev = torch.cat((X1_rev, X2_rev), axis = 1)
        Z2_rev = Z2_rev.to(device)
        y_hat_rev = model(Z2_rev, X_rev)

        y_hat = torch.cat((y_hat, torch.flip(y_hat_rev, [1])), 1)
        y_hat_list.append(y_hat.detach().cpu())
    return y_hat_list


test.__test__ = False   # ChromaFold's name; not a pytest test


def chromafold_inference(pbulk_dict, scatac_dict, ctcf_dict, chrom, model_path, offset):
    """inference.py's main() for one chromosome of hg38, given its loaded inputs
    (load_atac_eval and load_ctcf_motif)."""
    from gatac_chromafold._model import branch_cov, branch_pbulk, trunk
    from gatac_chromafold.prediction import ENDS

    device = torch.device("cuda")
    input_size = 4010000
    test_step = 1e04
    start_dict = {c: 0 for c in ENDS["hg38"]}
    end_dict = ENDS["hg38"]
    effective_genome_size = 2913022398
    normalized_pbulk_dict = normalize_bulk_dict(pbulk_dict, effective_genome_size)
    test_dataset = test_Dataset(
        input_size, normalized_pbulk_dict, scatac_dict, ctcf_dict,
        *get_starts(chrom_list=[chrom[3:]], start_dict=start_dict, end_dict=end_dict,
                    chrom_start_offset=offset, chrom_end_offset=5000000, multi_ct=False,
                    ct_list=["ct"], step=test_step),
        transform=normalize,
    )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0)
    mod_branch_pbulk = nn.DataParallel(branch_pbulk(), device_ids=[0])
    mod_branch_cov = nn.DataParallel(branch_cov(), device_ids=[0])
    model = nn.DataParallel(trunk(mod_branch_pbulk, mod_branch_cov), device_ids=[0])
    model.to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        y_hat_list = test(test_loader, model, device)
    return np.concatenate([x.numpy() for x in y_hat_list], axis=0)


# ---------------------------------------------------------------------------
# Data, cache
# ---------------------------------------------------------------------------

DATA = os.path.abspath(os.path.expanduser(
    os.environ.get("GATAC_CHROMAFOLD_DATA") or "~/.cache/gatac-chromafold"))


def _path(name):
    return os.path.join(DATA, name)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _download(name):
    """``name`` from BASE_URL into the data directory, unless already there with its checksum."""
    path = _path(name)
    if os.path.exists(path) and _sha256(path) == DOWNLOADS[name]:
        return path
    os.makedirs(DATA, exist_ok=True)
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(BASE_URL + name, path + ".part")
    if _sha256(path + ".part") != DOWNLOADS[name]:
        raise RuntimeError(f"{name}: checksum mismatch")
    os.replace(path + ".part", path)
    return path


def _key(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(pickle.dumps(p, protocol=5))
    return h.hexdigest()


def _cached(name, inputs, run, rerun=False):
    """``run()``'s result, from the data directory while ``inputs`` (a :func:`_key`) is unchanged."""
    path = _path(f"reference_{name}.pkl")
    if not rerun and os.path.exists(path):
        with open(path, "rb") as f:
            entry = pickle.load(f)
        if entry["inputs"] == inputs:
            print(f"{name}: cached ChromaFold result")
            return entry["value"]
        print(f"{name}: inputs changed, rerunning ChromaFold")
    value = run()
    with open(path + ".tmp", "wb") as f:
        pickle.dump({"inputs": inputs, "value": value}, f, protocol=5)
    os.replace(path + ".tmp", path)
    return value


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _moved_insertions(frag, size):
    """Per (chrom, cell, bin): ChromaFold's count minus GATAC's, from the fragments whose
    ``end`` is a multiple of ``size`` (ChromaFold: bin ``end // size``; GATAC: the bin before)."""
    m = frag[frag["end"] % size == 0]
    b = m["end"] // size
    return pd.concat([
        pd.DataFrame({"chrom": m["chrom"], "barcode": m["barcode"], "bin": b, "d": 1}),
        pd.DataFrame({"chrom": m["chrom"], "barcode": m["barcode"], "bin": b - 1, "d": -1}),
    ]).groupby(["chrom", "barcode", "bin"], observed=True)["d"].sum()


def _padded(a, n_bins):
    """``a`` (cells × bins, or bins) with zero bins up to ``n_bins``."""
    if sp.issparse(a):
        return sp.hstack([a, sp.csr_matrix((a.shape[0], n_bins - a.shape[1]))]).tocsr()
    return np.concatenate([a, np.zeros(n_bins - len(a), a.dtype)])


# ---------------------------------------------------------------------------


@pytest.mark.reproducibility
def test_chromafold_inputs(rerun_reference=False):
    fragments_tsv = _download("atac_pbmc_5k.tsv.gz")
    gff = _download("gencode_v41_GRCh38.gff3.gz")
    sizes = {c: int(ga.pp.HG38[c]) for c in VALID_CHROM}

    # --- GATAC: fragments, cell metrics, then the plugin -------------------------
    t_pre = {}
    parquet, t_pre["make_parquet"] = _timed(
        lambda: ga.io.make_parquet(fragments_tsv, _path("atac_pbmc_5k.parquet")))
    metrics, t_pre["compute_metrics"] = _timed(lambda: ga.pp.compute_metrics(parquet, gff))
    inputs, prepare_time = _timed(lambda: gcf.prepare(parquet, "hg38", metrics=metrics))
    tiles, profile, bulk = inputs.tiles, inputs.metacells, inputs.pseudobulk
    M = tiles.obsm["chromafold_metacells"]
    cells = list(tiles.obs_names)
    print(f"{len(cells):,} cells; " + ", ".join(f"{k} {v:.1f}s" for k, v in t_pre.items())
          + f", prepare {prepare_time:.1f}s")

    # the steps of prepare that ChromaFold's code does, timed on their own
    t = {}
    sizes = inputs.chrom_sizes
    _, t["metacells"] = _timed(lambda: metacells(tiles, use_rep="X_iterative_lsi", key_added="timing"))
    _, t["tiles"] = _timed(lambda: insertion_tiles(parquet, sizes, cells, tile_size=500))
    _, t["profiles"] = _timed(lambda: profiles(tiles, "chromafold_metacells"))
    _, t["track"] = _timed(lambda: insertion_tracks(
        parquet, sizes, bin_size=50, groups=pd.Series("all", index=cells)))
    gatac_time = sum(t.values())

    # --- ChromaFold, given the same cells, fragments and LSI -------------------
    lsi = pd.DataFrame(tiles.obsm["X_iterative_lsi"], index=[f"{SAMPLE}#{c}" for c in cells])
    frag = FragmentFile(parquet).read_categorical(["chrom", "start", "end", "barcode", "count"]).to_pandas()
    frag = frag[frag["chrom"].isin(VALID_CHROM) & frag["barcode"].isin(cells)]
    frag["barcode"] = frag["barcode"].cat.remove_unused_categories()
    print(f"{len(frag):,} fragments")

    def run_metacells():
        mask, seconds = _timed(lambda: metacell_computing(lsi.copy()))
        return sp.csr_matrix(mask.astype(np.float32)), seconds

    def run_tiles():
        f = frag.rename(columns={"chrom": 0, "start": 1, "end": 2, "barcode": 3, "count": 4})
        (tile_500bp, barcodes, pbulk), seconds = _timed(
            lambda: atac_processing(f, list(lsi.index), pd.DataFrame({1: pd.Series(sizes)})))
        return {
            "barcodes": barcodes,
            "tiles": {c: m.astype(np.int32) for c, m in tile_500bp.items()},
            "pbulk": {c: v.ravel().astype(np.int32) for c, v in pbulk.items()},
        }, seconds

    mask, cf_mc_time = _cached("metacells", _key(CHROMAFOLD_COMMIT, lsi), run_metacells, rerun_reference)
    ref, cf_tile_time = _cached("tiles", _key(CHROMAFOLD_COMMIT, DOWNLOADS, cells, sizes),
                                run_tiles, rerun_reference)
    # cells are in the same order in GATAC's matrix, the LSI and ChromaFold's tiles (np.unique)
    assert list(ref["barcodes"]) == cells
    cf_tiles = {c: ref["tiles"][c].astype(np.float64) for c in VALID_CHROM}
    cf_profiles, cf_prof_time = _timed(lambda: get_metacell_profile(cf_tiles, mask.toarray()))
    cf_time = cf_mc_time + cf_tile_time + cf_prof_time

    # --- comparisons -----------------------------------------------------------
    # metacells: identical, in the same order
    mc_equal = mask.shape == M.T.shape and (mask != M.T.astype(np.float32)).nnz == 0

    chrom_of_tile = tiles.var["chrom"].to_numpy()

    def cols(c):
        i = np.flatnonzero(chrom_of_tile == c)
        return slice(i[0], i[-1] + 1)

    # 500 bp tiles: identical but for the moved insertions
    row = pd.Index(cells)
    moved = _moved_insertions(frag, 500)
    n_moved, tile_diff = 0, 0
    for c in VALID_CHROM:
        theirs = ref["tiles"][c]
        g = _padded(tiles.X[:, cols(c)].astype(np.int32), theirs.shape[1])
        d = moved.loc[c]
        n_moved += int((d > 0).sum())
        expected = g + sp.csr_matrix(
            (d.to_numpy(), (row.get_indexer(d.index.get_level_values(0).astype(str)),
                            d.index.get_level_values(1))),
            shape=g.shape)
        tile_diff += (theirs != expected).nnz

    # metacell profiles: profiles() of ChromaFold's own tiles must give its profiles;
    # end to end, they differ by the moved insertions
    cf_adata = ad.AnnData(sp.hstack([cf_tiles[c] for c in VALID_CHROM]).tocsr(),
                          obs=pd.DataFrame(index=cells))
    cf_adata.obsm["metacells"] = M
    cf_aggregate = profiles(cf_adata, "metacells")
    edges = np.cumsum([0] + [cf_tiles[c].shape[1] for c in VALID_CHROM])
    agg_diff, prof_diff, prof_nnz = 0, 0, 0
    for i, c in enumerate(VALID_CHROM):
        theirs = cf_profiles[c]
        agg_diff += (cf_aggregate.X[:, edges[i]:edges[i + 1]].T != theirs).nnz
        prof_diff += (_padded(profile.X[:, cols(c)], theirs.shape[0]).T != theirs).nnz
        prof_nnz += theirs.nnz
    del cf_adata, cf_aggregate, cf_tiles, cf_profiles

    # 50 bp track: identical but for the moved insertions
    moved50 = _moved_insertions(frag, 50).groupby(level=["chrom", "bin"], observed=True).sum()
    bulk_diff = 0
    for c in VALID_CHROM:
        theirs = ref["pbulk"][c]
        expected = _padded(bulk.track("all", c), len(theirs)).astype(np.int64)
        d = moved50.loc[c]
        np.add.at(expected, d.index.to_numpy(), d.to_numpy())
        bulk_diff += int((theirs != expected).sum())

    # --- predictions on PREDICT_CHROM ------------------------------------------
    # (a) ChromaFold's inference.py code on ChromaFold's preprocessing, and predict()
    #     reading that preprocessing from ChromaFold's file layout: the same inputs
    # (b) predict() on GATAC's inputs, against (a): end to end
    from gatac_chromafold.data import checkpoint, ctcf_track

    layout = _path("chromafold_layout")
    os.makedirs(os.path.join(layout, "atac"), exist_ok=True)
    with open(os.path.join(layout, "atac", "cf_tile_pbulk_50bp_dict.p"), "wb") as f:
        pickle.dump({c: ref["pbulk"][c].astype(np.float64)[:, None] for c in VALID_CHROM}, f)
    with open(os.path.join(layout, "atac", "cf_tile_500bp_dict.p"), "wb") as f:
        pickle.dump({c: ref["tiles"][c].astype(np.float64) for c in VALID_CHROM}, f)
    np.save(os.path.join(layout, "atac", "cf_tile_500bp_barcode.npy"), np.asarray(cells))
    pd.DataFrame(mask.toarray().astype(np.int64), columns=cells).to_csv(
        os.path.join(layout, "atac", "cf_metacell_mask.csv"))

    chrom, offset = PREDICT_CHROM, -2_000_000
    gcf.predict(inputs, chroms=[chrom], offset=offset)      # downloads, CUDA start-up
    pred_same, pred_time = _timed(lambda: gcf.predict(
        data_path=layout, cell_type="cf", chroms=[chrom], offset=offset))
    pred_gatac = gcf.predict(inputs, chroms=[chrom], offset=offset)
    track = ctcf_track("hg38")
    # ChromaFold's code in full float32, as predict() runs: cuDNN's TF32 convolutions,
    # its default on Ampere and later GPUs, move its predictions by up to ~7e-3
    torch.backends.cudnn.allow_tf32 = False
    y_ref, ref_time = _timed(lambda: chromafold_inference(
        {c: sp.csr_matrix(ref["pbulk"][c].astype(np.float64)[:, None]) for c in VALID_CHROM},
        get_metacell_profile({chrom: ref["tiles"][chrom].astype(np.float64)}, mask.toarray()),
        {chrom: sp.csc_matrix(track[chrom].astype(np.float64)[None, :]).T},   # load_ctcf_motif
        chrom, checkpoint("CTCFmotif"), offset))
    same_diff = float(np.abs(pred_same.stripes[chrom] - y_ref).max())
    y_gatac = pred_gatac.stripes[chrom]
    e2e_r = float(np.corrcoef(y_gatac.ravel(), y_ref.ravel())[0, 1])
    e2e_diff = np.abs(y_gatac - y_ref)

    n_ins = 2 * len(frag)
    log = [
        f"Cells:\t{len(cells):,} (TSS enrichment >= 4, >= 1,000 fragments), {len(frag):,} fragments",
        "GATAC preprocessing:\t" + ", ".join(f"{k} {v:.2f}s" for k, v in t_pre.items())
        + f"; gatac_chromafold.prepare (cell filters, tiles, iterative LSI, metacells, profiles, "
          f"track) {prepare_time:.2f}s",
        f"GATAC:\t{gatac_time:.2f}s (metacells {t['metacells']:.2f}s, 500 bp tiles {t['tiles']:.2f}s, "
        f"metacell profiles {t['profiles']:.2f}s, 50 bp track {t['track']:.2f}s)",
        f"ChromaFold:\t{cf_time:.2f}s (metacells {cf_mc_time:.2f}s, 500 bp tiles and 50 bp track "
        f"{cf_tile_time:.2f}s, metacell profiles {cf_prof_time:.2f}s)",
        f"Speedup:\t{cf_time / gatac_time:.1f}x",
        f"Metacells:\t{M.shape[1]:,}, identical: {mc_equal}",
        f"500 bp tiles:\t{tile_diff} entries differ beyond the {n_moved:,} insertions at end % 500 == 0 "
        f"({n_moved / n_ins:.4%} of {n_ins:,})",
        f"Metacell profiles:\t{agg_diff} entries differ given ChromaFold's tiles; end to end "
        f"{prof_diff:,} of {prof_nnz:,} differ, from the moved insertions",
        f"50 bp track:\t{bulk_diff} bins differ beyond the "
        f"{int((frag['end'] % 50 == 0).sum()):,} insertions at end % 50 == 0",
        f"Predictions ({chrom}, {len(y_ref):,} windows):\tGATAC predict {pred_time:.2f}s, "
        f"ChromaFold inference {ref_time:.2f}s ({ref_time / pred_time:.1f}x)",
        f"  same inputs:\tmax |difference| {same_diff:.2e}",
        f"  GATAC inputs vs ChromaFold's:\tr = {e2e_r:.5f}, |difference| mean {e2e_diff.mean():.4f}, "
        f"max {e2e_diff.max():.3f} (prediction sd {y_ref.std():.3f})",
    ]
    with open(_path("chromafold_inputs.log"), "w") as f:
        for line in log:
            print(line)
            f.write(line + "\n")
    assert mc_equal, "metacells differ"
    assert tile_diff == 0, f"{tile_diff} tile entries differ"
    assert agg_diff == 0, f"{agg_diff} metacell profile entries differ"
    assert bulk_diff == 0, f"{bulk_diff} 50 bp bins differ"
    # float32 rounding: GPU against CPU co-accessibility, batching (measured 2.9e-6 on an A100)
    assert same_diff < 1e-4, f"predictions on the same inputs differ by {same_diff}"
    # the moved insertions (measured r = 0.99985)
    assert e2e_r > 0.999, f"end-to-end prediction correlation {e2e_r}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaFold's scATAC inputs: GATAC vs ChromaFold")
    parser.add_argument("--rerun-reference", action="store_true",
                        help="Rerun ChromaFold's preprocessing even when its result is cached")
    args = parser.parse_args()
    test_chromafold_inputs(rerun_reference=args.rerun_reference)
