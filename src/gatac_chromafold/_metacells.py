"""
ChromaFold's metacells and their profiles.

ChromaFold's ``metacell_computing`` (preprocessing_pipeline/scATAC_preparation.py)
follows Cicero: each metacell is a seed cell's ``n_neighbors`` nearest cells in
the LSI, and seeds are chosen greedily, a cell becoming one when its
neighbourhood shares fewer than ``max_overlap`` cells with every seed's. Its
own quirks are kept, so that the metacells are ChromaFold's: the first cell is
always the first seed, the others are visited in the order of NumPy's legacy
``RandomState(10)`` shuffle, and ``max_overlap`` is a number of cells.

:func:`metacells` finds the neighbours on the GPU and keeps, per cell, the
seeds it belongs to, instead of ChromaFold's dense cells × cells matrix.
"""

from __future__ import annotations

import logging

import anndata as ad
import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit

logger = logging.getLogger(__name__)


@njit(cache=True)
def _greedy_seeds(nbrs: np.ndarray, order: np.ndarray, max_overlap: int) -> np.ndarray:
    """
    Seed mask of the greedy selection: ``order[0]``, then each cell of ``order``
    whose neighbourhood ``nbrs[i]`` shares fewer than ``max_overlap`` cells with
    every seed's.

    The overlaps of a candidate with all the seeds come from the seeds each of
    its neighbours belongs to, kept as a linked list per cell, so a candidate
    costs the size of those lists rather than a pass over every seed.
    """
    n, k = nbrs.shape
    seed = np.zeros(n, np.bool_)
    head = np.full(n, -1, np.int64)  # per cell: its last membership entry
    cap = 1 << 16
    e_seed = np.empty(cap, np.int64)  # per entry: the seed (its rank) ...
    e_next = np.empty(cap, np.int64)  # ... and the cell's previous entry
    n_entries = 0
    shared = np.zeros(n, np.int64)
    touched = np.empty(n, np.int64)
    n_seeds = 0
    for t in range(n):
        i = order[t]
        if seed[i]:
            continue
        n_touched = 0
        most = 0
        for a in range(k):
            e = head[nbrs[i, a]]
            while e >= 0:
                s = e_seed[e]
                if shared[s] == 0:
                    touched[n_touched] = s
                    n_touched += 1
                shared[s] += 1
                if shared[s] > most:
                    most = shared[s]
                e = e_next[e]
        for b in range(n_touched):
            shared[touched[b]] = 0
        if most >= max_overlap:
            continue
        seed[i] = True
        if n_entries + k > cap:
            cap = max(2 * cap, n_entries + k)
            grown = np.empty(cap, np.int64)
            grown[:n_entries] = e_seed[:n_entries]
            e_seed = grown
            grown = np.empty(cap, np.int64)
            grown[:n_entries] = e_next[:n_entries]
            e_next = grown
        for a in range(k):
            c = nbrs[i, a]
            e_seed[n_entries] = n_seeds
            e_next[n_entries] = head[c]
            head[c] = n_entries
            n_entries += 1
        n_seeds += 1
    return seed


def _knn(X: np.ndarray, k: int) -> np.ndarray:
    """Exact euclidean kNN indices, ``(n, k)``, each cell its own first neighbour."""
    from cuml.neighbors import NearestNeighbors

    Xg = cp.asarray(X, dtype=cp.float32)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", output_type="cupy").fit(Xg)
    _, idx = nn.kneighbors(Xg)
    return cp.asnumpy(idx).astype(np.int64)


def metacells(
    adata: ad.AnnData,
    use_rep: str = "X_iterative_lsi",
    n_neighbors: int = 100,
    max_overlap: int = 33,
    random_state: int = 10,
    key_added: str = "chromafold_metacells",
) -> None:
    """
    ChromaFold's metacells of the cells of ``adata``, on ``adata.obsm[use_rep]``.

    Sets ``adata.obsm[key_added]``, the sparse cells × metacells membership
    (CSR, float32 ones), metacells in the order of their seed cells, and
    ``adata.uns[key_added]``: ``names`` (the seed cells) and ``params``.
    """
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm[{use_rep!r}] not found")
    X = np.asarray(adata.obsm[use_rep])
    n = X.shape[0]
    if not 1 <= n_neighbors <= n:
        raise ValueError(f"n_neighbors must be between 1 and the number of cells ({n})")

    nbrs = _knn(X, n_neighbors)
    order = np.arange(n)
    np.random.RandomState(random_state).shuffle(order)
    seed = _greedy_seeds(nbrs, np.concatenate([[0], order]), int(max_overlap))

    seeds = np.flatnonzero(seed)
    n_meta = len(seeds)
    M = sp.csr_matrix(
        (
            np.ones(n_meta * n_neighbors, np.float32),
            (nbrs[seeds].ravel(), np.repeat(np.arange(n_meta), n_neighbors)),
        ),
        shape=(n, n_meta),
    )
    adata.obsm[key_added] = M
    adata.uns[key_added] = {
        "names": np.asarray(adata.obs_names[seeds]),
        "params": {
            "use_rep": use_rep,
            "n_neighbors": int(n_neighbors),
            "max_overlap": int(max_overlap),
            "random_state": int(random_state),
        },
    }
    logger.info(
        f"{n_meta:,} metacells of {n_neighbors} cells from {n:,} cells; "
        f"{int((M.getnnz(axis=1) > 0).sum()):,} cells in at least one"
    )


def profiles(adata: ad.AnnData, key: str = "chromafold_metacells", max_nnz: int = 1 << 27) -> ad.AnnData:
    """
    Metacells × features sums of ``adata.X``, ``M.T @ X`` for the membership
    ``M = adata.obsm[key]`` (ChromaFold's ``get_metacell_profile``), on the GPU
    in slabs of cells of at most ``max_nnz`` entries.
    """
    M = sp.csr_matrix(adata.obsm[key], dtype=np.float64)
    stored = adata.uns.get(key, {})
    names = stored.get("names") if isinstance(stored, dict) else None
    if names is None or len(names) != M.shape[1]:
        names = [f"{key}_{i}" for i in range(M.shape[1])]
    X = sp.csr_matrix(adata.X)
    bounds = np.unique(np.append(np.searchsorted(X.indptr, np.arange(0, X.nnz, max_nnz), side="right") - 1, X.shape[0]))
    total = None
    for r0, r1 in zip(bounds[:-1], bounds[1:]):
        Mt = cusp.csr_matrix(M[r0:r1].T.tocsr())
        if Mt.nnz == 0:
            continue
        part = X[r0:r1]
        part = cusp.csr_matrix(
            (cp.asarray(part.data).astype(cp.float64), cp.asarray(part.indices), cp.asarray(part.indptr)),
            shape=part.shape,
        )
        prod = Mt @ part
        total = prod if total is None else total + prod
        del Mt, part, prod
    result = sp.csr_matrix(total.get()) if total is not None else sp.csr_matrix((M.shape[1], X.shape[1]))
    result.sort_indices()
    del total
    cp.get_default_memory_pool().free_all_blocks()
    obs = pd.DataFrame({"n_cells": np.diff(M.tocsc().indptr)}, index=pd.Index(np.asarray(names).astype(str)))
    return ad.AnnData(X=result, obs=obs, var=adata.var.copy())
