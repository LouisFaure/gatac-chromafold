"""
ChromaFold's published files: model checkpoints (GitHub) and CTCF motif
tracks (Google Drive), downloaded once into $GATAC_CHROMAFOLD_DATA (default
~/.cache/gatac-chromafold) and checked against their sha256.

The CTCF tracks are pickles. They are read with an unpickler that only builds
the NumPy and SciPy objects they hold, so a tampered file cannot run code, and
converted to ``.npz`` on first use.
"""
from __future__ import annotations

import hashlib
import io
import os
import pickle
import urllib.request
from pathlib import Path

import numpy as np

COMMIT = "1c5dd9b07d851d196c31c26024d90c0335f1bd75"

#: Checkpoints of ChromaFold's repository: name -> sha256. "CTCFmotif" is the
#: published model; "CTCFChIP" takes a CTCF ChIP-seq track instead of the
#: motif track. Both share the architecture in _model.py.
CHECKPOINTS = {
    "CTCFmotif": ("chromafold_CTCFmotif.pth.tar",
                  "c3116053093d87a9e95d9e79ee938d42f5e3a5942e4f07a3cd86e9fefc0512e4"),
    "CTCFChIP": ("chromafold_CTCFChIP_fullmodel.pth.tar",
                 "ef45763857522adc80e50c4e648f364688f2d5b7b720e9b808be19e193ed3c15"),
}

#: ChromaFold's precomputed CTCF motif score tracks: genome -> (Drive file id, sha256)
CTCF_TRACKS = {
    "hg38": ("1E9K7zU8lPbVvMrN04PE1w7sEjJOIm2EH",
             "a0719ad09348c440a8bb22a01e2faa41db1caa3a898a948691cbce59f0b8a2e5"),
    "hg19": ("141mb1c48rllqvLIJtsmN7kIrQqQb66Ar",
             "ed10bcc6d5e1de60a63f561dd0d56165fa837f12d3f4f1937b7bb9092343714b"),
    "mm10": ("1nAU-q96QlycLKYgutYHEg_EVpx5khpWe",
             "a7ddd7030c1226a61a9a0d24dad11dfd2622c856665ce0aa6e78463b44511c85"),
}


def data_dir() -> Path:
    d = Path(os.path.expanduser(os.environ.get("GATAC_CHROMAFOLD_DATA") or "~/.cache/gatac-chromafold"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(path: Path, sha256: str, download) -> Path:
    """``path``, downloaded with ``download(tmp_path)`` unless already there with its checksum."""
    if path.exists() and _sha256(path) == sha256:
        return path
    tmp = path.with_name(path.name + ".part")
    download(tmp)
    got = _sha256(tmp)
    if got != sha256:
        tmp.unlink()
        raise RuntimeError(f"{path.name}: sha256 {got}, expected {sha256}")
    os.replace(tmp, path)
    return path


def checkpoint(name: str = "CTCFmotif") -> Path:
    """Path of a ChromaFold checkpoint (:data:`CHECKPOINTS`), downloaded on first use."""
    if name not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint {name!r}; one of {list(CHECKPOINTS)}, or give a path")
    fname, sha = CHECKPOINTS[name]
    url = f"https://github.com/viannegao/ChromaFold/raw/{COMMIT}/checkpoints/{fname}"
    return _fetch(data_dir() / fname, sha, lambda tmp: urllib.request.urlretrieve(url, tmp))


class _ArrayUnpickler(pickle.Unpickler):
    """Unpickles NumPy arrays and SciPy sparse matrices only."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("scipy.sparse.csc", "csc_matrix"), ("scipy.sparse._csc", "csc_matrix"),
        ("scipy.sparse.csr", "csr_matrix"), ("scipy.sparse._csr", "csr_matrix"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"{module}.{name} is not allowed in a CTCF track")
        if module.startswith("numpy") and name == "_reconstruct":
            from numpy._core.multiarray import _reconstruct
            return _reconstruct
        if module.startswith("scipy.sparse"):
            import scipy.sparse as sp
            return getattr(sp, name)
        return getattr(np, name)


def ctcf_track(genome: str) -> dict[str, np.ndarray]:
    """
    ChromaFold's CTCF motif score track of ``genome`` (``"hg38"``, ``"hg19"``
    or ``"mm10"``): per chromosome, float32 scores in 50 bp bins,
    ``length // 50 + 1`` of them.
    """
    if genome not in CTCF_TRACKS:
        raise ValueError(f"no CTCF motif track for {genome!r}; one of {list(CTCF_TRACKS)}")
    npz = data_dir() / f"{genome}_ctcf_motif_score.npz"
    if not npz.exists():
        file_id, sha = CTCF_TRACKS[genome]

        def download(tmp):
            import gdown
            gdown.download(id=file_id, output=str(tmp), quiet=True)

        p = _fetch(data_dir() / f"{genome}_ctcf_motif_score.p", sha, download)
        with open(p, "rb") as f:
            raw = _ArrayUnpickler(io.BufferedReader(f)).load()
        tracks = {c: np.asarray(m.toarray()).ravel().astype(np.float32) for c, m in raw.items()}
        np.savez(npz.with_suffix(".tmp.npz"), **tracks)
        os.replace(npz.with_suffix(".tmp.npz"), npz)
        p.unlink()
    with np.load(npz) as z:
        return {c: z[c] for c in z.files}
