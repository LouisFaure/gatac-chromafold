"""The GPU windowing, co-accessibility and stripe assembly of ``predict`` against ChromaFold's code."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from gatac_chromafold.prediction import ContactPrediction, _jaccard, _windows  # noqa: E402


def cpu_jaccard_vstripe(x):
    """chromafold/utils.py"""
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
    """chromafold/utils.py"""
    c = cpu_jaccard_vstripe(x.permute(1, 0))
    c[c != c] = 0
    return c


def test_jaccard_is_chromafolds():
    """Batched Jaccard of the centre tiles = cpu_batch_corcoeff_vstripe, window by window."""
    rng = np.random.default_rng(0)
    z = rng.poisson(0.3, size=(3, 8020, 40)).astype(np.float32)
    z[1, :500] = 0           # zero-padded start
    got = _jaccard(torch.as_tensor(z)).numpy()
    for b in range(3):
        want = cpu_batch_corcoeff_vstripe(torch.as_tensor(z[b].T)).numpy()   # metacells x tiles
        np.testing.assert_allclose(got[b], want, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("start", [-2_000_000, -10_000, 0, 30_000])
def test_windows_pad_like_chromafold(start):
    """A window starting before the chromosome is its in-chromosome part, zero-padded on the left."""
    x = torch.arange(1, 200_001, dtype=torch.float32)
    size, res = 80_200, 50
    a = int(start / res)
    want = x[max(0, a): a + size]
    want = torch.cat([torch.zeros(size - len(want)), want])
    got = _windows(x, np.array([start]), res, size)[0]
    assert torch.equal(got, want)
    z = torch.arange(1, 3 * 30_000 + 1, dtype=torch.float32).reshape(-1, 3)
    a = int(start / 500)
    want = z[max(0, a): a + 8020]
    want = torch.cat([torch.zeros(8020 - len(want), 3), want])
    assert torch.equal(_windows(z, np.array([start]), 500, 8020)[0], want)


def test_matrix_places_and_averages_stripes():
    """Stripe of centre c: (c-200+j, c), then (c, c+1+j); pairs predicted twice are averaged."""
    centers = np.arange(0, 600)
    y = np.empty((len(centers), 400), np.float32)
    y[:, :200] = 1.0     # every vertical prediction
    y[:, 200:] = 3.0     # every horizontal prediction
    pred = ContactPrediction(stripes={"chrA": y}, centers={"chrA": centers}, chrom_sizes={"chrA": 6_000_000})
    m = pred.matrix("chrA").toarray()
    assert m.shape == (600, 600)
    assert np.all(np.tril(m) == 0)
    d = np.arange(600)
    for k in (1, 50, 200):
        diag = np.diag(m, k)
        # (i, i+k): horizontally from centre i, vertically from centre i+k
        assert np.all(diag == 2.0)
    assert np.all(np.diag(m, 201) == 0)
    # distinct values land at their pairs
    y2 = np.zeros_like(y)
    c = 300
    y2[c, :200] = np.arange(200)            # (c-200+j, c)
    y2[c, 200:] = 1000 + np.arange(200)     # (c, c+1+j)
    pred.stripes["chrA"] = y2
    m = pred.matrix("chrA")
    assert m[c - 200 + 7, c] == pytest.approx(7 / 2)
    assert m[c, c + 1 + 7] == pytest.approx((1000 + 7) / 2)
    df = pred.to_dataframe("chrA")
    assert set(df.columns) == {"start1", "start2", "score"}
    del d
