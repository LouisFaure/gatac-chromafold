# gatac-chromafold

[ChromaFold](https://github.com/viannegao/ChromaFold) (Gao et al., *Nat Commun* 2024)
predicts Hi-C contact maps from scATAC-seq. This GATAC extension builds its
inputs from a GATAC fragment file on the GPU, without ArchR or R:

| ChromaFold input | ChromaFold's preprocessing | Here |
|---|---|---|
| Cells | ArchR `createArrowFiles`, TSS ≥ 4, ≥ 1,000 fragments | `gatac.pp.compute_metrics` + the same thresholds, or your own cells |
| Embedding | ArchR `addIterativeLSI` (2 iterations, 25k features, 30 dims) | `gatac.dr.iterative_lsi`, same parameters, or any `obsm` embedding |
| Metacells | Python, dense cells × cells kNN | the same metacells given the same embedding, the kNN on the GPU |
| 500 bp insertion tiles, summed per metacell | pandas | streamed from the fragment file on the GPU (`gatac.io.FragmentFile`) |
| 50 bp track of all cells | pandas | the same, in 50 bp bins |

```bash
pip install "gatac-chromafold[rapids-cu12]"   # or rapids-cu13, matching nvidia-smi's CUDA version
```

It needs GATAC 0.2.0 or later, and uses only its public API.

```python
import gatac as ga

metrics = ga.pp.compute_metrics("tcell.parquet", "gencode.gff3.gz")
inputs = ga.chromafold.prepare("tcell.parquet", "hg38", metrics=metrics)
pred = ga.chromafold.predict(inputs, genome="hg38", chroms=["chr19"])
pred.matrix("chr19")          # 10 kb contact map (upper triangle, up to 2 Mb), sparse
inputs.write("chromafold_data", "tcell")   # or ChromaFold's file layout, for its own scripts
```

```bash
gatac chromafold prepare tcell.parquet -o chromafold_data -c tcell --metrics tcell_metrics.csv
gatac chromafold predict chromafold_data -c tcell -o predictions --chroms chr19
```

`predict` runs the published ChromaFold model (`_model.py`, MIT, copied from
ChromaFold) on the GPU, with ChromaFold's windowing, normalisation and Jaccard
co-accessibility computed in batches instead of its per-window CPU
dataloader. The checkpoint (GitHub) and ChromaFold's CTCF motif track
(Google Drive; hg38, hg19, mm10) are downloaded on first use into
`$GATAC_CHROMAFOLD_DATA` (default `~/.cache/gatac-chromafold`) and checked
against their sha256; the track pickle is read with an unpickler that only
builds NumPy and SciPy arrays. `predict` also reads a ChromaFold data
directory (`data_path=`, `cell_type=`), e.g. one made by ChromaFold's own
preprocessing.

Insertions are counted as GATAC and SnapATAC2 place them, both ends of each
fragment (a fragment with both ends in a tile counts twice, as in ChromaFold),
the second one at `end - 1` where ChromaFold's code puts it at `end`; the 0.1 % of insertions
(at 500 bp) of fragments ending on a bin boundary land one bin earlier.
Otherwise the inputs are ChromaFold's own: see `tests/reproducibility`.

## Tests

```bash
uv sync --extra rapids-cu12
uv run --extra rapids-cu12 pytest     # unit tests, synthetic data, seconds
uv run --extra rapids-cu12 pytest -m reproducibility tests/reproducibility
# vs ChromaFold's code on PBMC 5k: inputs and chr19 predictions (downloads ~1.3 GB)
```

Pass the extras to `uv run` too: without them it syncs the environment
without RAPIDS's pins, e.g. pandas 3, which cuDF does not support.
