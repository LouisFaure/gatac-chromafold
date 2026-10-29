# gatac-chromafold

[ChromaFold](https://github.com/viannegao/ChromaFold) (Gao et al., *Nat Commun* 2024)
predicts Hi-C contact maps from scATAC-seq. This GATAC extension builds its
inputs from a GATAC fragment file on the GPU, without ArchR or R:

| ChromaFold input | ChromaFold's preprocessing | Here |
|---|---|---|
| Cells | ArchR `createArrowFiles`, TSS ≥ 4, ≥ 1,000 fragments | `gatac.pp.compute_metrics` + the same thresholds, or your own cells |
| Embedding | ArchR `addIterativeLSI` (2 iterations, 25k features, 30 dims) | `gatac.dr.iterative_lsi`, same parameters, or any `obsm` embedding |
| Metacells | Python, dense cells × cells kNN | `gatac.pp.metacells`: the same metacells given the same embedding |
| 500 bp insertion tiles, summed per metacell | pandas | `gatac.pp.make_tile_matrix(count_strategy="insertion")`, `gatac.pp.aggregate` |
| 50 bp track of all cells | pandas | `gatac.pp.insertion_tracks` |

```python
import gatac as ga

metrics = ga.pp.compute_metrics("tcell.parquet", "gencode.gff3.gz")
inputs = ga.chromafold.prepare("tcell.parquet", "hg38", metrics=metrics)
inputs.write("chromafold_data", "tcell")   # ChromaFold's layout, for its inference.py
```

```bash
gatac chromafold prepare tcell.parquet -o chromafold_data -c tcell --metrics tcell_metrics.csv
```

Insertions are counted as GATAC counts them, the second one of a fragment at
`end - 1` where ChromaFold's code puts it at `end`; the 0.1 % of insertions
(at 500 bp) of fragments ending on a bin boundary land one bin earlier.
Otherwise the inputs are ChromaFold's own: see `tests/reproducibility`.

Not yet here: the CTCF motif track (download ChromaFold's precomputed one) and
running the model.

## Tests

```bash
uv run pytest                                      # unit tests, synthetic data, seconds
uv run pytest -m reproducibility tests/reproducibility   # vs ChromaFold's code on PBMC 5k (downloads ~1 GB)
```
