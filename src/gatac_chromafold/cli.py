"""``gatac chromafold``: ChromaFold's inputs from a fragment file."""
from __future__ import annotations

import logging
import sys


def register(subparsers) -> None:
    p = subparsers.add_parser("chromafold", help="ChromaFold contact-map prediction")
    sub = p.add_subparsers(dest="chromafold_command", required=True)

    prep = sub.add_parser(
        "prepare",
        help="Write ChromaFold's scATAC inputs of one cell type, in its file layout",
        description="Filter cells as ChromaFold does (or take them from an .h5ad), compute the "
                    "iterative LSI, metacells, 500 bp insertion tiles and the 50 bp track, and "
                    "write them under OUTPUT/atac/ for ChromaFold's inference.py.",
    )
    prep.add_argument("input", help="Fragments .parquet file of the cell type")
    prep.add_argument("-o", "--output", required=True, help="ChromaFold data directory")
    prep.add_argument("-c", "--cell-type", required=True, help="Cell type name, ChromaFold's -ct")
    prep.add_argument("-g", "--genome", default="hg38", help="hg38, hg19, mm10 or mm39 (default: hg38)")
    group = prep.add_mutually_exclusive_group(required=True)
    group.add_argument("--metrics", help="Metrics CSV of gatac metrics, to filter cells "
                                         "(TSS enrichment >= 4, >= 1,000 fragments)")
    group.add_argument("--cells", help=".h5ad whose cells to use")
    prep.add_argument("--use-rep", help="With --cells, the obsm embedding for the metacells "
                                        "(default: compute ChromaFold's iterative LSI)")
    prep.set_defaults(func=_prepare)


def _prepare(args) -> None:
    import anndata as ad

    from .inputs import prepare

    try:
        cells = ad.read_h5ad(args.cells) if args.cells else None
        inputs = prepare(args.input, args.genome, cells=cells, use_rep=args.use_rep,
                         metrics=args.metrics)
        paths = inputs.write(args.output, args.cell_type)
    except (ValueError, KeyError) as e:
        logging.error(str(e))
        sys.exit(1)
    logging.info(f"{inputs}; wrote {', '.join(str(p) for p in paths.values())}")
