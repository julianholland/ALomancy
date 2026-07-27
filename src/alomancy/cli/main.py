import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="alomancy",
        description="ALomancy — active learning workflow for MLIPs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    res = sub.add_parser("results", help="Inspect and post-process workflow results")
    res.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate all plots from an existing results directory",
    )
    res.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        metavar="PATH",
        help="Path to the results/ directory (default: ./results)",
    )
    res.add_argument(
        "--no-parity",
        action="store_true",
        help="Skip parity plots (avoids loading MACE models — much faster)",
    )

    sub.add_parser(
        "add-hpc",
        help="Interactive wizard to add an HPC system to ALomancy",
    )

    args = parser.parse_args()

    if args.command == "add-hpc":
        from alomancy.cli.add_hpc import add_hpc_wizard

        add_hpc_wizard()
    elif args.command == "results":
        if args.replot:
            from alomancy.cli.replot import replot_results

            replot_results(args.results_dir.resolve(), no_parity=args.no_parity)
        else:
            res.print_help()
