import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="alomancy",
        description="ALomancy — active learning workflow for MLIPs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="Run the AL workflow described by a YAML config",
    )
    run.add_argument("config", type=Path, help="Path to the run's YAML config")

    rep = sub.add_parser(
        "report",
        help="Write the per-loop AL report(s) for an existing results directory",
    )
    rep.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        metavar="PATH",
        help="Path to the results/ directory (default: ./results)",
    )
    which = rep.add_mutually_exclusive_group()
    which.add_argument(
        "--loop", type=int, metavar="N", help="Report for AL loop N (default: latest)"
    )
    which.add_argument(
        "--all", action="store_true", help="Reports for every completed AL loop"
    )
    rep.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="The run's YAML config (default: results/run_config.yaml)",
    )

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

    list_hpc = sub.add_parser(
        "list-hpc",
        help="List configured HPC systems and a summary of their settings",
    )
    list_hpc.add_argument(
        "--check-remote",
        action="store_true",
        help="Also ssh to each host to report its installed alomancy version",
    )

    sub.add_parser(
        "upgrade-hpc",
        help="Upgrade the alomancy package on one or more configured HPC systems",
    )

    nuke = sub.add_parser(
        "nuke",
        help="Delete local ExPyRe job state (job cache, unsynced stage dirs)",
    )
    nuke.add_argument(
        "--expyre-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to the ExPyRe local directory to nuke (default: this "
            "run's resolved local_stage_dir, i.e. wherever expyre.config "
            "resolves to from the current directory -- falls back to "
            "~/.expyre if expyre has no HPC configured yet)"
        ),
    )

    args = parser.parse_args()

    if args.command == "run":
        from alomancy import ALomancy

        ALomancy(args.config).run()
    elif args.command == "report":
        from alomancy.cli.report import completed_loops, write_reports

        if args.all:
            loops = completed_loops(args.results_dir)
        elif args.loop is not None:
            loops = [args.loop]
        else:
            loops = None
        for path in write_reports(
            args.results_dir, loops=loops, config_path=args.config
        ):
            print(path)
    elif args.command == "add-hpc":
        from alomancy.cli.add_hpc import add_hpc_wizard

        add_hpc_wizard()
    elif args.command == "list-hpc":
        from alomancy.cli.list_hpc import list_hpc as list_hpc_table

        print(list_hpc_table(check_remote=args.check_remote))
    elif args.command == "upgrade-hpc":
        from alomancy.cli.upgrade_hpc import upgrade_hpc_wizard

        upgrade_hpc_wizard()
    elif args.command == "results":
        if args.replot:
            from alomancy.cli.replot import replot_results

            replot_results(args.results_dir.resolve(), no_parity=args.no_parity)
        else:
            res.print_help()
    elif args.command == "nuke":
        from alomancy.cli.nuke import nuke_expyre_results, resolve_default_expyre_dir

        expyre_dir = args.expyre_dir or resolve_default_expyre_dir()
        nuke_expyre_results(expyre_dir.resolve())
