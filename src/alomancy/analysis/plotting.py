import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import MaxNLocator

from alomancy.analysis.colors import PALETTE, add_logo_watermark, setup_alomancy_style

logger = logging.getLogger(__name__)


class Plot:
    def __init__(
        self,
        data: pl.DataFrame | dict | list,
        title: str,
        xlabel: str,
        ylabel: str,
        directory: str = ".",
        log_scale_y: bool = False,
        error_bars: bool = False,
    ):
        """
        data: pl.DataFrame or dict-like, where each column/field is a series
        to plot against its row number
        """
        self.data = data
        self.error_bars = error_bars
        self.log_scale_y = log_scale_y
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.directory = directory
        self.filename = str(
            Path(self.directory, f"{title.replace(' ', '_').lower()}_plot.png")
        )
        self.fig = None
        self.ax = None

    def find_data(self, data_name):
        if isinstance(self.data, pl.DataFrame):
            return self.data[data_name]

    def create(self):
        logger.debug(
            "Creating plot with data columns: %s",
            self.data.columns if hasattr(self.data, "columns") else self.data,
        )
        import matplotlib

        setup_alomancy_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_prop_cycle(matplotlib.cycler(color=PALETTE))
        if isinstance(self.data, pl.DataFrame):
            for col in self.data.columns:
                ax.plot(
                    range(self.data.height),
                    self.data[col].to_numpy(),
                    marker="o",
                    linestyle="-",
                    label=col,
                )
        elif isinstance(self.data, dict):
            for key, values in self.data.items():
                ax.plot(
                    range(len(values)), values, marker="o", linestyle="-", label=key
                )
        else:
            ax.plot(self.data, marker="o", linestyle="-")
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)
        if self.log_scale_y:
            ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        add_logo_watermark(fig)
        self.fig, self.ax = fig, ax

    def save(self):
        logger.debug("Saving plot to %s", self.filename)
        plt.savefig(self.filename)
        if self.fig is not None:
            plt.close(self.fig)

    def clear(self):
        logger.debug("Clearing plot data")
        if isinstance(self.data, pl.DataFrame):
            self.data = self.data.clear()
        elif isinstance(self.data, dict):
            self.data = {k: [] for k in self.data}
        else:
            self.data = []

    def update(self, new_data):
        logger.debug("Updating plot with new data")
        if isinstance(self.data, pl.DataFrame) and isinstance(new_data, pl.DataFrame):
            self.data = pl.concat([self.data, new_data], how="diagonal_relaxed")
        elif isinstance(self.data, dict) and isinstance(new_data, dict):
            for k, v in new_data.items():
                self.data.setdefault(k, []).extend(v)
        elif isinstance(self.data, list):
            self.data.extend(new_data)
        logger.debug("Updated data: %s", self.data)


def mae_al_loop_plot(
    all_avg_results: pl.DataFrame,
    mlip_committee_job_dict: dict,
    directory: Path = Path("results"),
) -> None:
    import matplotlib

    setup_alomancy_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_prop_cycle(matplotlib.cycler(color=PALETTE))

    # One row per loop; "al_loop" holds the loop number, so a skipped loop
    # shows as a gap rather than shifting later loops left.
    x = (
        all_avg_results["al_loop"].to_list()
        if "al_loop" in all_avg_results.columns
        else list(range(all_avg_results.height))
    )
    name = mlip_committee_job_dict["name"]

    for col, label in (
        ("mae_e_per_atom", "Best model energy MAE (eV/atom)"),
        ("mae_f", "Best model force MAE (eV/Å)"),
    ):
        if col not in all_avg_results.columns:
            continue
        y = all_avg_results[col].to_numpy()
        std_col = f"{col}_std_dev"
        if std_col in all_avg_results.columns:
            yerr = all_avg_results[std_col].to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linestyle="-",
                capsize=4,
                capthick=1.2,
                linewidth=1.5,
                label=label,
            )
        else:
            ax.plot(x, y, marker="o", linestyle="-", linewidth=1.5, label=label)

    ax.set_xlabel("AL Loop Iteration")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(f"{name} AL Loop MAE (best committee member per loop)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    add_logo_watermark(fig)
    filename = Path(directory) / f"{name}_al_loop_mae_plot.png"
    plt.savefig(str(filename))
    plt.close(fig)
    logger.debug("Saved AL loop MAE plot to %s", filename)


if __name__ == "__main__":
    # Example usage
    example_data = pl.DataFrame(
        {
            "al_loop": [0, 1, 2],
            "mae_e_per_atom": [0.1, 0.2, 0.15],
            "mae_f": [0.05, 0.07, 0.06],
        }
    )
    mae_al_loop_plot(
        all_avg_results=example_data,
        mlip_committee_job_dict={"name": "Example Committee"},
        directory=Path("."),
    )
    plt.show()
