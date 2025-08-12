from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class Plot:
    def __init__(
        self,
        data: pd.DataFrame,
        title: str,
        xlabel: str,
        ylabel: str,
        directory: str = ".",
    ):
        """
        data: pd.DataFrame or dict-like, where each column/field is a series to plot
        """
        self.data = data
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.directory = directory
        self.filename = str(
            Path(self.directory, f"{title.replace(' ', '_').lower()}_plot.png")
        )

    def find_data(self, data_name):
        if isinstance(self.data, pd.DataFrame):
            return self.data[data_name]

    def create(self):
        print(
            "Creating plot with data columns:",
            self.data.columns if hasattr(self.data, "columns") else self.data,
        )
        plt.figure(figsize=(10, 6))
        if isinstance(self.data, pd.DataFrame):
            for col in self.data.columns:
                plt.plot(
                    self.data.index,
                    self.data[col],
                    marker="o",
                    linestyle="-",
                    label=col,
                )
        elif isinstance(self.data, dict):
            for key, values in self.data.items():
                plt.plot(
                    range(len(values)), values, marker="o", linestyle="-", label=key
                )
        else:
            plt.plot(self.data, marker="o", linestyle="-", color="b")
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.grid(True)
        plt.legend()

    def show(self):
        plt.show()

    def save(self):
        print(f"Saving plot to {self.filename}")
        plt.savefig(self.filename)

    def clear(self):
        print("Clearing plot data")
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data.iloc[0:0]
        elif isinstance(self.data, dict):
            self.data = {k: [] for k in self.data}
        else:
            self.data = []

    def update(self, new_data):
        print("Updating plot with new data")
        if isinstance(self.data, pd.DataFrame) and isinstance(new_data, pd.DataFrame):
            self.data = pd.concat([self.data, new_data], ignore_index=True)
        elif isinstance(self.data, dict) and isinstance(new_data, dict):
            for k, v in new_data.items():
                self.data.setdefault(k, []).extend(v)
        elif isinstance(self.data, list):
            self.data.extend(new_data)
        print("Updated data:", self.data)


def mae_al_loop_plot(
    all_avg_results, mlip_committee_job_dict, directory=Path("results")
):
    plot_object = Plot(
        data=all_avg_results[["mae_e", "mae_f"]],
        title=f"{mlip_committee_job_dict['name']} AL Loop MAE",
        xlabel="AL Loop Iteration",
        ylabel="Mean Absolute Error",
        directory=directory,
    )
    plot_object.create()
    plot_object.save()


if __name__ == "__main__":
    # Example usage
    example_data = pd.DataFrame(
        {"mae_e": [0.1, 0.2, 0.15], "mae_f": [0.05, 0.07, 0.06]}
    )
    mae_al_loop_plot(
        all_avg_results=example_data,
        mlip_committee_job_dict={"name": "Example Committee"},
        directory=Path("."),
    )
    plt.show()
