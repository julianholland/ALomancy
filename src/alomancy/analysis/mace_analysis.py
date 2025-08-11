from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def mace_recover_train_txt_final_results(
    mlip_committee_job_dict: dict,
) -> pd.DataFrame:
    """
    Recover final results from train.txt files in MACE AL loop directories.
    """

    al_loop_dirs = list(Path.glob(Path("results"), "al_loop_*"))
    all_avg_results = []
    for al_loop_dir in al_loop_dirs:
        results_files = list(
            Path.glob(
                Path(al_loop_dir, mlip_committee_job_dict["name"]),
                "fit_*/results/*train.txt",
            )
        )
        results = []
        for results_file in results_files:
            with open(results_file) as file:
                data_line = file.readlines()[-1]
                result = dict(eval(data_line))
                results.append(result)

        avg_result = {
            key: np.mean([np.float32(result[key]) for result in results])
            for key in results[0]
            if key not in ["mode", "epoch", "head"]
        }
        all_avg_results.append(avg_result)
    return pd.DataFrame(all_avg_results)


def mace_al_loop_average_error(all_avg_results, plot=False):
    df = pd.DataFrame(all_avg_results)
    if plot:
        plot_object = Plot(
            data=df[["mae_e", "mae_f"]],
            title="MACE AL Loop MAE",
            xlabel="AL Loop Iteration",
            ylabel="Mean Absolute Error",
            directory=str(Path("results")),
        )
        plot_object.create()
        plot_object.save()


if __name__ == "__main__":
    df = mace_al_loop_average_error(plot=True)
