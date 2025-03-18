""" data_plot.py (c) Yang Yang, 2024-2025
This file implements a loss plot utility for the algorithm.
"""

import matplotlib
import matplotlib.pyplot as plt
import os
from torch import Tensor
from typing import Dict, Tuple
from _global import info

matplotlib.use("TkAgg")


class LossCurvePlotter:
    def __init__(self, title: str = 'Loss Curve', xlabel: str = 'Epoch', ylabel: str = 'Loss',
                 save_dir=None, constant_dicts: Tuple[Dict[str, float], Dict[str, float]] = None, nu: float = None):
        """
        Initializes the plotter.
        :param title: (str) Title of the plot.
        :param xlabel: (str) Label for the x-axis.
        :param ylabel: (str) Label for the y-axis.
        :param save_dir: (str) Directory to save plots. If None, plots are not saved.
        :param constant_dicts: (beta_dict and rho_dict).
        :param nu: (float) The nu parameter.
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.save_dir = os.path.abspath(save_dir)
        self.epochs = []
        self.losses = []
        self.extra_info = constant_dicts if constant_dicts is not None else ({}, {})
        self.nu = nu

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def update(self, epoch: int, loss: float):
        """
        Updates the data for the plot.
        :param epoch: (int) The current epoch.
        :param loss: (float) The loss value at this epoch.
        """
        if isinstance(loss, Tensor):
            loss = loss.item()
        self.epochs.append(epoch)
        self.losses.append(loss)

    def plot(self, show: bool = True, save_name: str = None):
        """
        Plots the loss curve.
        :param show: (bool) Whether to display the plot.
        :param save_name: (str) Filename to save the plot. If None, plot is not saved.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(right=0.75)  # Leave space on the right for extra info

        # Plot the loss curve
        ax.plot(self.epochs, self.losses, label="Loss", color="blue", marker="o")
        ax.set_title(self.title, fontsize=16)
        ax.set_xlabel(self.xlabel, fontsize=14)
        ax.set_ylabel(self.ylabel, fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(fontsize=12)

        dict1, dict2 = self.extra_info
        info_text1 = "\n".join([f"{key}: {value}" for key, value in dict1.items()])
        info_text2 = "\n".join([f"{key}: {value}" for key, value in dict2.items()])

        right_ax = fig.add_axes((0.8, 0.1, 0.2, 0.8), frame_on=False)  # x, y, width, height
        right_ax.axis("off")  # Hide the axis
        right_ax.text(0, 0.5, (f"Nu: {self.nu}\n\n" if self.nu is not None else str())
                      + f"Beta Values:\n{info_text1}\n\nRho Values:\n{info_text2}",
                      fontsize=12, color="black", verticalalignment="center", horizontalalignment="left")

        # Save the plot if needed
        if save_name and self.save_dir:
            save_path = self.__get_save_path(save_name)
            plt.savefig(save_path, dpi=300)
            info(f"Plot saved to {save_path}")

        if show:
            plt.show()
        plt.close()

    def reset(self):
        """
        Resets the data for the plot.
        """
        self.epochs = []
        self.losses = []

    def __get_save_path(self, loss_filename: str) -> str:
        if not loss_filename:
            loss_filename = 'ADMMLossCurve.png'
        if not '.png' or '.jpg' in loss_filename:
            loss_filename += '.png'
        loss_filename = os.path.join(self.save_dir, loss_filename)
        if os.path.isfile(loss_filename):
            filename, suffix = loss_filename[:-4], loss_filename[-3:]
            i = 1
            loss_filename = filename + f'_{i}.' + suffix
            while os.path.isfile(loss_filename):
                i += 1
                loss_filename = f'{filename}_{i}.{suffix}'
        return loss_filename


if __name__ == "__main__":
    plotter = LossCurvePlotter(title="Training Loss", save_dir="./plots")
    for epoch in range(1, 11):
        loss = 0.5 / epoch
        plotter.update(epoch, loss)
    plotter.plot(save_name="loss_curve.png")

