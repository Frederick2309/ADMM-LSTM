"""
  (c) Yang Yang, 2024-2025
  This file visualizes the predictions, making comparison of different algorithms.
"""

import os
import torch
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from _global import error, log_assert
from demo import init
from typing import Dict
from glob import glob

plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.size': 20,              # base font size
    'axes.titlesize': 16,         # title
    'axes.labelsize': 20,         # x and y labels
    'xtick.labelsize': 18,        # x tick labels
    'ytick.labelsize': 18,        # y tick labels
    'legend.fontsize': 16,        # legend
    'figure.titlesize': 18,       # figure title (if using plt.suptitle)
    'font.sans-serif': 'Heiti TC',  # support Chinese
    'axes.unicode_minus': False,  # support Chinese
    'figure.subplot.left': 0.1,
    'figure.subplot.right': 0.9,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.top': 0.9,
})


color_list = {
    'Ground Truth': 'black',
    'Fast ADMM-LSTM': 'b',
    'ADMM-LSTM-L': 'g',
    'SGD': 'r',
    'Adam': 'c',
    'Adagrad': 'm'
}


def load_models(path: str = 'SAVED_MODELS/*.pt') -> Dict[str, torch.nn.Module]:
    models = {}
    for model_file in glob(path):
        try:
            models.update({os.path.basename(model_file).split('.')[0]: torch.load(model_file, weights_only=False)})
        except Exception as e:
            error(f'Unexpected error while loading {model_file}: {e}.')
    return models


def plot_in_one_plot(time_span: np.ndarray, data: Dict[str, torch.Tensor]) -> None:
    legends = ['Ground Truth']
    length = len(time_span)
    plt.figure(1, figsize=(20, 6))
    plt.plot(time_span, data['Ground Truth'][:length], color=color_list['Ground Truth'])
    for i, (method_name, predictions) in enumerate(data.items()):
        if method_name == 'Ground Truth':
            continue
        log_assert(predictions.shape[1] == 1, f'Only 1-dimensional features are supported.')
        predictions = predictions[:length].squeeze(1).detach().numpy()
        plt.plot(time_span, predictions, color=color_list[method_name])
        legends.append(method_name)
    plt.legend(
        legends, loc='upper left',
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fancybox=True,
    )
    plt.xlabel('Time')
    plt.ylabel('Prediction')
    plt.grid(True)
    plt.xlim([time_span[0], time_span[-1]])
    # plt.ylim([0, 1])
    plt.show()


def plot_with_subplots(time_span: np.ndarray, data: Dict[str, torch.Tensor]) -> None:
    fig, ax = plt.subplots(len(data))
    for i, (method_name, predictions) in enumerate(data.items()):
        log_assert(predictions.shape[1] == 1, f'Only 1-dimensional features are supported.')
        predictions = predictions.squeeze(1).detach().numpy()
        ax[i].plot(time_span, predictions)
        ax[i].set_title(method_name)
        ax[i].set_xlabel('Time')
        ax[i].set_ylabel('Prediction')
        ax[i].grid(True)
    plt.show()


def predict(features: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    return model(features).detach()


def predict_all(models: Dict[str, torch.nn.Module],
                features: torch.Tensor, ground_truths: torch.Tensor) -> Dict[str, torch.Tensor]:
    predictions = {}
    for model_name, model in models.items():
        predictions.update({model_name: model(features)})
    predictions.update({'Ground Truth': ground_truths})
    return predictions


def plot_all(time_span: np.ndarray, data: Dict[str, torch.Tensor], in_subplots: bool = False):
    if in_subplots:
        plot_with_subplots(time_span, data)
    else:
        plot_in_one_plot(time_span, data)


if __name__ == '__main__':
    _, _, ((_, _, test_x, test_y), _, _), _, _ = init('demo')
    model_dict = load_models()
    time_range = np.array(list(range(test_x.size(0))))
    model_predictions = predict_all(model_dict, test_x, test_y)
    plot_all(time_range, model_predictions)
