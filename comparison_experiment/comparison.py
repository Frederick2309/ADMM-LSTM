""" comparison.py (c) Yang Yang, 2024-2025
  In this file we compare the effectiveness of the optimization methods.
"""
import re
import os.path
import numpy as np
import scipy.io as sio
from blocks.lstm import LSTM
from demo import admm_demo, init
from grad_based.grad_based import sgd_demo, adam_demo, adagrad_demo
from typing import List, Dict
from matplotlib import pyplot as plt
from _global import device, color_list, global_dict, info, error, log_assert

plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.size': 20,              # base font size
    'axes.titlesize': 16,         # title
    'axes.labelsize': 20,         # x and y labels
    'xtick.labelsize': 18,        # x tick labels
    'ytick.labelsize': 18,        # y tick labels
    'legend.fontsize': 20,        # legend
    'figure.titlesize': 18,       # figure title (if using plt.suptitle)
    'font.sans-serif': 'Heiti TC',  # support Chinese
    'axes.unicode_minus': False,  # support Chinese
    'figure.subplot.left': 0.1,
    'figure.subplot.right': 0.9,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.top': 0.9,
})


with_admm_s = False
with_admm_l = True


# def plot_data(plot_epochs: List[int], demo_return_values: List[Dict[str, List[float] or str]],
#               with_initial: bool) -> None:
#     legend_list = [method['name'] for method in demo_return_values]
#     if not with_initial:
#         plot_epochs = plot_epochs[1:]
#     for i in range(len(demo_return_values)):
#         return_values = demo_return_values[i]
#         name, train_loss, val_loss = return_values['name'], return_values['train_loss'], return_values['val_loss']
#         plt.subplot(2, 1, 1)  # plot training losses
#         if not with_initial:
#             train_loss, val_loss = train_loss[1:], val_loss[1:]
#         try:
#             plt.plot(plot_epochs, train_loss, color=color_list[i], linestyle='-', marker='o', label='train loss')
#             plt.subplot(2, 1, 2)  # plot validation losses
#             plt.plot(plot_epochs, val_loss, color=color_list[i], linestyle='-', marker='o', label='val loss')
#         except ValueError as e:
#             error(f'ValueError when handling {name}: {e}.\n'
#                   f'Shapes: x - {len(train_loss)}, y - {len(val_loss)}, epochs - {len(plot_epochs)} '
#                   f'(With the 0th epoch: {with_initial})')
#
#     def configure_subplot(subplot_order: int, subplot_title: str):
#         plt.subplot(2, 1, subplot_order)
#         plt.title(subplot_title)
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend(legend_list, loc='upper right', fontsize=14)
#         plt.grid(True)
#         plt.yscale('symlog', linthresh=0.01)
#         plt.xlim([0 if with_initial else 1, num_epochs])
#
#     configure_subplot(1, 'Training Loss Curves')
#     configure_subplot(2, 'Validation Loss Curves')
#     plt.subplots_adjust(hspace=0.3)
#     plt.show()

def plot_data(plot_epochs: List[int], demo_return_values: List[Dict[str, List[float] or str]],
              with_initial: bool) -> None:
    legend_list = [method['name'] for method in demo_return_values]
    if not with_initial:
        plot_epochs = plot_epochs[1:]

    # plot training loss
    plt.figure(1, figsize=(20, 5))
    for i in range(len(demo_return_values)):
        return_values = demo_return_values[i]
        name, train_loss = return_values['name'], return_values['train_loss']
        if not with_initial:
            train_loss = train_loss[1:]
        try:
            plt.plot(plot_epochs, train_loss, color=color_list[i], linestyle='-', marker='o', label='train loss')
        except ValueError as e:
            error(f'ValueError when handling {name}: {e}.\n'
                  f'Shapes: x - {len(train_loss)}, epochs - {len(plot_epochs)} '
                  f'(With the 0th epoch: {with_initial})')

    # plt.title('Training Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(legend_list, loc='upper right',
               frameon=True,
               edgecolor="black",
               facecolor="white",
               framealpha=1.0,
               fancybox=True,
               )
    plt.grid(True)
    plt.yscale('symlog', linthresh=0.01)
    plt.xlim([0 if with_initial else 1, num_epochs])
    plt.show()

    # plot validation loss
    plt.figure(2, figsize=(20, 5))
    for i in range(len(demo_return_values)):
        return_values = demo_return_values[i]
        name, val_loss = return_values['name'], return_values['val_loss']
        if not with_initial:
            val_loss = val_loss[1:]
        try:
            plt.plot(plot_epochs, val_loss, color=color_list[i], linestyle='-', marker='o', label='val loss')
        except ValueError as e:
            error(f'ValueError when handling {name}: {e}.\n'
                  f'Shapes: x - {len(val_loss)}, epochs - {len(plot_epochs)} '
                  f'(With the 0th epoch: {with_initial})')

    # plt.title('Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(legend_list, loc='upper right',
               frameon=True,
               edgecolor="black",
               facecolor="white",
               framealpha=1.0,
               fancybox=True,
               )
    plt.grid(True)
    plt.yscale('symlog', linthresh=0.01)
    plt.xlim([0 if with_initial else 1, num_epochs])
    plt.show()


def generate_model(with_grad: bool, input_size: int, hidden_size: int, output_size: int) -> LSTM:
    return LSTM(input_size, hidden_size, output_size, with_grad).to(device)


if __name__ == '__main__':

    (
        num_epochs, hidden_size,
        ((train_x, train_y, test_x, test_y), example_dict, dataset_name),
        _, save, g_args
    ) = init('comp')

    input_size, output_size = train_x.size(2), train_y.size(1)

    if with_admm_s:
        try:
            from comparison_experiment.admm_s.results import admm_s_loss
        except ImportError:
            error('Cannot import results.py. Generate a result first.')
        log_assert(len(admm_s_loss['train_loss']) >= num_epochs + 1,
                   f'Got ADMM-LSTM-S train epoch: {len(admm_s_loss["train_loss"])}. Expected: > {num_epochs + 1}')
        log_assert(len(admm_s_loss['val_loss']) >= num_epochs + 1,
                   f'Got ADMM-LSTM-S val epoch: {len(admm_s_loss["val_loss"])}. Expected: > {num_epochs + 1}')
        admm_s_loss.update({
            'train_loss': admm_s_loss['train_loss'][:(num_epochs + 1)],
            'val_loss': admm_s_loss['val_loss'][:(num_epochs + 1)]
        })
    else:
        admm_s_loss = None

    loss_list = [admm_demo(
        num_epochs, generate_model(False, input_size, hidden_size, output_size), train_x, train_y, test_x, test_y,
        example_dict, False, save=save
    )] if not g_args['skip_fast'] else []

    epochs = list(range(num_epochs + 1))

    if with_admm_l:
        from admm_l.main import admm_l_demo
        loss_list.append(admm_l_demo(
            num_epochs, hidden_size, train_x, train_y, test_x, test_y, save=save
        ))

    if admm_s_loss:
        loss_list.append(admm_s_loss)

    loss_list += [
        sgd_demo(
            num_epochs, generate_model(True, input_size, hidden_size, output_size), train_x, train_y, test_x, test_y,
            lr=g_args['sgd'], save=save),
        adam_demo(
            num_epochs, generate_model(True, input_size, hidden_size, output_size), train_x, train_y, test_x, test_y,
            lr=g_args['adam'], save=save
        ),
        adagrad_demo(
            num_epochs, generate_model(True, input_size, hidden_size, output_size), train_x, train_y, test_x, test_y,
            lr=g_args['adagrad'], save=save
        )
    ]

    plot_data(epochs, loss_list, with_initial=False)

    if g_args['record_matlab_data']:
        dir_name = 'MATLAB_VAL_DATA'
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        save_path = os.path.join(
            dir_name,
            'MATLAB_Val_' + os.path.basename(global_dict['logger_filename']).split('.')[0] + '.mat'
        )
        sio.savemat(save_path, {
            re.sub('[ -]', '', method['name']): np.array(method['val_loss']) for method in loss_list
        })
        info(f'Validation loss has been saved to {save_path}. You can open it with MATLAB.')
