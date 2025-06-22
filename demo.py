""" demo.py (c) Yang Yang, 2024-2025
This file implements a training demo of ADMM. Usage: python demo.py.

The training process follows the following procedure:
  1. Defining constants such as penalty and normalization factors.
  2. Defining stopping criterion.
  3. While not stopping criterion, do
    - One-step update.
  4. Post-training analysis (plotting loss-epoch curve and more).

"""

__version__ = 'pre0.4'

import time
import argparse
import numpy as np
import torch
import os
from blocks.lstm import LSTM
from typing import Tuple, Dict, List, NoReturn, Any
from parameters import default_epoch
from admm import ADMMBasedOptimizer as ADMMBasedOptimizer, example_parameter_dictionary
# from admm_grad_lstm import ADMMGradLSTM as ADMMBasedOptimizer
from dataset import supported_datasets
from torch import nn
from data_plot import LossCurvePlotter
import scipy.io as sio
import _global
from _global import info, log_assert, error, warning, global_dict

device = _global.device

torch.manual_seed(0)


def generate_parser() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', required=False, default='GoogleStock', type=str,
                        help=f'Supported datasets: {supported_datasets}')
    parser.add_argument('--epoch', '-e', required=False, default=default_epoch, type=int,
                        help='Number of epochs')
    parser.add_argument('--num_train', '-nt', required=False, default='all', type=str,
                        help='Number of training samples')
    parser.add_argument('--num_val', '-nv', required=False, default='all', type=str,
                        help='Number of validation samples')
    parser.add_argument('--hidden', required=False, default=10, type=int,
                        help='Number of hidden neurons in the LSTM')
    parser.add_argument('--version', '-v', action='version',
                        version=f'%(prog)s {__version__} @Yang Yang in 2024',
                        help='Display version information')
    parser.add_argument('--seed', '-s', required=False, default=-1, type=int, help='Specify any integer '
                                                                                   '(-1 for random; leave unspecified '
                                                                                   'to use default value)')
    parser.add_argument('--yes', '-y', required=False, action='store_true',
                        help='Skip checking')
    parser.add_argument('--cpu', required=False, action='store_true', help='Use CPU when GPU is available')
    parser.add_argument('--comp_sgd', required=False, default=1.5, type=float,
                        help='Learning rate of SGD in COMP mode')
    parser.add_argument('--comp_adam', required=False, default=.2, type=float,
                        help='Learning rate of Adam in COMP mode')
    parser.add_argument('--comp_adagrad', required=False, default=1, type=float,
                        help='Learning rate of Adagrad in COMP mode')
    parser.add_argument('--comp_skip_fast', required=False, action='store_true', default=False,
                        help='Skip Fast ADMM-LSTM')
    parser.add_argument('--comp_record_matlab_data', required=False, action='store_true',
                        default=False, help='Record MATLAB data in COMP mode')
    parser.add_argument('--save', required=False, action='store_true', default=False,
                        help='Save model file')
    return parser


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def start(self):
        if not self.running:
            self.start_time = time.time()
            self.running = True
        else:
            warning("Timer is already running.")

    def stop(self):  # return milliseconds
        if self.running:
            elapsed_this_epoch = time.time() - self.start_time
            self.elapsed_time += elapsed_this_epoch
            self.start_time = None
            self.running = False
            return elapsed_this_epoch * 1e3
        else:
            warning("Timer is not running.")
            return self.elapsed_time * 1e3

    def pause(self):
        if self.running:
            self.elapsed_time += time.time() - self.start_time
            self.start_time = None
            self.running = False
        else:
            warning("Timer is not running.")

    def resume(self):
        if not self.running:
            self.start_time = time.time()
            self.running = True
        else:
            warning("Timer is already running.")

    def reset(self):
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def get_elapsed_time(self):  # return seconds
        if self.running:
            return self.elapsed_time + (time.time() - self.start_time)
        return self.elapsed_time


def parse_num_samples(num_samples: str) -> int or None:
    if num_samples == 'all' or num_samples == '\'all\'':
        return None
    try:
        num_samples = int(num_samples)
        log_assert(num_samples > 0, 'The number of samples must be a positive integer or \'all\'.')
        return num_samples
    except ValueError:
        error('Usage:\n  \'python demo.py --num_train all\' to use the whole training set.\n'
              '  \'python demo.py --num_train <(int) number_of_samples>\' to use part of it.\n'
              '  \'python demo.py --num_val all\' to use the whole validation set.\n'
              '  \'python demo.py --num_val <(int) number_of_samples>\' to use part of it.')


def init_google_stock(num_train: int or None,
                      num_val: int or None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from dataset import GoogleStockDataset
    train_x, train_y, val_x, val_y = GoogleStockDataset().data()
    if isinstance(num_train, int) and num_train < len(train_x):
        train_x = train_x[:num_train]
        train_y = train_y[:num_train]
    if isinstance(num_val, int) and num_val < len(val_x):
        val_x = val_x[:num_val]
        val_y = val_y[:num_val]
    return train_x, train_y, val_x, val_y


def init_GEFCOM2012_load_history(num_train: int or None,
                                 num_val: int or None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from dataset import GEFCom2012
    train_samples = GEFCom2012(
        path='datasets/GEFCOM2012_Data',
        load_object='Load_history/1~20'
    ).sample_dict['Load_history']
    val_samples = GEFCom2012(
        path='datasets/GEFCOM2012_Data',
        load_object='Load_history/21~30'
    ).sample_dict['Load_history']
    train_x, train_y, val_x, val_y = list(), list(), list(), list()
    for i, (x, y) in enumerate(train_samples):
        train_x.append(x)
        train_y.append(y)
        if num_train is not None and i == num_train:
            break
    for i, (x, y) in enumerate(val_samples):
        val_x.append(x)
        val_y.append(y)
        if num_val is not None and i == num_val:
            break
    return torch.stack(train_x), torch.stack(train_y), torch.stack(val_x), torch.stack(val_y)


def clip_samples(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor,
                 num_train: int or None, num_val: int or None
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(num_train, int) and num_train < len(train_x):
        num_train = min(num_train, len(train_x), len(train_y))
        train_x = train_x[:num_train]
        train_y = train_y[:num_train]
    if isinstance(num_val, int) and num_val < len(val_x):
        num_val = min(num_val, len(val_x), len(val_y))
        val_x = val_x[:num_val]
        val_y = val_y[:num_val]
    return train_x, train_y, val_x, val_y


def init_yahoo_finance(num_train: int or None,
                       num_val: int or None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from dataset import YahooFinance
    train_x, train_y, val_x, val_y = YahooFinance().closed_data()
    return clip_samples(train_x, train_y, val_x, val_y, num_train, num_val)


def init_mnist(num_train: int or None,
               num_val: int or None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from dataset import MNISTDataset
    train_x, train_y, val_x, val_y = MNISTDataset().data()
    return clip_samples(train_x, train_y, val_x, val_y, num_train, num_val)


def init_ucf101(num_train: int or None,
                num_val: int or None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from dataset import UCF101
    train_x, train_y, val_x, val_y = UCF101().data()
    return clip_samples(train_x, train_y, val_x, val_y, num_train, num_val)


def init_har(num_train: int or None,
             num_val: int or None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from dataset import HAR
    train_x, train_y, val_x, val_y = HAR().data()
    return clip_samples(train_x, train_y, val_x, val_y, num_train, num_val)


def init_ptb(num_train: int or None,
             num_val: int or None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from dataset import PTB
    train_x, train_y, val_x, val_y = PTB(verbose=False).data(sample_length=1000)
    return clip_samples(train_x, train_y, val_x, val_y, num_train, num_val)


def init_dna1(num_train: int or None,
              num_val: int or None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from dataset import DNA1
    train_x, train_y, val_x, val_y = DNA1().data()
    return clip_samples(train_x, train_y, val_x, val_y, num_train, num_val)


def init_sms_spam(num_train: int or None,
                  num_val: int or None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from dataset import SMSSpamRecognition
    train_x, train_y, val_x, val_y = SMSSpamRecognition().data()
    return clip_samples(train_x, train_y, val_x, val_y, num_train, num_val)


def parse_dataset(dataset_name: str, num_train: int, num_val: int) \
        -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                 Dict[str, Dict[str, float]], str] or NoReturn:
    log_assert(dataset_name in supported_datasets, f'Dataset {dataset_name} is not supported.')
    if dataset_name == 'GoogleStock':
        return init_google_stock(num_train, num_val), example_parameter_dictionary['GoogleStock'], "Google Stock"
    elif dataset_name == 'GEFCOM2012':
        return (init_GEFCOM2012_load_history(num_train, num_val), example_parameter_dictionary['GEFCOM2012'],
                "GEFCOM2012")
    elif dataset_name == 'YahooFinance':
        return init_yahoo_finance(num_train, num_val), example_parameter_dictionary['YahooFinance'], "Yahoo Finance"
    elif dataset_name == 'MNISTDataset':
        error('MNIST Dataset is currently not supported.')
        # return init_mnist(num_train, num_val), example_parameter_dictionary['MNISTDataset'], 'MNISTDataset'
    elif dataset_name == 'UCF101':
        error('UCF101 Dataset is currently not supported.')
    elif dataset_name == 'HAR':
        return init_har(num_train, num_val), example_parameter_dictionary['HAR'], 'HAR'
    elif dataset_name == 'PTB':
        error('PTB Dataset is currently not supported.')
    elif dataset_name == 'DNA1':
        return init_dna1(num_train, num_val), example_parameter_dictionary['DNA1'], 'DNA1'
    elif dataset_name == 'SMSSpam':
        error('SMSSpam Dataset is currently not supported.')
        # return init_sms_spam(num_train, num_val), example_parameter_dictionary['SMSSpam'], 'SMSSpamRecognition'
    else:
        error(f"Unsupported dataset: {dataset_name}.")


def init(mode: str) -> Any or NoReturn:
    """
    Initialize the training process.
    :return: (num_epoch, hidden_size, (train_x, train_y, val_x, val_y))
    """
    global device
    args = generate_parser().parse_args()

    log_assert(mode in ['demo', 'comp'], f'mode should be either demo or comp (Got: {mode}).')

    if args.cpu:
        device = torch.device('cpu')
    info(f'Program is running on {device.__str__().upper()}.')

    if args.seed < 0:
        torch.manual_seed(0)  # set default seed
    else:
        torch.manual_seed(args.seed)  # set manual seed

    global_dict.set('dataset', args.dataset)

    num_train, num_val = parse_num_samples(args.num_train), parse_num_samples(args.num_val)
    log_assert(isinstance(args.epoch, int) and args.epoch > 0, f'The number of epochs must be a positive integer.')
    demo_result = [args.epoch, args.hidden, parse_dataset(args.dataset, num_train, num_val), args.yes, args.save]
    if mode == 'demo':
        return demo_result
    return demo_result + [{
        'sgd': args.comp_sgd,
        'adam': args.comp_adam,
        'adagrad': args.comp_adagrad,
        'skip_fast': args.comp_skip_fast,
        'record_matlab_data': args.comp_record_matlab_data,
    }]


def save_model(name, model):
    save_dir = 'SAVED_MODELS'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, f'{name}.pt')
    torch.save(model, path)
    info(f"{name}: Saved model to {path}.")


def admm_demo(
        num_epoch: int, model: LSTM, train_x, train_y,
        test_x, test_y, example_dict, plot, record_matlab_data=False, save=False
) -> Dict[str, List[float] or str] or NoReturn:
    try:
        loss_func = nn.MSELoss()
        optimizer = ADMMBasedOptimizer(
            model=model,
            training_samples=(train_x, train_y),
            parameter_dictionary=example_dict,
        )  # the optimizer defined in admm.py

        train_loss_plotter = LossCurvePlotter(
            title='Training Loss',
            save_dir='plots',
            constant_dicts=(example_dict['beta'], example_dict['rho'])
        )  # plot training loss

        val_loss_plotter = LossCurvePlotter(
            title='Validation Loss',
            save_dir='plots',
            constant_dicts=(example_dict['beta'], example_dict['rho'])
        )  # plot validation loss

        timer = Timer()

        def update_loss(present_epoch: int, elapsed_time: float = None) -> None:
            """
            Update loss information at current epoch.
            """
            training_loss = loss_func(model(train_x), train_y)
            validation_loss = loss_func(model(test_x), test_y)
            if present_epoch == 0:
                info(f'Training has started.')
            else:
                info(f'Epoch {present_epoch} has done in {elapsed_time: >.2f} ms. ')
            info(f'Present loss: Training: {training_loss: >3.8f} | Validation: {validation_loss: >3.8f}.')
            train_loss_plotter.update(present_epoch, training_loss)
            val_loss_plotter.update(present_epoch, validation_loss)

        update_loss(0)
        for epoch in range(1, num_epoch + 1):
            info(f'Starting epoch {epoch}/{num_epoch}.')
            timer.start()
            optimizer.step()  # feel free to use the optimizer as gradient-based ones in PyTorch
            update_loss(epoch, timer.stop())

        info(f'Training has finished. Total time elapsed: {timer.get_elapsed_time(): .2f} seconds.')

        train_loss_plotter.plot(show=plot, save_name='ADMMTrainingLoss')
        val_loss_plotter.plot(show=plot, save_name='ADMMValidationLoss')

        if record_matlab_data:
            sio.savemat(global_dict.get('logger_filename').split('.')[0] + '_Val.mat', {
                'epoch': np.array(val_loss_plotter.epochs),
                'loss': np.array(val_loss_plotter.losses),
            })

        if save:
            save_model('Fast ADMM-LSTM', model)

        return {
            'name': 'Fast ADMM-LSTM',
            'train_loss': train_loss_plotter.losses,
            'val_loss': val_loss_plotter.losses,
        }

    except KeyboardInterrupt:
        info('Training aborted by user. Process has terminated.')
        exit(0)


def training_demo(plot: bool = True) -> Dict[str, List[float] or str] or NoReturn:
    """
    This demo trains the LSTM-Linear model defined in blocks/lstm.py.
    :param plot: Show the training and validation loss curve after training.
    """

    num_epoch, hidden_size, ((train_x, train_y, test_x, test_y), example_dict, dataset_name), yes, save = init('demo')
    input_size, output_size = train_x.size(2), train_y.size(1)
    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)

    info(f'Training summary: \n'
         f'  - Dataset: {dataset_name}.\n'
         f'  - Number of epochs: {num_epoch}.\n'
         f'  - Number of training samples: {train_x.size(0)} (Shape: {list(train_x.shape)}, {list(train_y.shape)}).\n'
         f'  - Number of validation samples: {test_x.size(0)} (Shape: {list(test_x.shape)}, {list(test_y.shape)}).\n'
         f'  - Number of hidden layers: {hidden_size}.\n'
         f'  - User-defined constants:\n'
         f'      beta: {example_dict["beta"]}.\n'
         f'      rho: {example_dict["rho"]}.\n')

    if not yes:
        command = input(f'Input \'c\' or \'q\' to abort, any other key to continue: ')
        if command == 'c' or command == 'q':
            info('Training aborted. Process has terminated.')
            exit(0)

    return admm_demo(num_epoch, model, train_x, train_y, test_x, test_y, example_dict, plot)


if __name__ == '__main__':
    training_demo(plot=True)  # entry of program
