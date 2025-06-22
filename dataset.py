""" dataset.py (c) Yang Yang, 2024-2025
This file defines a dataset for our demo project.
Readers are encouraged to define their own on the basis of our implementation.
"""

import cv2
import re
import av
import csv
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.nn.functional import one_hot
from typing import List, Tuple, Dict, NoReturn
from torch.utils.data import Dataset
from _global import (info, warning, error, global_dict, PATH, device, log_assert, deprecated, total_memory,
                     current_memory_usage, WHITE)

supported_datasets = [
    'GoogleStock',
    'GEFCOM2012',
    'YahooFinance',
    'HAR',
    'DNA1',
]

__all__ = ['supported_datasets'] + [dataset for dataset in supported_datasets]

if 'dataset_names' not in global_dict.keys():
    global_dict['dataset_names'] = list()


def _gen_id() -> str:
    i = len(global_dict['dataset_names'])
    dataset_id = f'dataset{i}'
    while dataset_id in global_dict['dataset_names']:
        i += 1
        dataset_id = f'dataset{i}'
    global_dict['dataset_names'].append(dataset_id)
    return dataset_id


class ADMMDataset(Dataset):
    """
    This class is the prototype of dataset. The minimal specification
    of ADMMDataset should include the initialization of either
    self.sample_dict = dict() or self.samples = list(), where
    self.sample_dict should satisfy the following structure:
     - { 'dataset_name' : [(x, y), (x, y), ...], ... }
    self.samples should satisfy the following structure:
     - [(x, y), (x, y), ...]
    where x is the input, y is the truth.

    :param dataset_name: [Optional] The name or code of the dataset.
    :param predict_single_value: True if you want to use x to predict a single y (i.e. y is a batch of scalers).

    Notice that if dataset_name is not specified, a global id will be
    assigned to the dataset. When both self.samples and self.sample_dict
    are specified, self.samples will be used in prior.
    """

    def __init__(self, *args, dataset_name: str = None, predict_single_value: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_dict = None
        self.samples = None
        self.dataset_name = dataset_name
        self.__assign_name()
        self.predict_single_value = predict_single_value

    def __assign_name(self) -> None:
        if self.dataset_name is None:
            self.dataset_name = _gen_id()
        elif self.dataset_name in global_dict['dataset_names']:
            warning(f'{self.dataset_name} is already defined. A new id will be assigned.')
            self.dataset_name = _gen_id()
        info(f'Loading new dataset: {self.dataset_name}.')

    def __retrieve_by_key(self, key: str) -> List[Tuple[torch.Tensor]]:
        if self.sample_dict is None:
            error('self.sample_dict is not defined.')
        if key not in self.sample_dict.keys():
            error(f'Key {key} is not found')
        return self.sample_dict[key]

    def __retrieve_by_list(self):
        if self.samples is None:
            error('self.samples is not defined.')
        return iter(self.samples)

    def iter_with_key(self, key: str):
        return iter(self.__retrieve_by_key(key))

    def len_with_key(self, key: str):
        try:
            assert self.sample_dict is not None
        except AssertionError:
            error('self.sample_dict is not defined.')
        return len(self.__retrieve_by_key(key))

    def __iter__(self):
        return self.__retrieve_by_list()

    def __len__(self):
        try:
            assert self.samples is not None
        except AssertionError:
            return self.len_with_key(next(iter(self.sample_dict.keys())))
        return len(self.samples)

    def __summary(self):
        pass

    def dataset_type(self):
        if self.samples is None and self.sample_dict is None:
            raise NotImplementedError('Either self.samples or self.sample_dict must be defined.')
        return self.samples is None


class GEFCom2012(ADMMDataset):
    """
    This class implements a dataset to load data provided by
    Global Energy Forecasting Competition 2012. Readers need to
    download the data from the website before instantiate this class.

     - Website: https://www.sciencedirect.com/science/article/pii/S0169207013000745?via%3Dihub#s000065

    Unzip the data file you have downloaded, the structure of the
    repository is shown as follows:
     - GEFCOM2012_Data
       - Load
         - Load_history.csv
         - temperature_history.csv
         - other assistant files
       - Wind
         - benchmark.csv
         - Read Me First.txt
         - test.csv
         - train.csv
         - windforecasts_wf1.csv
         - windforecasts_wf2.csv
         - windforecasts_wf3.csv
         - windforecasts_wf4.csv
         - windforecasts_wf5.csv
         - windforecasts_wf6.csv
         - windforecasts_wf7.csv
         - windpowermeasurements.csv

    To instantiate this class, you need to specify the following parameters.
    :param path: The path to GEFCOM2012_Data directory (top-level).
    :param load_object: The dataset to be loaded which can be specified
      with either a string or a list of strings. The string needs to be
      in the following form:
                           <csv_file>/<data_name>
      data_name can be omitted if there isn't any. Specifically, the
      following patterns are supported.
        STRING (csv_file/data_name):        DATA to be loaded:
        Load_history/<day_num>              load history data (number of days: day_num)
        Load_history/<day1~day2>            load history data (from day1 to day2)
        temperature_history/<day_num>       temperature history data (number of days: day_num)
    :param uniform: Divide the numbers by their maximum such that they are
      all uniformed into [0, 1].
    :param predict_single_value: True if you want to use x to predict a single y (i.e. y is a batch of scalars).
    """

    def __init__(self, path: str, load_object: str or List[str],
                 dataset_name: str = None, uniform: bool = True, predict_single_value: bool = True, **kwargs) -> None:
        super().__init__(dataset_name=dataset_name, predict_single_value=predict_single_value, **kwargs)
        self.uniform = uniform
        self.file_list = ['Load_history', 'temperature_history']
        self.path = self.__parse_path(path)
        self.object_list = self.__parse_load_object(load_object)
        self.__summary()
        self.sample_dict = dict()
        self.__run()

    def __parse_path(self, path: str) -> str:
        if not path.startswith('/'):
            path = os.path.join(PATH, path)
        if not os.path.exists(path):
            error(f'Dataset {self.dataset_name} must be initialized with a valid path (Got: {path}).')
        return os.path.abspath(path)

    def __parse_load_object(self, load_object: str or List[str]) -> List[Tuple[str, str, int] or Tuple[str, str, str]]:
        load_object = [load_object] if isinstance(load_object, str) else load_object
        return [self.__parse_dataset_string(string) for string in load_object]

    def __parse_dataset_string(self, load_object: str) -> Tuple[str, str, int] or Tuple[str, str, str]:
        try:
            file_name, data_name = load_object.split('/')
        except ValueError:
            error(f'Illegal format of target dataset. Got {load_object}, missing dataset_name, data_name or \'/\'. '
                  f'Usage: <dataset_name>/<data_name>.')
        if file_name.endswith('.csv'):
            file_name = file_name.replace('.csv', '')
        if file_name not in self.file_list:
            error(f'File {file_name} not found in {self.file_list}. '
                  f'The following files are supported: {self.file_list}. ')
        if self.file_list.index(file_name) < 2:  # deal with Load_history or temperature_history
            date_range = tuple()
            if '~' in data_name:
                try:
                    date_range = [int(day) for day in data_name.split('~')]
                    date_range = (date_range[0], date_range[1])
                except ValueError:
                    error(f'day1 and day2 must be integers (Got {date_range}).')
            else:
                try:
                    date_range = (1, int(data_name))
                except ValueError:
                    error(f'When using Load_history or temperature_history, '
                          f'data_name must be either a number that represents how '
                          f'many days of data you want to read, or <day1>~<day2>'
                          f'to represent a range of data (Got: {date_range}).')
            return os.path.join(self.path, f'Load/{file_name}.csv'), file_name, date_range
        return f'Wind/{file_name}', file_name, data_name

    def __summary(self) -> None:
        summary = (f'GEFCom2012 dataset {self.dataset_name} has been initialized. '
                   f'Summary: \n - Dataset path: {self.path}. \n - Loaded datasets: \n')
        for i, (file_path, file_name, data_name) in enumerate(self.object_list):
            file_path = os.path.basename(file_path)
            if self.file_list.index(file_name) < 2:
                day1, day2 = data_name
                summary += (f'   {i + 1}. {file_name}: Loaded day {day1} to {day2} '
                            f'at \'{file_path}\'. \n')
            else:
                summary += f'   {i + 1}. {file_name}: Loaded {data_name} at \'{file_path}\'. \n'
        info(summary)

    def __run(self):
        # deal with each type of dataset
        for file_path, file_name, data_name in self.object_list:
            if self.file_list.index(file_name) < 2:
                self.__run_load(file_path, file_name, data_name)
            else:
                self.__run_wind()

    def __run_load(self, file_path: str, file_name: str, data_range: tuple):
        day1, day2 = data_range
        data_list, maximum = self.__read_csv_data_in_load(file_path, day1, day2)
        if file_name not in self.sample_dict.keys():
            self.sample_dict[file_name] = list()
        if self.uniform:
            for day, row in enumerate(data_list):
                if day == day2 - day1:
                    break
                for current_index in range(0, 24):
                    sample_x = torch.stack([row[i] / maximum if i < 24 else data_list[day + 1][i - 24] / maximum
                                            for i in range(current_index, current_index + 24)])
                    sample_y = data_list[day + 1][current_index] / maximum \
                        if self.predict_single_value else torch.stack([data_list[day + 1][i] / maximum if i < 24
                                                                       else data_list[day + 2][i - 24] / maximum
                                                                       for i in
                                                                       range(current_index, current_index + 24)])
                    self.sample_dict[file_name].append((sample_x, sample_y))
        else:
            for day, row in enumerate(data_list):
                if day == day2 - day1:
                    break
                for current_index in range(0, 24):
                    sample_x = torch.stack([row[i] if i < 24 else data_list[day + 1][i - 24]
                                            for i in range(current_index, current_index + 24)])
                    sample_y = data_list[day + 1][current_index] \
                        if self.predict_single_value else torch.stack([data_list[day + 1][i] if i < 24
                                                                       else data_list[day + 2][i - 24]
                                                                       for i in
                                                                       range(current_index, current_index + 24)])
                    self.sample_dict[file_name].append((sample_x, sample_y))

    @staticmethod
    def __read_csv_data_in_load(file_path: str, day1: int, day2: int) -> Tuple[List[List[torch.Tensor]], int]:
        data_list = list()
        maximum = 0
        with open(file_path, mode='r') as f:
            reader = csv.DictReader(f)
            cols = [f'h{i}' for i in range(1, 25)]
            for day, row in enumerate(reader):
                if day < day1 - 1:
                    continue
                data_list.append(list())
                for col in cols:
                    number = torch.tensor([float(str(row[col]).replace(',', ''))],
                                          dtype=torch.float, device=device)
                    if number > maximum:
                        maximum = number
                    data_list[day - day1].append(number)
                if day == day2 + 1:
                    break
        return data_list, maximum

    def __run_wind(self):
        pass

    def display(self, raw: bool = False, number: int = 2) -> None:
        contents = f'Contents in {self.dataset_name}: \n'
        if raw:
            contents = str(self.sample_dict)
        else:
            for key in self.sample_dict.keys():
                contents += f'\'{key}\':\n'
                for i, sample in enumerate(self.sample_dict[key]):
                    contents += f'\t(x) {[num.item() for num in sample[0]]} \n'
                    contents += f'\t(y) {[num.item() for num in sample[1]]} \n'
                    if i == number:
                        contents += f'\t... And the rest {len(self.sample_dict[key]) - number} samples.\n'
                        break
        info(contents)


@deprecated
class ADMMLoader:
    """
    This class implements a dataloader to work with ADMMDataset and whatever
    class that inherits from it. It is like a PyTorch Loader but is designed
    to also work with sample_dict objects.

    :param dataset: An ADMMDataset object that has been initialized.
      (An error will occur if it is not)
    :param batch_size: The number of samples to be learned from at one time.
    :param shuffle: Randomly shuffle the order of samples.
    :param use_dict: Load the sample_dict instead of samples list.
    :param dataset_name: If working with sample_dict, then it is the key of the
      dict which we want to load. Otherwise, it will be ignored.
    """

    def __init__(self, dataset: ADMMDataset, batch_size: int = 5, shuffle: bool = True,
                 use_dict: bool = False, dataset_name: str = None):
        self.original_dataset = dataset
        self.original_dataset_name = dataset.dataset_name
        self.dataset_name = dataset_name
        self.predict_single_value = dataset.predict_single_value
        if not dataset.dataset_type():
            self.dataset: list = dataset.samples
        else:
            if dataset_name is None:
                error(f'A dataset name must be assigned when loading a dataset with sample dict.')
            try:
                self.dataset: list = dataset.sample_dict[dataset_name]
            except KeyError:
                error(f'Dataset \'{dataset_name}\' does not exist in \'{self.original_dataset_name}\'.')
        self.use_dict = use_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            batch_x = torch.stack([x for x, y in batch])
            batch_y = torch.stack([y for x, y in batch]) \
                if self.predict_single_value else torch.stack([y for x, y in batch])
            yield batch_x, batch_y

    def __len__(self):
        # returns the number of batches
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def display(self, num_batches: int = 1) -> None:
        # num_batches: the number of batches to be displayed
        contents = (f'Displaying {num_batches} batch(es) from the loader: \n'
                    f' - Dataset name: {self.original_dataset_name}/{self.dataset_name} \n'
                    f' - Batch size: {self.batch_size} \n'
                    f' - Shuffle: {self.shuffle} \n'
                    f' - Total batches: {len(self)} \n')

        batch_count = 0
        for batch_x, batch_y in self:
            batch_count += 1
            contents += (f' - Batch {batch_count}: \n'
                         f'\t- Input shape {list(batch_x.shape)} \n'
                         f'\t- Truth shape {list(batch_y.shape)} \n')
            for i in range(self.batch_size):
                contents += (f'\t  - (x) {batch_x[i].tolist()} \n'
                             f'\t  - (y) {batch_y[i].tolist()} \n')
            if batch_count == num_batches:
                break
        info(contents)


@deprecated('No authoritative source of GoogleStockDataset. Use it on your own risk.')
class GoogleStockDataset:
    def __init__(self) -> None:
        """
            Copyright Liu, et al.
        """
        import xlrd
        try:
            Stock_data = xlrd.open_workbook(
                "datasets/GoogleStock/GOOG.xls"
            )
        except FileNotFoundError:
            Stock_data = xlrd.open_workbook(
                "../datasets/GoogleStock/GOOG.xls"
            )
        stock_data = Stock_data.sheet_by_index(0)
        stock_X = []
        stock_Y = []
        for i in range(1, 4706):
            stock_X.append(stock_data.cell_value(i, 5))
            stock_Y.append(stock_data.cell_value(i, 4))
        input_X = torch.zeros((4705,))
        output_Y = torch.zeros((4705,))
        for i in range(4705):
            output_Y[i] = stock_Y[i]
            input_X[i] = stock_X[i]
        stock_X_max = max(input_X)
        stock_Y_max = max(output_Y)
        X = list()
        Y = list()
        for i in range(4705):
            value_X = input_X[i] / stock_X_max
            X.append(value_X)
            value_Y = output_Y[i] / stock_Y_max
            Y.append(value_Y)
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in range(10, 4234):
            train_x.append(X[i - 10:i])
            train_y.append(Y[i])

        for i in range(4244, 4705):
            test_x.append(X[i - 10:i])
            test_y.append(Y[i])
        train_x = torch.tensor(train_x).to(device)
        train_y = torch.tensor(train_y).to(device)
        train_y = train_y.reshape(4224, 1)
        test_x = torch.tensor(test_x).to(device)
        test_y = torch.tensor(test_y).to(device)
        test_y = test_y.reshape(461, 1)
        self.train_x, self.train_y, self.test_x, self.test_y = (
            train_x.unsqueeze(2), train_y, test_x.unsqueeze(2), test_y
        )

    def data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.train_x, self.train_y, self.test_x, self.test_y


class YahooFinance:
    """
    Yahoo Finance dataset
      - Yahoo Finance, https://finance.yahoo.com
      - Ran Aroussi, https://pypi.org/project/yfinance
    """
    def __init__(self, stock_symbol: str = None,
                 start_date: str = '2018-01-01', end_date: str = '2024-12-31') -> None:
        import yfinance as yf
        from sklearn.preprocessing import MinMaxScaler
        if stock_symbol is None:
            stock_symbol = 'AAPL'  # Apple Inc
        self.stock_symbol = stock_symbol
        try:
            self.data = yf.download(stock_symbol, start_date, end_date).dropna()
        except Exception as e:
            error(e)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def closed_data(self, input_size: int = 60) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if os.path.isdir('datasets/YahooFinance'):
            paths = os.listdir('datasets/YahooFinance')
            if paths:
                return (
                    torch.load('datasets/YahooFinance/train_x.pt', map_location=device, weights_only=False),
                    torch.load('datasets/YahooFinance/train_y.pt', map_location=device, weights_only=False),
                    torch.load('datasets/YahooFinance/test_x.pt', map_location=device, weights_only=False),
                    torch.load('datasets/YahooFinance/test_y.pt', map_location=device, weights_only=False)
                )

        try:
            closed_data = self.scaler.fit_transform(self.data['Close'].dropna().values.reshape(-1, 1))
        except ValueError:
            error(f'Failed to access Yahoo Finance dataset. Possible solutions: \n'
                  f' - Check you network connection.\n'
                  f' - If you are in China, you will need a proxy.\n'
                  f' - Check yfinance updates (Run pip install --upgrade yfinance).')
        x, y = self.__create_sequences(closed_data, input_size)
        indices = torch.randperm(x.size(0))
        x, y = x[indices], y[indices]
        train_val_ratio = 0.8
        num_train = round(len(x) * train_val_ratio)
        results = x[:num_train], y[:num_train], x[num_train:], y[num_train:]
        os.makedirs('datasets/YahooFinance', exist_ok=True)
        for result, name in zip(results, ('train_x', 'train_y', 'test_x', 'test_y')):
            torch.save(result, f'datasets/YahooFinance/{name}.pt')
        return results

    @staticmethod
    def __create_sequences(data, window_size) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = list(), list()
        for i in range(len(data) - window_size):
            seq = data[i:i + window_size]
            target = data[i + window_size]
            inputs.append(seq)
            labels.append(target)
        return (torch.tensor(np.array(inputs), dtype=torch.float, device=device).clone().detach(),
                torch.tensor(np.array(labels), dtype=torch.float, device=device).clone().detach())


@deprecated('MNIST dataset has been removed')
class MNISTDataset:
    def __init__(self, root: str = 'datasets/MNIST', download: bool = True) -> None:
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = datasets.MNIST(root=root, train=True, download=download, transform=transform)
        self.test_dataset = datasets.MNIST(root=root, train=False, download=download, transform=transform)
        raise RuntimeError("MNIST has been removed from available datasets")

    def data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        train_x = self.train_dataset.data.float().reshape(-1, 28, 28)  # Shape (N, 28, 28)
        test_x = self.test_dataset.data.float().reshape(-1, 28, 28)  # Shape (N, 28, 28)
        train_y = one_hot(self.train_dataset.targets, num_classes=10).float()
        test_y = one_hot(self.test_dataset.targets, num_classes=10).float()
        return train_x, train_y, test_x, test_y

    def show(self, num_samples: int = 6, dataset_type: str = 'train') -> None:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("TkAgg")
        dataset = self.train_dataset if dataset_type == 'train' else self.test_dataset
        fig, axes = plt.subplots(1, num_samples, figsize=(12, 2))
        for i in range(num_samples):
            img, label = dataset[i]
            axes[i].imshow(img.squeeze(0), cmap='gray')
            axes[i].set_title(f"Label: {label}")
            axes[i].axis('off')
        plt.show()


@deprecated('UCF dataset has been removed')
class UCF101:
    def __init__(self, path: str = 'datasets/UCF-101', train_val_split: str = 'ucfTrainTestlist',
                 use_categories: List[str] or None = None, verbose: bool = True) -> None:
        info('Loading UCF101. This might take a few minutes.')
        self.verbose = verbose
        path = path if path.startswith('/') else os.path.join(os.path.abspath(os.getcwd()), path)
        self.path_dict = {
            'root': path,
            'train_val_split': os.path.join(path, train_val_split)
        }
        for name, abs_path in self.path_dict.items():
            log_assert(os.path.isdir(abs_path), f'Cannot find {name} at {abs_path}.')
        if use_categories is None:
            use_categories = [
                'ApplyEyeMakeup', 'Archery', 'Basketball', 'Biking', 'Diving', 'TaiChi'
            ]
        if len(use_categories) == 0:
            error('Must specify categories to load.')
        self.use_categories = use_categories
        self.training_samples = self.read_train_val_split('trainlist01.txt')
        self.validation_samples = self.read_train_val_split('testlist01.txt')
        raise RuntimeError("UCF101 has been removed from available datasets")

    def read_train_val_split(self, spec_file_name: str
                             ) -> Dict[str, List[Dict[str, str or int or av.container.InputContainer]]] or NoReturn:
        pattern = (r'(?P<Path>(?P<DirName>\w+)/(?P<Filename>v_(?P<Category>\w+)_g(?P<Group>\d{2})'
                   r'_c(?P<Clip>\d{2}).avi))\s*\d*')
        spec_path = os.path.join(self.path_dict['train_val_split'], spec_file_name)
        root = self.path_dict['root']
        path_dict: Dict[str, List[Dict[str, str or int or av.container.InputContainer]]] = dict()
        try:
            with open(spec_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line.rstrip()
                    reg_expr = re.match(pattern, line)
                    if reg_expr is None:
                        continue
                    category = reg_expr.group('Category')
                    filename = reg_expr.group('Filename')
                    dir_name = reg_expr.group('DirName')
                    if not reg_expr or category not in self.use_categories:
                        continue
                    path = os.path.join(root, reg_expr.group('Path'))
                    try:
                        container = av.open(path)
                        if container.streams:
                            if category not in path_dict.keys():
                                path_dict[category] = list()
                            path_dict[category].append({
                                'path': path,
                                'container': container,
                                'label': self.use_categories.index(category)
                            })
                        elif self.verbose:
                            warning(f'AVI file {filename} in {dir_name} is damaged and therefore ignored.')
                    except av.InvalidDataError:
                        if self.verbose:
                            warning(f'File {filename} in {dir_name} is invalid and therefore ignored.')
                return path_dict
        except FileNotFoundError:
            error(f'Specification file {spec_file_name} is not found in {self.path_dict["train_val_split"]}.')

    @staticmethod
    def process_video(frames_each_video: int, container: av.container.InputContainer,
                      scale_to: int) -> torch.Tensor:
        stream = container.streams.video[0]
        total_frames = stream.frames
        interval = torch.floor(torch.tensor(total_frames / (frames_each_video + 1),
                                            dtype=torch.float, requires_grad=False)).item()
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i % interval == 0 and len(frames) < frames_each_video:
                frame_array = cv2.resize(np.array(frame.to_image()), (scale_to, scale_to), interpolation=cv2.INTER_AREA)
                gray_frame = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)
                frames.append(torch.flatten(torch.tensor(gray_frame / 255., dtype=torch.float, device=device)))
        return torch.stack(frames).clone().detach()

    def process_samples(self, sample_dict: Dict[str, List[Dict[str, str or int or av.container.InputContainer]]],
                        frames_each_video: int, scale_to: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = [], []
        num_categories = len(self.use_categories)
        for _, samples in sample_dict.items():
            if not samples:
                continue
            for sample in samples:
                container, label = sample['container'], sample['label']
                x.append(self.process_video(frames_each_video, container, scale_to))
                y.append(torch.nn.functional.one_hot(torch.tensor(label),
                                                     num_categories).to(torch.float).to(device).clone().detach())
        return torch.stack(x), torch.stack(y)

    def data(self, frames_each_video: int = 20,
             scale_to: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        train_x, train_y = self.process_samples(self.training_samples, frames_each_video, scale_to)
        test_x, test_y = self.process_samples(self.validation_samples, frames_each_video, scale_to)
        return train_x, train_y, test_x, test_y


class HAR:
    """
    Human Activity Recognition Dataset (HAR) by D. Anguita, et al.
     - v1 Using acceleration measurements for activity recognition: An effective learning algorithm for constructing neural classifiers
     - Website: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
    """

    def __init__(self) -> None:
        self.har_root = r'datasets/HAR'
        self.x_train_path = r'X_train.txt'
        self.x_test_path = r'X_test.txt'
        self.y_train_path = r'y_train.txt'
        self.y_test_path = r'y_test.txt'
        self.float_pattern = r'(?P<float>[+-]?\d*\.?\d+)e(?P<exp>[+-]?\d*)'

    def load_file(self, file_name: str, use_int: bool) -> List[List[float]] or List[int] or NoReturn:
        contents = []
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                for line in file:
                    row = []
                    for data in line.strip().split():
                        if use_int:
                            try:
                                row.append(int(data))
                            except ValueError:
                                error(f'Invalid data: {data}')

                        elif 'e' in data:
                            results = re.match(self.float_pattern, data)
                            if results is not None:
                                row.append(float(results.group('float')) * pow(10, int(results.group('exp'))))
                            else:
                                error(f'Invalid data: {data}')

                        else:
                            error(f'Invalid data: {data} (not int mode)')
                    contents.append(row)

            return contents
        except FileNotFoundError:
            error(f'File {file_name} is not found.')
        except Exception as e:
            error(f'An unexpected error occurred while loading {file_name}: {e}.')

    @staticmethod
    def get_lengths(y_list: List[int]) -> List[int]:
        current_len, y_before, lengths = 0, -1, []
        for y in y_list:
            if y != y_before:  # start of each segment
                if current_len > 0:
                    lengths.append(current_len)
                current_len = 1
                y_before = y
            else:
                current_len += 1
        lengths.append(current_len)  # consider the last segment
        return lengths

    def process_samples(self, x_path: str, y_path: str, minimal_window: int = 10) \
            -> Tuple[torch.Tensor, torch.Tensor] or NoReturn:
        x_contents, y_contents = self.load_file(x_path, False), self.load_file(y_path, True)
        assert len(x_contents) == len(y_contents), (f'Incompatible lengths: '
                                                    f'x = {len(x_contents)}, y = {len(y_contents)}.')
        length_list = self.get_lengths(y_contents)
        length_iter = iter(length_list)
        if not length_list:
            error(f'Length list is empty. Dataset is not loaded.')
        info(f'Using minimal window size: {minimal_window}.')
        if min(length_list) < minimal_window:
            warning(f'Minimum length ({min(length_list)}) is less than minimal window ({minimal_window}). '
                    f'Samples of shorter lengths will not be used.')

        x_final, y_final, x_action, y_action, y_before, i = [], [], [], [], y_contents[0], 0
        current_len = next(length_iter)
        while True:
            if current_len >= minimal_window:
                indices = np.linspace(i, i + current_len - 1, minimal_window, dtype=int).tolist()
                x_final.append([x_contents[i] for i in indices])
                y_final.append(y_contents[i])
            i += current_len
            try:
                current_len = next(length_iter)
            except StopIteration:
                break

        return (torch.tensor(x_final, dtype=torch.float, device=device),
                torch.nn.functional.one_hot(torch.tensor(y_final, dtype=torch.long, device=device)).squeeze(1))

    def data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] or NoReturn:
        train_x, train_y = self.process_samples(
            os.path.join(self.har_root, self.x_train_path),
            os.path.join(self.har_root, self.y_train_path)
        )
        test_x, test_y = self.process_samples(
            os.path.join(self.har_root, self.x_test_path),
            os.path.join(self.har_root, self.y_test_path)
        )
        return (train_x.to(device), train_y.to(device).to(torch.float),
                test_x.to(device), test_y.to(device).to(torch.float))


@deprecated('PTB dataset has been removed')
class PTB:
    def __init__(self, dataset_path: str = 'datasets/PTB',
                 patient_range: List[int] | None = None, include_na: bool = False,
                 number_of_leads: int = 15, verbose: bool = True,
                 train_test_split: List[int] = None,
                 max_memory_usage: float = 0.8) -> None:
        if train_test_split is None:
            train_test_split = [3, 1]
        import wfdb
        from glob import glob
        from _global import MAGENTA, color_wrapper
        self.data_lib: Dict[str, Dict[str, str | int | np.ndarray]] = {}
        self.diagnosis_map = {}
        self.include_na = include_na
        self.warning_count = 0
        self.number_of_leads = number_of_leads
        self.train_test_split = train_test_split
        inf = float('inf')
        self.minimum_length = inf
        self.max_memory_usage = max_memory_usage
        warning_count = 0
        if patient_range is None:
            patient_range = [1, 294]
        info('Loading PTB dataset. This might take a while...')

        def warning_callback_func():
            nonlocal warning_count
            warning_count += 1

        with tqdm([f'{patient_id:03d}' for patient_id in range(patient_range[0], patient_range[1] + 1)],
                  desc=f'{WHITE}Reading dataset. Progress: ',
                  bar_format='{l_bar}{bar: 30}{r_bar}', unit='sample') as pbar:
            for patient_id in pbar:
                dir_path = os.path.join(dataset_path, f'patient{patient_id}')
                if not os.path.isdir(dir_path):
                    warning(f'Patient {patient_id} is not found.', verbose=verbose, callback_func=warning_callback_func)
                    continue
                for dat_file in glob(os.path.join(dir_path, '*.dat')):
                    basename = os.path.basename(dat_file).split('.')[0]
                    if basename in self.data_lib.keys():
                        warning(
                            f'Basename {basename} already exists in patient {self.data_lib[basename]["patient_id"]}. '
                            f'Abandoning {patient_id}/{basename}.', verbose=verbose,
                            callback_func=warning_callback_func)
                        continue
                    path_with_basename = dat_file[:-4]
                    signals = None
                    try:
                        signals = wfdb.rdrecord(path_with_basename)
                    except FileNotFoundError as e:
                        if not os.path.isfile(os.path.join(dir_path, f'{basename}.xyz')):
                            warning(f'Missing annotation file (.xyz) for record {patient_id}/{basename}.',
                                    verbose=verbose, callback_func=warning_callback_func)
                        else:
                            warning(f'File not found: {e}.')
                    except Exception as e:
                        warning(f'An unexpected error occurred while loading {patient_id}/{basename}: {e}.',
                                verbose=verbose, callback_func=warning_callback_func)
                    if signals.n_sig != number_of_leads:
                        warning(f'Number of leads incompatible for {patient_id}/{basename} '
                                f'(Expected: {number_of_leads}, got: {signals.n_sig}).',
                                verbose=verbose, callback_func=warning_callback_func)
                        continue
                    diagnosis = self.get_diagnosis(f'{path_with_basename}.hea')
                    if diagnosis:
                        if diagnosis in self.diagnosis_map.keys():
                            diagnosis_id = self.diagnosis_map[diagnosis]
                        else:
                            diagnosis_id = len(self.diagnosis_map)
                            self.diagnosis_map[diagnosis] = diagnosis_id
                    else:
                        warning(f'Diagnosis unavailable for {patient_id}/{basename} (Not exist or N/A). '
                                f'Use include_na=True to include N/A samples.',
                                verbose=verbose, callback_func=warning_callback_func)
                        continue
                    signal = signals.p_signal
                    self.minimum_length = min(float(signal.shape[0]), self.minimum_length)
                    self.data_lib[basename] = {
                        'patient_id': patient_id,
                        'record': signal,
                        'diagnosis': diagnosis_id
                    }

        if self.minimum_length == inf:
            error(f'No PTB data has been loaded. Please check the path to your dataset (Current: {dataset_path}).')
        self.minimum_length = int(self.minimum_length)
        self.num_types = len(self.diagnosis_map)

        info(f'Found types: {self.num_types}. Warnings: {warning_count}. '
             + ('' if verbose or warning_count == 0 else f'Set {color_wrapper("verbose", MAGENTA)}='
                                                         f'{color_wrapper("True", MAGENTA)} to see the warnings. ')
             + f'Minimum length: {self.minimum_length}. '
             + 'Generating labels...')

        raise RuntimeError('PTB dataset has been removed')

    def get_diagnosis(self, header_path: str) -> str | None:
        with open(header_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            if '# Reason for admission' in line:
                diagnosis = line.strip().split(': ')[-1]
                if not self.include_na and (diagnosis == 'n/a' or diagnosis == 'N/A'):
                    break
                return diagnosis
        return None

    def data(self, sample_length: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device_total_memory = total_memory()
        sample_length = self.minimum_length if sample_length is None else min(sample_length, self.minimum_length)
        records, labels = [], []
        with tqdm(self.data_lib.items(), desc=f'{WHITE}Generating samples. Progress: ',
                  bar_format='{l_bar}{bar: 30}{r_bar}', unit='sample') as pbar:
            for _, value in pbar:
                memory_usage = current_memory_usage() / pow(1024, 3)
                memory_percentage = memory_usage / device_total_memory
                if memory_percentage > self.max_memory_usage:
                    error(f'Memory usage exceeds maximum allowed: {memory_percentage} > {self.max_memory_usage}.')
                pbar.set_postfix(MemoryUsage=f'{memory_percentage * 100:.1f}%', refresh=True)
                records.append(self.downsample_to(value['record'], sample_length))
                labels.append(one_hot(torch.tensor(value['diagnosis'], dtype=torch.long), num_classes=self.num_types))
        records, labels = torch.stack(records).to(torch.float), torch.stack(labels).to(torch.float)
        num_samples = len(records)
        assert num_samples == len(labels), f'Length of records ({len(records)}) and labels ({len(labels)}) mismatch.'
        indices = torch.randperm(num_samples)
        shuffled_features, shuffled_labels = records[indices], labels[indices]
        num_train = round(num_samples * self.train_test_split[0] / sum(self.train_test_split))
        return (self.min_max_uniform(shuffled_features[:num_train].to(device)), shuffled_labels[:num_train].to(device),
                self.min_max_uniform(shuffled_features[num_train:].to(device)), shuffled_labels[num_train:].to(device))

    @staticmethod
    def downsample_to(sample: torch.tensor or np.ndarray, sample_length: int) -> torch.tensor:
        sample = torch.as_tensor(sample, dtype=torch.float)
        original_length = sample.size(0)
        indices = torch.linspace(0, original_length - 1, sample_length).long()
        return sample[indices, :]

    @staticmethod
    def min_max_uniform(feature: torch.Tensor) -> torch.Tensor:
        sample_max, sample_min = feature.max().item(), feature.min().item()
        return (feature - sample_min) / (sample_max - sample_min)

    def show(self, basename: str = None, leads: int or List[int] = None,
             record: np.ndarray or torch.Tensor = None) -> None:
        if leads is None:
            leads = list(range(1, self.number_of_leads + 1))
        from matplotlib import use
        use('TkAgg')
        from matplotlib import pyplot as plt
        if record is None:
            log_assert(basename is not None, 'Basename must be specified when record is None.')
            log_assert(basename in self.data_lib.keys(), f'Cannot find basename: {basename}.')
            record = self.data_lib[basename]['record']
            user_defined_record = False
        else:
            leads = ['user-defined']
            user_defined_record = True
            record = record.numpy() if isinstance(record, torch.Tensor) else record
        if isinstance(leads, int):
            leads = [leads]
        fig, axes = plt.subplots(len(leads), 1)

        def plot_axes(current_axes, lead_id, data):
            current_axes.plot(list(range(len(data))), data)
            current_axes.set_xlabel(f'Lead {lead_id}')
            current_axes.grid(True)

        for i, lead in enumerate(leads):
            if user_defined_record:
                plot_axes(axes, lead, record)
            elif 0 <= lead <= self.number_of_leads:
                current_lead_record = record[:, lead - 1]
                try:
                    plot_axes(axes[i], lead, current_lead_record)
                except TypeError:
                    plot_axes(axes, lead, current_lead_record)
            else:
                error(f'Lead number exceeds the maximum '
                      f'(Expected: <{self.number_of_leads}; Got: {lead} in {leads}).')
        plt.show()


class DNA1:
    def __init__(self, data_file: str = 'datasets/DNA1/promoters.data', train_test_split: List[int] = None,) -> None:
        """
        Molecular Biology (Promoter Gene Sequences) Dataset
         - Website: https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences
        """
        if train_test_split is None:
            train_test_split = [4, 1]
        self.train_test_split = train_test_split
        self.map = {
            'a': one_hot(torch.tensor(0, dtype=torch.long), 4).to(torch.float),
            'c': one_hot(torch.tensor(1, dtype=torch.long), 4).to(torch.float),
            'g': one_hot(torch.tensor(2, dtype=torch.long), 4).to(torch.float),
            't': one_hot(torch.tensor(3, dtype=torch.long), 4).to(torch.float),
        }
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            error(f'Data file {data_file} not found.')
        except IOError as e:
            error(f'Cannot open data file {data_file}: {e}')
        pattern = r'(?P<p_sym>[+-]),[\w\W\d]*,\s+(?P<seq>[actg]+)'
        self.features, self.labels = [], []
        with tqdm(lines, desc=f'{WHITE}Loading data file. Progress: ',
                  bar_format='{l_bar}{bar: 30}{r_bar}', unit='sample') as pbar:
            for i, line in enumerate(pbar):
                try:
                    groups = re.match(pattern, line.strip())
                    promoter_symbol, sequence = groups.group('p_sym'), groups.group('seq')
                except AttributeError:
                    warning(f'Cannot parse line {i + 1}')
                    continue
                log_assert(promoter_symbol is not None and sequence is not None,
                           'Cannot parse data file. Please check the format')
                self.features.append(self.encode_seq(sequence))
                self.labels.append(torch.as_tensor(0.) if promoter_symbol == '-' else torch.as_tensor(1.))
        self.features = torch.stack(self.features).to(torch.float)
        self.labels = torch.stack(self.labels).unsqueeze(1).to(torch.float)

    def encode_seq(self, seq: str) -> torch.Tensor:
        encoded_seq = []
        for symbol in seq:
            encoded_seq.append(self.map[symbol])
        return torch.stack(encoded_seq).to(torch.float)

    def data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = self.features.size(0)
        indices = torch.randperm(num_samples)
        num_train = round(num_samples * self.train_test_split[0] / sum(self.train_test_split))
        shuffled_features, shuffled_labels = self.features[indices], self.labels[indices]
        return (shuffled_features[:num_train], shuffled_labels[:num_train],
                shuffled_features[num_train:], shuffled_labels[num_train:])


@deprecated('SMSSpam has been removed')
class SMSSpamRecognition:
    def __init__(self, data_file: str = 'datasets/SMSSpamRecognition/SMSSpamCollection',
                 train_test_split: List[int] = None, maximum_length: int = 25) -> None:
        if train_test_split is None:
            train_test_split = [4, 1]
        self.train_test_split = train_test_split
        self.maximum_length = maximum_length
        self.features, self.labels = self.process_text(self.read_data(data_file))
        raise RuntimeError('SMSSpam has been removed')

    @staticmethod
    def read_data(path: str) -> List[List[str or int]]:
        with open(path, 'r') as f:
            lines = f.readlines()
        pattern = r'(?P<label>\w+)\s+(?P<text>[\w\W]+)'
        data = []
        with tqdm(lines, desc=f'{WHITE}Reading SMS texts. Progress: ',
                  bar_format='{l_bar}{bar: 30}{r_bar}', unit='sample') as pbar:
            for i, line in enumerate(pbar):
                try:
                    groups = re.match(pattern, line.strip())
                    label, text = groups.group('label'), groups.group('text')
                    assert label in ['ham', 'spam']
                except AttributeError or AssertionError:
                    warning(f'Cannot parse line {i}: {line}.')
                    continue
                data.append([text, 0 if label == 'ham' else 1])
        return data

    def process_text(self, data: List[List[str or int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        import html
        import string
        import unicodedata
        alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + ' '
        char_to_num = {char: idx for idx, char in enumerate(alphabet)}
        num_features = len(char_to_num)
        padding_token = torch.zeros(num_features, dtype=torch.float)

        def preprocess_text(text: str) -> str:
            # 1: normalize
            text = unicodedata.normalize('NFKD', text)
            text = ''.join([c for c in text if not unicodedata.combining(c)])
            # 2: clean special chars
            text = html.unescape(text)
            text = re.sub(r'<.*?>', '', text)                # handle HTML tags
            text = re.sub(r'\s+', ' ', text)                 # handle spaces
            text = re.sub(r'[—–-]', ' ', text)            # handle -
            text = re.sub(r'http\S+|www\S+', 'URL', text)    # handle urls
            text = re.sub(r'\S+@\S+', 'EMAIL', text)         # handle emails
            text = re.sub(r'[?!;]', '.', text)               # handle unimportant symbols
            text = re.sub(r'£', '$', text)                   # handle currency sign
            text = re.sub(r'[‘’“”]', '\'', text)  # handle quotations
            # 3: from upper to lower
            text = text.lower()
            return text

        encoded_data, maximum_length = [], 0
        encoded_tensors, encoded_labels = [], []

        with tqdm(data, desc=f'{WHITE}Preprocessing SMS texts. Progress: ',
                  bar_format='{l_bar}{bar: 30}{r_bar}', unit='sample') as pbar:
            for i, (sms, label) in enumerate(pbar):
                sms = preprocess_text(sms)
                if len(sms) > self.maximum_length:
                    continue
                else:
                    encoded_tensor = []
                    for char in sms:
                        try:
                            encoded_tensor.append(
                                one_hot(
                                    torch.as_tensor(char_to_num[char], dtype=torch.long),
                                    num_classes=num_features
                                ).to(torch.float)
                            )
                        except KeyError:
                            encoded_tensor.append(padding_token)
                    maximum_length = max(maximum_length, len(encoded_tensor))
                    encoded_data.append([encoded_tensor, label])

        with tqdm(encoded_data, desc=f'{WHITE}Generating paddings. Progress: ',
                  bar_format='{l_bar}{bar: 30}{r_bar}', unit='sample') as pbar:
            for i, (feature, label) in enumerate(pbar):
                feature += [padding_token] * (maximum_length - len(feature))
                encoded_tensors.append(torch.stack(feature).to(torch.float))
                encoded_labels.append(one_hot(torch.tensor(label, dtype=torch.long), num_classes=2).to(torch.float))
        return torch.stack(encoded_tensors), torch.stack(encoded_labels)

    def data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = self.features.size(0)
        indices = torch.randperm(num_samples)
        num_train = round(num_samples * self.train_test_split[0] / sum(self.train_test_split))
        shuffled_features, shuffled_labels = self.features[indices], self.labels[indices]
        return (shuffled_features[:num_train], shuffled_labels[:num_train],
                shuffled_features[num_train:], shuffled_labels[num_train:])


if __name__ == '__main__':
    dataset = SMSSpamRecognition()
    main_train_x, main_train_y, main_test_x, main_test_y = dataset.data()
    info(f'Shapes: \n'
         f'  Train x: {list(main_train_x.shape)} \n'
         f'  Train y: {list(main_train_y.shape)} \n'
         f'  Test x:  {list(main_test_x.shape)} \n'
         f'  Test y:  {list(main_test_y.shape)}')
    info(torch.max(main_train_x))
