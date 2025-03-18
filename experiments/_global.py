""" _global.py (c) Yang Yang, 2024-2025
Stores global variables such as global dictionaries and path info.
"""

import os
import inspect
import functools
import psutil
from typing import Any, Dict, NoReturn, Callable

""" format tools """
WHITE = '\033[37m'
RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[34m'
YELLOW = '\033[33m'
RESET = '\033[0m'
MAGENTA = '\033[35m'

color_list = [
    'b',  # Blue
    'g',  # Green
    'r',  # Red
    'c',  # Cyan
    'm',  # Magenta
    'y',  # Yellow
    'k',  # Black
    '#FF5733',  # Custom Hex color (e.g., orange-red)
    '#33FF57',  # Custom Hex color (e.g., lime green)
    '#3357FF',  # Custom Hex color (e.g., blue)
    '#8A2BE2',  # Custom Hex color (e.g., blue-violet)
    '#D2691E',  # Custom Hex color (e.g., chocolate)
    '#FF1493',  # Custom Hex color (e.g., deep pink)
]


def color_wrapper(msg: str, color: str) -> str:
    return f'{color}{msg}{RESET}'


from datetime import datetime


""" error management """


def get_location() -> str:
    frame_error = inspect.currentframe().f_back.f_back
    error_filename = frame_error.f_code.co_filename
    error_line = frame_error.f_lineno
    error_func = frame_error.f_code.co_name
    frame_calling = frame_error.f_back
    if frame_calling is None:
        frame_calling = frame_error
    calling_filename = frame_calling.f_code.co_filename
    calling_line = frame_calling.f_lineno
    return (f'  - Error occurs when calling function \"{error_func}\" in File \"{calling_filename}\", line {calling_line};\n'
            f'  - Error is reported in File \"{error_filename}\", line {error_line}. \n')


def time(f: str = "%H:%M:%S") -> str:
    return datetime.now().strftime(f)


""" global variables """


class GlobalDict:
    def __init__(self):
        self.contents = dict()

    def set(self, key: str, value: Any):
        self.contents[key] = value

    def get(self, key: str):
        return self.contents[key]

    def keys(self):
        return self.contents.keys()

    def __setitem__(self, key, value):
        self.set(key, value)

    def __getitem__(self, key):
        return self.get(key)


global_dict = GlobalDict()


""" path and project structure """
PATH = os.path.abspath(os.getcwd())


""" function decorators """


def deprecated(msg: str = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = msg or f'{func.__name__} is deprecated and will be removed in future versions.'
            warning(message, 'deprecated')
            return func(*args, **kwargs)
        return wrapper
    return decorator


""" logging """
import logging


if 'loggers' not in global_dict.keys():
    global_dict['loggers'] = dict()


def __log_init(filename: str = None) -> logging.Logger:
    loggers: Dict[str, logging.Logger] = global_dict['loggers']
    if filename is None:
        if not len(loggers.keys()):
            filename = 'logs/ADMMRunningLogs.log'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        else:
            filename = next(iter(loggers.keys()))
    if filename not in global_dict['loggers'].keys():
        if os.path.exists(filename):
            if filename.endswith('.log'):
                filename = filename[:-len('.log')]
            i = 1
            filename = f'{filename}_{i}.log'
            while os.path.exists(filename):
                i += 1
                filename = f'{filename[:-len(f"_{i - 1}.log")]}_{i}.log'
        loggers[filename] = logging.getLogger(filename)
        loggers[filename].setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        loggers[filename].addHandler(file_handler)
    global_dict.set('logger_filename', filename)
    return loggers[filename]


def log_info(msg: str = '', filename: str = None) -> None:
    __log_init(filename).info(f'{msg}')


def log_warning(msg: str = '', filename: str = None) -> None:
    __log_init(filename).warning(f'{msg}')


def log_error(msg: str = '', filename: str = None) -> None:
    __log_init(filename).error(f'{msg} \n{get_location()}')


def callback(callback_func: Callable or None = None, *callback_args: Any):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            callback_func(*callback_args)
            return results
        return wrapper
    return decorator


def warning(msg: Any = str(), warning_type: str = None, use_logger: bool = True,
            verbose: bool = True, callback_func: Callable = None, *callback_args: Any) -> None:
    if verbose:
        assert warning_type is None or warning_type in ['deprecated']
        if warning_type == 'deprecated':
            warning_string = 'DEPRECATED WARNING'
        else:
            warning_string = 'WARNING'
        if use_logger:
            log_warning(msg)
        print(f'[{time()}] {color_wrapper(warning_string, YELLOW)}: {msg}')
    if callback_func is not None:
        callback_func(*callback_args)


def error(msg: Any = str(), code: int = 1, use_logger: bool = True, assertion: bool = False) -> NoReturn:
    msg = f'{msg} \n{get_location()}'
    if use_logger:
        log_error(msg)
    print(f'[{time()}] {color_wrapper("ASSERTION FAILURE" if assertion else "ERROR", RED)}: {msg}')
    exit(code)


def info(msg: Any = str(), use_logger: bool = True) -> None:
    if use_logger:
        log_info(msg)
    print(f'[{time()}] {color_wrapper("INFO", GREEN)}: {msg}')


def log_assert(condition: bool, msg: Any = str(), code: int = 1) -> None or NoReturn:
    if not condition:
        error(msg, code, assertion=True)
    return


""" version check """
# python >= 3.10: use match syntax
import sys
sys_info = sys.version_info


def version_check():
    if sys_info < (3, 10):
        error(f'Python version must be at least 3.10 (Got: {sys_info[0], sys_info[1]}).')


""" device management """
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


GLOBAL_PROCESS = psutil.Process(os.getpid())


def current_memory_usage():
    return GLOBAL_PROCESS.memory_info().rss


def total_memory():
    return psutil.virtual_memory().total / pow(1024, 3)  # GB


def move(m: torch.nn.Module or torch.Tensor):
    m.to(device)
