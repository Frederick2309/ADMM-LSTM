""" parameters.py (c) 2025 Yang, et al.
This file stores all the parameters tuned for ADMM.
"""

from typing import Dict

__all__ = ['example_parameter_dictionary', 'default_epoch']

default_epoch = 100

example_parameter_dictionary: Dict[str, Dict[str, Dict[str, float]]] = {
    'GoogleStock': {
        'rho': {
            'i': 1., 'f': 1., 'g': 1., 'o': 1., 'c': 0.008, 'h': 0.00045, 'y': 0.0000562
            # 'i': 1, 'f': 1, 'g': 1, 'o': 1, 'c': 1e-3, 'h': 1e-4, 'y': 1e-5
        },
        'beta': {
            'wi': 8e-7, 'vi': 8e-7, 'wf': 8e-7, 'vf': 8e-7, 'wg': 8e-7, 'vg': 8e-7, 'wo': 8e-7, 'vo': 8e-7, 'wy': 8e-7
        }
    },
    'GEFCOM2012': {
        'rho': {
            'i': 1, 'f': 1, 'g': 1, 'o': 1, 'c': 0.1, 'h': 0.01, 'y': 0.01
            # 'i': 1, 'f': 1, 'g': 1, 'o': 1, 'c': 0.1, 'h': 0.001, 'y': 0.0001  # without dual y
        },
        'beta': {
            'wi': 8e-7, 'vi': 8e-7, 'wf': 8e-7, 'vf': 8e-7, 'wg': 8e-7, 'vg': 8e-7, 'wo': 8e-7, 'vo': 8e-7, 'wy': 8e-7
        }
    },
    'YahooFinance': {
        'rho': {
            # 'i': 3., 'f': 3., 'g': 3., 'o': 3., 'c': 0.012, 'h': 0.0012, 'y': 0.000082  # use lipschitz c
            # 'i': 1, 'f': 1, 'g': 1, 'o': 1, 'c': 0.1, 'h': 0.02, 'y': 0.01  # use iterative c
            'i': 1, 'f': 1, 'g': 1, 'o': 1, 'c': 0.1, 'h': 0.02, 'y': 0.01  # use iterative c
        },
        'beta': {
            'wi': 1e-8, 'vi': 1e-8, 'wf': 1e-8, 'vf': 1e-8, 'wg': 1e-8, 'vg': 1e-8, 'wo': 1e-8, 'vo': 1e-8, 'wy': 1e-8
        }
    },
    'MNISTDataset': {  # deprecated
        'rho': {
            'i': 1, 'f': 1, 'g': 1, 'o': 1, 'c': 0.012, 'h': 0.0012, 'y': 0.00005
        },
        'beta': {
            'wi': 1, 'vi': 1, 'wf': 1, 'vf': 1, 'wg': 1, 'vg': 1, 'wo': 1, 'vo': 1, 'wy': 10
        }
    },
    'UCF101': {  # deprecated
        'rho': {
            'i': .1, 'f': .1, 'g': .1, 'o': .1, 'c': 0.008, 'h': 0.0001, 'y': 0.000001
        },
        'beta': {
            'wi': 1e-9, 'vi': 1e-9, 'wf': 1e-9, 'vf': 1e-9, 'wg': 1e-9, 'vg': 1e-9, 'wo': 1e-9, 'vo': 1e-9, 'wy': 1e-9
        }
    },
    'HAR': {
        'rho': {
            'i': 1.5, 'f': 1.5, 'g': 1.5, 'o': 1.5, 'c': 0.005, 'h': 8e-04, 'y': 4e-04  # for hidden = 128, y < 4e-6
            # 'i': 1, 'f': 1, 'g': 1, 'o': 1, 'c': 3e-3, 'h': 2e-3, 'y': 1e-4  # with dual y
        },
        'beta': {
            'wi': 8e-7, 'vi': 8e-7, 'wf': 8e-7, 'vf': 8e-7, 'wg': 8e-7, 'vg': 8e-7, 'wo': 8e-7, 'vo': 8e-7, 'wy': 8e-7
        }
    },
    'PTB': {
        'rho': {  # 'i': .8, 'f': .8, 'g': .8, 'o': .8, 'c': 5e-4, 'h': 5e-5, 'y': 8.5e-6
            'i': .8, 'f': .8, 'g': .8, 'o': .8, 'c': 5e-4, 'h': 5e-4, 'y': 1e-5  # sample_len = 1000
        },
        'beta': {
            'wi': 8e-7, 'vi': 8e-7, 'wf': 8e-7, 'vf': 8e-7, 'wg': 8e-7, 'vg': 8e-7, 'wo': 8e-7, 'vo': 8e-7, 'wy': 8e-7
        }
    },
    'DNA1': {
        'rho': {
            # 'i': 1., 'f': 1., 'g': 1., 'o': 1., 'c': 0.001, 'h': 0.0002, 'y': 0.0004  # use lipschitz to update c
            'i': 1., 'f': 1., 'g': 1., 'o': 1., 'c': 0.001, 'h': 0.03, 'y': 0.002
            # 'i': 1., 'f': 1., 'g': 1., 'o': 1., 'c': 0.002, 'h': 0.003, 'y': 0.004
        },
        'beta': {
            'wi': 8e-9, 'vi': 8e-9, 'wf': 8e-9, 'vf': 8e-9, 'wg': 8e-9, 'vg': 8e-9, 'wo': 8e-9, 'vo': 8e-9, 'wy': 8e-9
        }
    },
    'SMSSpam': {
        'rho': {
            'i': 1.0, 'f': 1.0, 'g': 1.0, 'o': 1.0, 'c': 0.01, 'h': 0.001, 'y': 4e-05
        },
        'beta': {
            'wi': 8e-9, 'vi': 8e-9, 'wf': 8e-9, 'vf': 8e-9, 'wg': 8e-9, 'vg': 8e-9, 'wo': 8e-9, 'vo': 8e-9, 'wy': 8e-9
        }
    },
}
