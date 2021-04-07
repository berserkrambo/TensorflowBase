# -*- coding: utf-8 -*-
# ---------------------

import os
import tensorflow as tf

PYTHONPATH = '..:.'
if os.environ.get('PYTHONPATH', default=None) is None:
    os.environ['PYTHONPATH'] = PYTHONPATH
else:
    os.environ['PYTHONPATH'] += (':' + PYTHONPATH)

import yaml
import socket
import random
import numpy as np
from path import Path
from typing import Optional
import termcolor


def set_seed(seed=None):
    # type: (Optional[int]) -> int
    """
    set the random seed using the required value (`seed`)
    or a random value if `seed` is `None`
    :return: the newly set seed
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    return seed


class Conf(object):
    HOSTNAME = socket.gethostname()

    CAPRA_PATH = Path('/nas/softechict-nas-3/rgasparini')


    def __init__(self, conf_file_path=None, exp_name=None, log=True):
        # type: (str, str, bool) -> None
        """
        :param conf_file_path: optional path of the configuration file
        :param seed: desired seed for the RNG; if `None`, it will be chosen randomly
        :param exp_name: name of the experiment
        :param log: `True` if you want to log each step; `False` otherwise
        """
        self.exp_name = exp_name
        self.log_each_step = log

        # print project name and host name
        self.project_name = Path(__file__).parent.parent.basename()
        m_str = f'┃ {self.project_name}@{Conf.HOSTNAME} ┃'
        u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
        b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
        print(u_str + '\n' + m_str + '\n' + b_str)

        # define output paths
        self.project_log_path = Path('./log')  # alternative: Path(Conf.CAPRA_PATH / 'log' / self.project_name)
        self.exp_log_path = self.project_log_path / exp_name
        self.exp_weights_path = self.exp_log_path / 'weights'

        self.keys_to_hide = list(self.__dict__.keys()) + ['keys_to_hide']

        # if the configuration file is not specified
        # try to load a configuration file based on the experiment name
        tmp = Path(__file__).parent / (self.exp_name + '.yaml')
        if conf_file_path is None and tmp.exists():
            conf_file_path = tmp

        # read the YAML configuation file
        if conf_file_path is None:
            y = {}
        else:
            conf_file = open(conf_file_path, 'r')
            y = yaml.load(conf_file, Loader=yaml.Loader)

        # read configuration parameters from YAML file
        # or set their default value
        self.lr = y.get('LR', 0.0001)  # type: float
        self.epochs = y.get('EPOCHS', 10)  # type: int
        self.n_workers = y.get('N_WORKERS', 4)  # type: int
        self.batch_size = y.get('BATCH_SIZE', 8)  # type: int
        self.stride = y.get('STRIDE', 4)  # type: int
        self.model = y.get('MODEL', "mobilenet")  # type: str
        seed = y.get('SEED', None)  # type: int
        self.seed = set_seed(seed)  # type: int
        default_device = 'cuda'
        self.device = y.get('DEVICE', default_device)  # type: str

        self.ds_path = y.get('DS_PATH','.') # type: str
        self.ds_path = Path(self.ds_path)

        self.input_shape = y.get('INPUT_SHAPE', (64,64,3))  # type: str
        self.input_shape = tuple([int(i) for i in self.input_shape.split(',')])

        self.export_tflite = y.get('EXPORT_TFLITE', True)  # type: bool
        assert self.export_tflite in [True, False], 'tflite export settings wrong, valid inputs are [True, False]'
        self.tflite_model_outpath = self.exp_weights_path.parent

    def write_to_file(self, out_file_path):
        # type: (str) -> None
        """
        Writes configuration parameters to `out_file_path`
        :param out_file_path: path of the output file
        """
        import re

        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        text = ansi_escape.sub('', str(self))
        with open(out_file_path, 'w') as out_file:
            print(text, file=out_file)


    def __str__(self):
        # type: () -> str
        out_str = ''
        for key in self.__dict__:
            if key in self.keys_to_hide:
                continue
            value = self.__dict__[key]
            if type(value) is Path or type(value) is str:
                value = value.replace(Conf.CAPRA_PATH, '$CAPRA_PATH')
                value = termcolor.colored(value, 'yellow')
            else:
                value = termcolor.colored(f'{value}', 'magenta')
            out_str += termcolor.colored(f'{key.upper()}', 'blue')
            out_str += termcolor.colored(': ', 'red')
            out_str += value
            out_str += '\n'
        return out_str[:-1]


    def no_color_str(self):
        # type: () -> str
        out_str = ''
        for key in self.__dict__:
            value = self.__dict__[key]
            if type(value) is Path or type(value) is str:
                value = value.replace(Conf.CAPRA_PATH, '$CAPRA_PATH')
            out_str += f'{key.upper()}: {value}\n'
        return out_str[:-1]


def show_default_params():
    """
    Print default configuration parameters
    """
    cnf = Conf(exp_name='default')
    print(f'\nDefault configuration parameters: \n{cnf}')


if __name__ == '__main__':
    show_default_params()
