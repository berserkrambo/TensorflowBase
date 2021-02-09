# -*- coding: utf-8 -*-
# ---------------------

import matplotlib


matplotlib.use('Agg')

from matplotlib import figure
import numpy as np
import PIL
from PIL.Image import Image
from path import Path
from typing import *
import cv2

def imread(path):
    # type: (Union[Path, str]) -> Image
    """
    Reads the image located in `path`
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')

def imread_cv(path):
    # type: (Union[Path, str]) -> Image
    """
    Reads the image located in `path`
    :param path:
    :return:
    """
    img = cv2.imread(path)
    assert img is not None, 'img is None'
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pyplot_to_numpy(pyplot_figure):
    # type: (figure.Figure) -> np.ndarray
    """
    Converts a PyPlot figure into a NumPy array
    :param pyplot_figure: figure you want to convert
    :return: converted NumPy array
    """
    pyplot_figure.canvas.draw()
    x = np.fromstring(pyplot_figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    x = x.reshape(pyplot_figure.canvas.get_width_height()[::-1] + (3,))
    return x


def worker_init_fun(worker_id):
    return np.random.seed(worker_id)