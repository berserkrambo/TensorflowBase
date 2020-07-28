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

def letterbox(img, new_shape=(416, 416), color=(128, 128, 128),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)