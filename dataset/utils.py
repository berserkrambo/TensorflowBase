import numpy as np
import cv2
import numexpr as ne

def scale_coords(coords, meta):
    # Rescale coords1 (xyxy) from img1_shape to img0_shape
    coords *= 4
    img0_shape, img1_shape = meta["img0_shape"], meta["img1_shape"]
    gain = (max(img1_shape) / max(img0_shape))  # gain  = old / new
    coords[:, [0, 2]] -= meta["dw"]  # x padding
    coords[:, [1, 3]] -= meta["dh"]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clip(min=0)
    return coords

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


def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return int(max(0, min(r1, r2, r3)))


def get_2d_gaussian(shape, sigma=1):
    # type: ((int, int), float) -> np.ndarray
    """
    :param shape: output map shape
    :param sigma: Gaussian standard dev.
    :return: map with a Gaussian in the center
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    local_dict = {'x': x, 'y': y, 'sigma': sigma}
    h = ne.evaluate('exp(-(x**2 + y**2) / (2* (sigma**2)))', local_dict=local_dict)
    # alternative (with standard numpy): h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, sigma, peak_value=1):
    # type: (np.ndarray, (int, int), float, float) -> None
    """
    Draw a Gaussian (on `heatmap`) centered in `center` with the required sigma value.
    NOTE: inplace function --> `heatmap` will be modified!

    :param heatmap: map on which to draw the Gaussian; shape (H, W)
    :param center: Gaussian center
    :param sigma: Gaussian standard deviation
    :param peak_value: Gaussian peak value (default 1)
    """
    diameter = sigma * 6 + 1
    radius = (diameter - 1) // 2
    gaussian = get_2d_gaussian((diameter, diameter), sigma=sigma)

    x, y = center
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * peak_value, out=masked_heatmap)