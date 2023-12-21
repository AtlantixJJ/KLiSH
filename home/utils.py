import time, math, glob, os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from base64 import b64encode, b64decode
from datetime import datetime
from scipy.stats import mode

from lib.visualizer import get_label_color, has_label_color, which_label
from lib.misc import imwrite


def to_serialized_tensor(x, dtype="float32"):
    """Deprecated."""
    np_arr = x.detach().cpu().numpy().astype(dtype)
    return b64encode(np_arr.tobytes()).decode("utf-8")


def from_serialized_tensor(s, dtype="float32"):
    """Deprecated."""
    np_arr = np.fromstring(b64decode(s), dtype=dtype)
    return torch.from_numpy(np_arr)


def to_compact_LSE(LSE):
    """Deprecated."""
    weight = torch.cat([l[0].weight[:, :, 0, 0] for l in LSE.extractor], 1)
    return {
        "n_class": LSE.n_class,
        "layers": LSE.layers,
        "dims": LSE.dims,
        "layer_weight": to_serialized_tensor(LSE.layer_weight),
        "weight": to_serialized_tensor(weight),
    }


def from_compact_LSE(LSE, cdict):
    """Deprecated."""
    LSE.requires_grad_(False)
    LSE.layer_weight.copy_(from_serialized_tensor(cdict["layer_weight"]))
    weight = from_serialized_tensor(cdict["weight"])
    count = 0
    for i, e in enumerate(LSE.extractor):
        e[:, :, 0, 0].copy_(weight[:, count : count + LSE.dims[i]])
        count += LSE.dims[i]
    LSE.requires_grad_(True)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def color_mask(image, color):
    r = image[:, :, 0] == color[0]  # np.abs(image[:, :, 0] - color[0]) < threshold
    g = image[:, :, 1] == color[1]  # np.abs(image[:, :, 1] - color[1]) < threshold
    b = image[:, :, 2] == color[2]  # np.abs(image[:, :, 2] - color[2]) < threshold
    return r & g & b


def resize_crop(arr, scale_size=None, crop_size=None):
    if scale_size is not None:
        if type(scale_size) is int:
            scale_size = (scale_size, scale_size)
        arr = imresize(arr, scale_size)
    else:
        scale_size = (arr.shape[0], arr.shape[1])

    if crop_size is not None:
        H, W = scale_size
        nH, nW = crop_size
        xc, yc = H // 2, W // 2
        xd, yd = nH // 2, nW // 2
        arr = arr[xc - xd : xc + xd, yc - yd : yc + yd]
    return arr


def preprocess_label(arr, n_class, scale_size=None, crop_size=None):
    """Convert color-based label image into categorical labels."""
    arr = resize_crop(arr, scale_size, crop_size)
    size = arr.shape[:2]
    x = torch.from_numpy(arr)
    t = torch.zeros(size)
    for i in range(n_class):
        c = get_label_color(i)
        t[color_mask(x, c)] = i
    return t.unsqueeze(0)


def preprocess_image(arr, scale_size=None, crop_size=None):
    """arr in [0, 255], shape (H, W, C)"""
    arr = resize_crop(arr, scale_size, crop_size)
    size = arr.shape[:2]
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    t = (t - 127.5) / 127.5
    if size is not None:
        t = F.interpolate(t, size=size, mode="bilinear", align_corners=True)
    return t


def preprocess_mask(mask, size=None):
    """
    mask in [0, 255], shape (H, W)
    """
    mask = imresize(mask, size)
    t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    t = t / 255.0
    return t


def imresize(image, size, resample=Image.NEAREST):
    return np.array(Image.fromarray(image).resize(size, resample))


def neigborhood_voting(image_arr, i, j):
    """The color of the neighborhood of a pixel."""
    ys = []
    h, w = image_arr.shape[:2]
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if i + di >= 0 and j + dj >= 0 and i + di < h and j + dj < w:
                ncolor = image_arr[i + di, j + dj]
                if has_label_color(ncolor):
                    ys.append(which_label(ncolor))
    return mode(ys).mode[0]


def stroke2array(image, target_size=None):
    """Convert a stroke drawing image to its original color (due to compression loss) and the mask"""
    image = image.convert("RGBA")
    if target_size is not None:
        image = image.resize(target_size)
    w, h = image.size
    image_arr = np.asarray(image)
    origin = np.zeros([h, w, 3], dtype="uint8")
    mask = np.zeros([h, w], dtype="uint8")
    for i in range(h):
        for j in range(w):
            color = image_arr[i, j]
            masked = color[3] >= 255
            if masked and not has_label_color(color):
                # find an substitute from neighborhood voting
                color = get_label_color(neigborhood_voting(image_arr, i, j))
            origin[i, j] = color[:3]
            mask[i, j] = int(masked) * 255
    return origin, mask


def get_time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def save_plot_with_time(dirname, name):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fpath = os.path.join(dirname, "%s_%s.png" % (time_str, name))
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()


def copy_tensor(dst, src):
    dst.requires_grad = False
    dst.copy_(src)
    dst.requires_grad = True


def color_mask_tensor(image, color):
    r = image[0, :, :] == color[0]
    g = image[1, :, :] == color[1]
    b = image[2, :, :] == color[2]
    return r & g & b


def celeba_rgb2label(image):
    t = torch.zeros(image.shape[1:]).float()
    for i, c in enumerate(CELEBA_COLORS):
        t[color_mask_tensor(image, c)] = i
    return t


def rgb2label(image, color_list):
    t = torch.zeros(image.shape[1:]).float()
    for i, c in enumerate(color_list):
        t[color_mask_tensor(image, c)] = i
    return t


class Timer(object):
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


def list_collect_data(
    data_dir,
    keys=[
        "origin_latent",
        "origin_noise",
        "image_stroke",
        "image_mask",
        "label_stroke",
        "label_mask",
    ],
):
    dic = {}
    for key in keys:
        keyfiles = glob.glob(f"{data_dir}/*{key}*")
        keyfiles.sort()
        dic[key] = keyfiles
    return dic


def plot_dic(dic, title="", file=None):
    n = len(dic.items())
    edge = int(math.sqrt(n))
    if edge**2 < n:
        edge += 1
    fig = plt.figure(figsize=(4 * edge, 3 * edge))
    for i, (k, v) in enumerate(dic.items()):
        ax = fig.add_subplot(edge, edge, i + 1)
        ax.plot(v)
        ax.set_title(k)
    if len(title) > 0:
        plt.suptitle(title)
    plt.tight_layout()
    if file is not None:
        plt.savefig(file, bbox_inches="tight")
        plt.close()


def plot_heatmap(dic, title="", file=None):
    n = len(dic.items())
    edge = int(math.sqrt(n))
    if edge**2 < n:
        edge += 1
    fig = plt.figure(figsize=(3 * edge, 3 * edge))
    for i, (k, v) in enumerate(dic.items()):
        ax = fig.add_subplot(edge, edge, i + 1)
        ax.imshow(v)
        ax.set_title(k)
    if len(title) > 0:
        plt.suptitle(title)
    if file is not None:
        plt.savefig(file, bbox_inches="tight")
        plt.close()


def window_sum(arr, size=10):
    """
    Args:
      arr : 1D numpy array
    """
    cumsum = np.cumsum(arr)
    windowsum = np.zeros_like(cumsum)
    windowsum[:size] = cumsum[:size]
    windowsum[size:] = cumsum[size:] - cumsum[:-size]
    return windowsum
