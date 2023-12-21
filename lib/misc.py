"""Miscellaneous utilities."""
import os
import torch
import numpy as np
from PIL import Image
from threading import Thread
from collections import OrderedDict


class DictRecorder(OrderedDict):
    """Accumulate values to list in a dictionary."""

    def __init__(self):
        super().__init__()

    def add(self, key, val):
        """Accumulate val into key (a list)."""
        if key not in self:
            self.update({key: []})
        if type(val) is torch.Tensor:
            self[key].append(val.detach().cpu().numpy())
        else:
            self[key].append(val)


class GeneralThread(Thread):
    """Function interface threading."""

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args, self.kwargs = args, kwargs

    def run(self):
        self.res = self.func(*self.args, **self.kwargs)


### Environment ###


def set_cuda_devices(device_ids, use_cuda=True):
    """Sets visible CUDA devices.

    Example:

    set_cuda_devices('0,1', True)  # Enable device 0 and 1.
    set_cuda_devices('3', True)  # Enable device 3 only.
    set_cuda_devices('all', True)  # Enable all devices.
    set_cuda_devices('-1', True)  # Disable all devices.
    set_cuda_devices('0', False)  # Disable all devices.

    Args:
      devices_ids: A string, indicating all visible devices. Separated with comma.
        To enable all devices, set this field as `all`.
      use_cuda: Whether to use cuda. If set as False, all devices will be
        disabled. (default: True)
    """
    if not use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return 0
    assert isinstance(device_ids, str)
    if device_ids.lower() == "all":
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        return 8
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids.replace(" ", "")
    return len(device_ids.split(","))


### Data I/O ###


def read_ade20k_labels(fpath="figure/ade20k_labels.csv"):
    """Read label file for ADE20K dataset

    Args:
      fpath : The store location of the label file.
    Returns:
      A label list. Note that the default label has one more background category comparing to the official label.
    """
    lines = open(fpath, "r").readlines()
    label_list = [l.split(",")[-1].split(";")[0].strip() for l in lines[1:]]
    return label_list


def read_selected_labels(fpath="figure/selected_labels.csv"):
    """Read the labels for models.

    Args:
      fpath : The path to label file. Stardard format is one line for each model, with the first column being the model name.

    Returns:
      A dict object specifying the mapping from model names to category names.
    """
    dic = {}
    with open(fpath, "r") as f:
        for line in f.readlines():
            items = line.split(",")
            dic[items[0]] = [i.strip() for i in items[1:]]
    return dic


def imread(fpath):
    """Read image and returns a numpy array in [0, 255] scale."""
    with open(os.path.join(fpath), "rb") as f:
        return np.asarray(Image.open(f), dtype="uint8")


def imwrite(fpath, image, format="RGB"):
    """Write an numpy image to file.

    Args:
      image : an array of shape [H, W, 3] and scale in [0, 255].
    """
    if ".jpg" in fpath or ".jpeg" in fpath:
        ext = "JPEG"
    elif ".png" in fpath:
        ext = "PNG"
    else:
        print(f"!> Extension not found!")
    with open(os.path.join(fpath), "wb") as f:
        Image.fromarray(image.astype("uint8")).convert(format).save(f, format=ext)


### Evaluation Utilities ###


def invert_dic(dic):
    """Invert the key order of a dictionary."""
    idic = {}
    for k1 in dic.keys():
        for k2 in dic[k1].keys():
            if k2 not in idic:
                idic[k2] = {}
            idic[k2][k1] = dic[k1][k2]
    return idic


def max_key(dic):
    """Get the maximum key (when keys can be sorted)."""
    keys = list(dic.keys())
    ind = np.argmax([dic[k] for k in keys])
    return ind, keys[ind], dic[keys[ind]]


def print_table(t):
    """Print a table.
    Args:
      t : A 2D numpy array or a 2D list.
    """
    for row in t:
        s = ""
        for e in row:
            try:
                s += f"{e:.3f}\t"
            except:
                s += f"{e:.3f}\t"
        print(s)


def listkey_convert(name, listkey, output=None):
    """Check which key in listkey is a substring of name and return a value.

    Args:
      name : The raw string. It may contain one or more keys from listkey.
      listkey : A list of keys.
      output : When output is None, the matched key will be returned directly.
               When output is a list, the function will return the element of
               output at index of the matched key.
    Returns:
      A matched key, or the output word corresponding to the index of the
      matched key, or an empty string if matching fails
    """
    for i, key in enumerate(listkey):
        if key in name:
            if output is not None:
                return output[i]
            return key
    return ""


def aggregate_iou(res):
    """Aggregate IoU of each instance into a global mIoU and IoU.

    Args:
      res : The result. Assumed to be a list. Item 1 is pixel accuracy, item 2
            is IoU. -1 means the category is missing in both detection and GT.
    Returns:
      mIoU, class-wise IoU
    """
    ic_iou = torch.stack([r[1] for r in res])
    c_iou = torch.zeros(ic_iou.shape[1])
    for c in range(ic_iou.shape[1]):
        val = ic_iou[:, c]
        val = val[val > -0.1]
        c_iou[c] = -1 if val.shape[0] == 0 else val.mean()
    mIoU = c_iou[c_iou > -1].mean()
    return mIoU, c_iou


def formal_name(name):
    """Convert the naming in code to naming in paper."""
    if type(name) is list:
        return [formal_name(n) for n in name]
    finds = ["ada", "stylegan", "pggan", "bedroom", "church", "celebahq", "ffhq", "car", "cat", "dog", "wild", "metface"]
    subs = ["ADA", "StyleGAN", "PGGAN", "Bedroom", "Church", "CelebAHQ", "FFHQ", "Car", "Cat", "Dog", "Wild", "MetFace"]
    for find, sub in zip(finds, subs):
        name = name.replace(find, sub)
    name = name.replace("_", "-")
    return name


### Utilities for Making Tables ###


def get_dic_depth(dic):
    """Get the depth of a dict."""
    v = next(iter(dic.values()))
    count = 1
    while type(v) is dict:
        count += 1
        v = next(iter(v.values()))
    return count


def trim_dic(dic):
    """Remove empty key-value pairs."""
    for k in list(dic.keys()):
        if type(dic[k]) is dict:
            if len(dic[k]) == 0:
                del dic[k]
            else:
                trim_dic(dic[k])


def dic2table(dic, transpose=True):
    """Convert dict of depth 2 to latex table.

    Args:
      dic : In the form of dic[row_key][col_key].
      transpose : When True, the row_key of dic corresponds to the
                  col_key of the output table.
    """
    strs = []
    col_names = list(next(iter(dic.values())).keys())
    ncols = len(col_names) + 1
    strs.append([""] + col_names)
    for row_name, row_vals in dic.items():
        strs.append([row_name] + [row_vals[k] for k in col_names])
    if transpose:
        nstrs = [[] for _ in range(len(strs[0]))]
        for j in range(len(strs[0])):
            for i in range(len(strs)):
                nstrs[j].append(strs[i][j])
        return nstrs
    return strs


def str_table_single_std(dic, table_header="", output_std=True):
    """Convert a mean-std dictionary to a table."""
    row_names = list(dic.keys())
    col_names = list(dic[row_names[0]].keys())
    strs = [[table_header] + [str(c) for c in col_names]]
    for row_name in row_names:
        if len(dic[row_name]) == 0:
            continue
        s = [str(row_name)]
        for col_name in col_names:
            if len(dic[row_name][col_name]) == 0:
                continue
            mean = dic[row_name][col_name]["mean"]
            std = dic[row_name][col_name]["std"]
            if output_std:
                item_str = f"{mean * 100:.1f} $\\pm$ {std * 100:.1f}"
            else:
                item_str = f"{mean * 100:.1f}"
            s.append(item_str)
        strs.append(s)
    return strs


def str_table_single(dic, table_header="", to_percentage=True):
    """Convert a dictionary to a table."""
    row_names = list(dic.keys())
    col_names = list(dic[row_names[0]].keys())
    strs = [[table_header] + [str(c) for c in col_names]]
    for row_name in row_names:
        if len(dic[row_name]) == 0:
            continue
        s = [str(row_name)]
        for col_name in col_names:
            if to_percentage:
                item_str = f"{dic[row_name][col_name] * 100:.1f}"
            else:
                item_str = dic[row_name][col_name]
            s.append(item_str)
        strs.append(s)
    return strs



def str_table_singlestr_table_multiple(dic, indicate_best=False):  # group, G, method
    """Format a string array to latex table with multicolumn header."""
    groups = list(dic.keys())  # 1st column name

    def latex_header(n):
        return f"\\multicolumn" + "{" + str(n) + "}" + "{c|}"

    strs = [
        ["Generator"]
        + [
            f"{latex_header(len(dic[g].keys()))}" + "{" + formal_name(g) + "}"
            for g in groups
        ]
    ]
    s = ["Dataset"]
    for g in groups:
        Gs = list(dic[groups[0]].keys())  # 2nd column name
        s.extend(formal_name(Gs))
    strs.append(s)

    s_ = []
    for group in dic.keys():
        for ds in dic[group].keys():
            if indicate_best:
                best_ind, best_method, best_val = max_key(dic[group][ds])
                for i, method in enumerate(dic[group][ds].keys()):
                    acc = f"{dic[group][ds][method] * 100:.1f}"
                    comp = (dic[group][ds][method] - best_val) / best_val * 100
                    if best_method == method:
                        item_str = "\\textbf{" + acc + "}"
                    else:
                        item_str = f"{acc} ({comp:.1f})"
                    if len(s_) <= i:
                        s_.append([method])
                    s_[i].append(item_str)
            else:
                for i, method in enumerate(dic[group][ds].keys()):
                    item_str = f"{dic[group][ds][method] * 100:.1f}"
                    if len(s_) <= i:
                        s_.append([method])
                    s_[i].append(item_str)
    strs.extend(s_)
    return strs


def str_latex_table(strs):
    """Format a string table to a latex table.

    Args:
      strs : A 2D string table. Each item is a cell.
    Returns:
      A single string for the latex table.
    """
    for i in range(len(strs)):
        for j in range(len(strs[i])):
            if "_" in strs[i][j]:
                strs[i][j] = strs[i][j].replace("_", "-")

        ncols = len(strs[0])
        seps = "".join(["c" for i in range(ncols)])
        s = []
        s.append("\\begin{table}")
        s.append("\\centering")
        s.append("\\begin{tabular}{%s}" % seps)
        s.append(" & ".join(strs[0]) + " \\\\\\hline")
        for line in strs[1:]:
            s.append(" & ".join(line) + " \\\\")
        s.append("\\end{tabular}")
        s.append("\\end{table}")

        for i in range(len(strs)):
            for j in range(len(strs[i])):
                if "_" in strs[i][j]:
                    strs[i][j] = strs[i][j].replace("\\_", "_")

    return "\n".join(s)


def str_csv_table(strs):
    """Format a string table to a csv table."""
    s = []
    for i in range(len(strs)):
        s.append(",".join(strs[i]))
    return "\n".join(s)
