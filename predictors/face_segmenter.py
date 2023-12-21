# python 3.7
"""Segmenter for face."""

import os
import numpy as np

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .base_predictor import BasePredictor
from .face_segmenter_network import UNet
from .deeplabv3plus import deeplabv3plus

from lib import op

CELEBA_FULL_CATEGORY = [
    "background",
    "skin",
    "nose",
    "eye_g",
    "l_eye",
    "r_eye",
    "l_brow",
    "r_brow",
    "l_ear",
    "r_ear",
    "mouth",
    "u_lip",
    "l_lip",
    "hair",
    "hat",
    "ear_r",
    "neck_l",
    "neck",
    "cloth",
]

CELEBAMASK_NUMCLASS = 15
CELEBA_CATEGORY15 = [
    "bg",
    "skin",
    "nose",
    "eye_g",
    "eye",
    "brow",
    "ear",
    "mouth",
    "u_lip",
    "l_lip",
    "hair",
    "hat",
    "ear_r",
    "neck",
    "cloth",
]
[
    "car***bg",
    "car***front",
    "car***side",
    "car***back",
    "car***roof",
    "car***license_plate",
    "car***wheel",
    "car***light",
    "car***window",
]
CELEBA_COLORS = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
]


class Configuration:
    def __init__(self, config_dict, clear=True):
        self.__dict__ = config_dict
        self.clear = clear


class FaceSegmenter(BasePredictor):
    def __init__(self, model_name="", train_size=512):
        self.n_class = CELEBAMASK_NUMCLASS
        self.train_size = (train_size, train_size)
        self.labels = CELEBA_CATEGORY15
        super().__init__("face_seg")

    def build(self):
        # if self.model_name == "unet":
        #  self.net = UNet(resolution=self.resolution)
        # elif self.model_name == "deeplabv3+":
        default_dict = {
            "MODEL_BACKBONE": "resnet50",
            "MODEL_SHORTCUT_DIM": 48,
            "MODEL_ASPP_OUTDIM": 256,
            "MODEL_BACKBONE_PRETRAIN": False,
            "MODEL_ASPP_HASGLOBAL": False,
            "MODEL_NUM_CLASSES": 15,
            "MODEL_FREEZEBN": False,
            "TRAIN_BN_MOM": 0.0003,
        }
        self.net = deeplabv3plus(Configuration(default_dict))

    def load(self):
        # Load pre-trained weights.
        assert os.path.isfile(self.weight_path)
        self.net.load_state_dict(torch.load(self.weight_path))

    def raw_prediction(self, images, size=None):
        """
        Expecting torch.Tensor as input
        """
        images = op.bu(images, self.resolution)
        x = self.net(images.clamp(-1, 1))  # (N, M, H, W)
        if size:
            x = op.bu(x, size)
        return x

    def __call__(self, images, size=None):
        """
        Expecting torch.Tensor as input
        """
        orig_shape = images.shape[2:]
        images = op.bu(images, self.train_size)
        x = self.net(images.clamp(-1, 1))  # (N, M, H, W)
        if size:
            x = op.bu(x, size)
        else:
            x = op.bu(x, orig_shape)
        return x.argmax(1)  # (N, H, W)
