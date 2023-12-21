# python 3.7
"""Helper function to build predictor."""

import torch
from .predictor_settings import PREDICTOR_POOL
from .face_segmenter import FaceSegmenter
from .scene_segmenter import SceneSegmenter
from .scene_predictor import ScenePredictor
from .face_predictor import FacePredictor
from .feature_extractor import FeatureExtractor
from .deeplabv3plus import deeplabv3plus


__all__ = ['P_from_name', 'build_predictor', 'build_extractor']


def P_from_name(fpath):
  if "deeplabv3+" in fpath:
    state_dict = torch.load(fpath)
    n_class = int(fpath[fpath.rfind("c")+1:fpath.rfind(".")])
    net = build_predictor("deeplabv3+", n_class=n_class)
    net.load_state_dict(state_dict)
    return net
  raise NotImplementedError(f'Unsupported predictor `{fpath}`!')


class Configuration():
	def __init__(self, config_dict, clear=True):
		self.__dict__ = config_dict
		self.clear = clear


def build_predictor(predictor_name, **kwargs):
  """Builds predictor by predictor name."""
  if predictor_name == 'face_seg':
    return FaceSegmenter(predictor_name)  
  if predictor_name == 'scene_seg':
    return SceneSegmenter(predictor_name)
  if predictor_name == 'scene':
    return ScenePredictor(predictor_name)
  if predictor_name[:len('celebahq_')] == 'celebahq_':
    return FacePredictor(predictor_name)
  if "deeplabv3+" in predictor_name:
    default_dict = {
      #'MODEL_BACKBONE': 'resnet18',
      #'MODEL_SHORTCUT_DIM': 24,
      #'MODEL_ASPP_OUTDIM': 128,
      'MODEL_BACKBONE': 'resnet50',
      'MODEL_SHORTCUT_DIM': 48,
      'MODEL_ASPP_OUTDIM': 256,
      'MODEL_BACKBONE_PRETRAIN': False,
      'MODEL_ASPP_HASGLOBAL': False,
      'MODEL_NUM_CLASSES': kwargs["n_class"],
      'MODEL_FREEZEBN': False,
      'TRAIN_BN_MOM': 0.0003}
    return deeplabv3plus(Configuration(default_dict))
  raise NotImplementedError(f'Unsupported predictor `{predictor_name}`!')


def build_extractor(architecture, spatial_feature=False, imagenet_logits=False):
  """Builds feature extractor by architecture name."""
  if architecture not in PREDICTOR_POOL:
    raise ValueError(f'Feature extractor with architecture `{architecture}` is '
                     f'not registered in `PREDICTOR_POOL` in '
                     f'`predictor_settings.py`!')
  return FeatureExtractor(architecture,
                          spatial_feature=spatial_feature,
                          imagenet_logits=imagenet_logits)
