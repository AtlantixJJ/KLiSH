import torch
import argparse, math, glob, sys
sys.path.insert(0, ".")
sys.path.insert(0, "thirdparty/pigan")
import numpy as np
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

import curriculums, generators
sys.modules['generators'] = generators # pickle migrating line


class ImplicitGenerator(torch.nn.Module):
  def __init__(self, name="ICelebA", size=256,
                ray_step_multiplier=2,
                lock_view_dependence=True):
    super().__init__()
    self.name = name
    if name == "ICelebA":
      self.config = curriculums.CelebA
      self.config['num_steps'] = self.config[0]['num_steps'] * ray_step_multiplier
      self.config['psi'] = 0.7
      self.config['lock_view_dependence'] = lock_view_dependence
      self.config['last_back'] = True
      self.config['nerf_noise'] = 0
      self.config = {key: value for key, value in self.config.items()
        if type(key) is str}
      self.net = torch.load("thirdparty/pigan/pretrained/CelebA/generator.pth")
      ema = torch.load("thirdparty/pigan/pretrained/CelebA/ema.pth")
      ema.copy_to(self.net.parameters())
      self.net.eval()

  def forward(self, z, size=256, generate_feature=False):
    if generate_feature:
      features, pixels, self.depth_map = self.net.staged_forward_feature(
        z, img_size=size, **self.config)
      return pixels, features
    else:
      pixels, self.depth_map = self.net.staged_forward(
        z, img_size=size, **self.config)
      return pixels
  
  def set_device(self, device):
    self.net.device = device
    self.net.siren.device = device
    self.net = self.net.to(device)
    self.net.siren = self.net.siren.to(device)
    return self
