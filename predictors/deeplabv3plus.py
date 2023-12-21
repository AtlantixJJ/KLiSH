# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import sys
sys.path.insert(0, ".")
import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from lib.sync_batchnorm import SynchronizedBatchNorm2d
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
      'resnet152', 'BasicBlock', 'Bottleneck']
bn_mom = 0.1
model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': '~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth',
  'resnet101': '/home/wangyude/.cache/torch/checkpoints/resnet101s-03a0f310.pth',
  'resnet152': '~/.cache/torch/checkpoints/resnet152s-36670e8b.pth',
  #'resnet50': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet50s-a75c83cf.zip',
  #'resnet101': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet101s-03a0f310.zip',
  #'resnet152': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet152s-36670e8b.zip'
}
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  """ResNet BasicBlock
  """
  expansion = 1
  def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
         norm_layer=None):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation, bias=False)
    self.bn1 = norm_layer(planes, momentum=bn_mom, affine=True)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                padding=previous_dilation, dilation=previous_dilation, bias=False)
    self.bn2 = norm_layer(planes, momentum=bn_mom, affine=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  """ResNet Bottleneck
  """
  # pylint: disable=unused-argument
  expansion = 4
  def __init__(self, inplanes, planes, stride=1, dilation=1,
         downsample=None, previous_dilation=1, norm_layer=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = norm_layer(planes, momentum=bn_mom, affine=True)
    self.conv2 = nn.Conv2d(
      planes, planes, kernel_size=3, stride=stride,
      padding=dilation, dilation=dilation, bias=False)
    self.bn2 = norm_layer(planes, momentum=bn_mom, affine=True)
    self.conv3 = nn.Conv2d(
      planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = norm_layer(planes * 4, momentum=bn_mom, affine=True)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.dilation = dilation
    self.stride = stride

  def _sum_each(self, x, y):
    assert(len(x) == len(y))
    z = []
    for i in range(len(x)):
      z.append(x[i]+y[i])
    return z

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

  Parameters
  ----------
  block : Block
    Class for the residual block. Options are BasicBlockV1, BottleneckV1.
  layers : list of int
    Numbers of layers in each block
  classes : int, default 1000
    Number of classification classes.
  dilated : bool, default False
    Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
    typically used in Semantic Segmentation.
  norm_layer : object
    Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    for Synchronized Cross-GPU BachNormalization).

  Reference:

    - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

    - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
  """
  # pylint: disable=unused-variable
  def __init__(self, block, layers, dilated=True, multi_grid=False,
         deep_base=True, norm_layer=nn.BatchNorm2d):
    self.inplanes = 128 if deep_base else 64
    super(ResNet, self).__init__()
    if deep_base:
      self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
        norm_layer(64, momentum=bn_mom, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        norm_layer(64, momentum=bn_mom, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
      )
    else:
      self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                  bias=False)
    self.bn1 = norm_layer(self.inplanes, momentum=bn_mom, affine=True)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
    if dilated:
      self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                      dilation=2, norm_layer=norm_layer)
      if multi_grid:
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                        dilation=4, norm_layer=norm_layer,
                        multi_grid=True)
      else:
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                        dilation=4, norm_layer=norm_layer)
    else:
      self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                      norm_layer=norm_layer)
      self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                      norm_layer=norm_layer)
    self.OUTPUT_DIM = 2048
    self.MIDDLE_DIM = 256

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, norm_layer):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
             kernel_size=1, stride=stride, bias=False),
        norm_layer(planes * block.expansion, momentum=bn_mom, affine=True),
      )

    layers = []
    #multi_dilations = [4, 8, 16]
    multi_dilations = [3, 4, 5]
    if multi_grid:
      layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
    elif dilation == 1 or dilation == 2:
      layers.append(block(self.inplanes, planes, stride, dilation=1,
                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
    elif dilation == 4:
      layers.append(block(self.inplanes, planes, stride, dilation=2,
                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
    else:
      raise RuntimeError("=> unknown dilation size: {}".format(dilation))

    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      if multi_grid:
        layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                  previous_dilation=dilation, norm_layer=norm_layer))
      else:
        layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                  norm_layer=norm_layer))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    l1 = self.layer1(x)
    l2 = self.layer2(l1)
    l3 = self.layer3(l2)
    l4 = self.layer4(l3)
    return [l1, l2, l3, l4]



class ResNetReference(nn.Module):
  """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

  Parameters
  ----------
  block : Block
    Class for the residual block. Options are BasicBlockV1, BottleneckV1.
  layers : list of int
    Numbers of layers in each block
  classes : int, default 1000
    Number of classification classes.
  dilated : bool, default False
    Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
    typically used in Semantic Segmentation.
  norm_layer : object
    Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    for Synchronized Cross-GPU BachNormalization).

  Reference:

    - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

    - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
  """
  # pylint: disable=unused-variable
  def __init__(self, block, layers, dilated=True, multi_grid=False,
         deep_base=True, norm_layer=nn.BatchNorm2d):
    self.inplanes = 128 if deep_base else 64
    super(ResNet, self).__init__()
    if deep_base:
      self.conv1 = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
      )
    else:
      self.conv1 = nn.Conv2d(1, 1, kernel_size=7, stride=2, padding=3,
                  bias=False)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 1, layers[0], norm_layer=norm_layer)
    self.layer2 = self._make_layer(block, 1, layers[1], stride=2, norm_layer=norm_layer)
    if dilated:
      self.layer3 = self._make_layer(block, 1, layers[2], stride=1,
                      dilation=2, norm_layer=norm_layer)
      if multi_grid:
        self.layer4 = self._make_layer(block, 1, layers[3], stride=1,
                        dilation=4, norm_layer=norm_layer,
                        multi_grid=True)
      else:
        self.layer4 = self._make_layer(block, 1, layers[3], stride=1,
                        dilation=4, norm_layer=norm_layer)
    else:
      self.layer3 = self._make_layer(block, 1, layers[2], stride=2,
                      norm_layer=norm_layer)
      self.layer4 = self._make_layer(block, 1, layers[3], stride=2,
                      norm_layer=norm_layer)
    self.OUTPUT_DIM = 1
    self.MIDDLE_DIM = 1

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, norm_layer):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
             kernel_size=1, stride=stride, bias=False),
        norm_layer(planes * block.expansion, momentum=bn_mom, affine=True),
      )

    layers = []
    #multi_dilations = [4, 8, 16]
    multi_dilations = [3, 4, 5]
    if multi_grid:
      layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
    elif dilation == 1 or dilation == 2:
      layers.append(block(self.inplanes, planes, stride, dilation=1,
                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
    elif dilation == 4:
      layers.append(block(self.inplanes, planes, stride, dilation=2,
                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
    else:
      raise RuntimeError("=> unknown dilation size: {}".format(dilation))

    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      if multi_grid:
        layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                  previous_dilation=dilation, norm_layer=norm_layer))
      else:
        layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                  norm_layer=norm_layer))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    l1 = self.layer1(x)
    l2 = self.layer2(l1)
    l3 = self.layer3(l2)
    l4 = self.layer4(l3)
    return [l1, l2, l3, l4]


def resnet18(pretrained=False, **kwargs):
  """Constructs a ResNet-18 model.

  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(
      model_urls['resnet18']), strict=False)
  model.OUTPUT_DIM = 512
  model.MIDDLE_DIM = 64
  return model


def resnet34(pretrained=False, **kwargs):
  """Constructs a ResNet-34 model.

  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(
      model_urls['resnet34']), strict=False)
  return model


def resnet50(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.

  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    #old_dict = model_zoo.load_url(model_urls['resnet50'])
    #model_dict = model.state_dict()
    #old_dict = {k: v for k,v in old_dict.items() if (k in model_dict)}
    #model_dict.update(old_dict)
    #model.load_state_dict(model_dict) 
    #print('%s loaded.'%model_urls['resnet50'])
    model.load_state_dict(model_zoo.load_url(
      model_urls['resnet50']), strict=False)
  return model


def resnet101(pretrained=False, **kwargs):
  """Constructs a ResNet-101 model.

  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
  if pretrained:
    old_dict = torch.load(model_urls['resnet101'])
    model_dict = model.state_dict()
    old_dict = {k: v for k,v in old_dict.items() if (k in model_dict)}
    model_dict.update(old_dict)
    model.load_state_dict(model_dict) 
    print('%s loaded.'%model_urls['resnet101'])
  return model


def resnet152(pretrained=False, **kwargs):
  """Constructs a ResNet-152 model.

  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
  if pretrained:
    old_dict = model_zoo.load_url(model_urls['resnet152'])
    model_dict = model.state_dict()
    old_dict = {k: v for k,v in old_dict.items() if (k in model_dict)}
    model_dict.update(old_dict)
    model.load_state_dict(model_dict) 
    print('%s loaded.'%model_urls['resnet152'])
  return model


def build_backbone(name, pretrained, **kwargs):
  POOL = {
    "resnet50" : resnet50,
    "resnet34" : resnet34,
    "resnet18" : resnet18}
  return POOL[name](pretrained, deep_base=False, **kwargs)


class ASPP(nn.Module):

  def __init__(self, dim_in, dim_out, rate=[1,6,12,18], bn_mom=0.1, has_global=True, batchnorm=SynchronizedBatchNorm2d):
    super(ASPP, self).__init__()
    self.dim_in = dim_in
    self.dim_out = dim_out
    self.has_global = has_global
    if rate[0] == 0:
      self.branch1 = nn.Sequential(
          nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=1,bias=False),
          batchnorm(dim_out, momentum=bn_mom, affine=True),
          nn.ReLU(inplace=True),
      )
    else:
      self.branch1 = nn.Sequential(
          nn.Conv2d(dim_in, dim_out, 3, 1, padding=rate[0], dilation=rate[0],bias=False),
          batchnorm(dim_out, momentum=bn_mom, affine=True),
          nn.ReLU(inplace=True),
      )
    self.branch2 = nn.Sequential(
        nn.Conv2d(dim_in, dim_out, 3, 1, padding=rate[1], dilation=rate[1],bias=False),
        batchnorm(dim_out, momentum=bn_mom, affine=True),
        nn.ReLU(inplace=True),
    )
    self.branch3 = nn.Sequential(
        nn.Conv2d(dim_in, dim_out, 3, 1, padding=rate[2], dilation=rate[2],bias=False),
        batchnorm(dim_out, momentum=bn_mom, affine=True),
        nn.ReLU(inplace=True),
    )
    self.branch4 = nn.Sequential(
        nn.Conv2d(dim_in, dim_out, 3, 1, padding=rate[3], dilation=rate[3],bias=False),
        batchnorm(dim_out, momentum=bn_mom, affine=True),
        nn.ReLU(inplace=True),
    )
    if self.has_global:
      self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=False)
      self.branch5_bn = batchnorm(dim_out, momentum=bn_mom, affine=True)
      self.branch5_relu = nn.ReLU(inplace=True)
      self.conv_cat = nn.Sequential(
          nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=False),
          batchnorm(dim_out, momentum=bn_mom, affine=True),
          nn.ReLU(inplace=True),
          nn.Dropout(0.5)
      )
    else:
      self.conv_cat = nn.Sequential(
          nn.Conv2d(dim_out*4, dim_out, 1, 1, padding=0),
          batchnorm(dim_out, momentum=bn_mom, affine=True),
          nn.ReLU(inplace=True),
          nn.Dropout(0.5)
      )

  def forward(self, x):
    result = None
    [b,c,row,col] = x.size()
    conv1x1 = self.branch1(x)
    conv3x3_1 = self.branch2(x)
    conv3x3_2 = self.branch3(x)
    conv3x3_3 = self.branch4(x)
    if self.has_global:
      global_feature = F.adaptive_avg_pool2d(x, (1,1))
      global_feature = self.branch5_conv(global_feature)
      global_feature = self.branch5_bn(global_feature)
      global_feature = self.branch5_relu(global_feature)
      global_feature = F.interpolate(global_feature, (row,col), None, 'bilinear', align_corners=True)

      feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
    else:
      feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
    result = self.conv_cat(feature_cat)

    return result


class deeplabv3plus(nn.Module):
  def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
    super(deeplabv3plus, self).__init__()
    self.cfg = cfg
    self.batchnorm = batchnorm
    self.n_class = cfg.MODEL_NUM_CLASSES
    self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=cfg.MODEL_BACKBONE_PRETRAIN, norm_layer=self.batchnorm, **kwargs)
    input_channel = self.backbone.OUTPUT_DIM
    self.aspp = ASPP(dim_in=input_channel,
        dim_out=cfg.MODEL_ASPP_OUTDIM,
        rate=[0, 6, 12, 18],
        bn_mom = cfg.TRAIN_BN_MOM,
        has_global = cfg.MODEL_ASPP_HASGLOBAL,
        batchnorm = self.batchnorm)
    #self.dropout1 = nn.Dropout(0.5)

    indim = self.backbone.MIDDLE_DIM
    self.shortcut_conv = nn.Sequential(
        nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, 3, 1, padding=1, bias=False),
        batchnorm(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
        nn.ReLU(inplace=True),
    )
    self.cat_conv = nn.Sequential(
        nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=False),
        batchnorm(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
        nn.ReLU(inplace=True),
        #nn.Dropout(0.5),
        nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=False),
        batchnorm(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
        nn.ReLU(inplace=True),
        #nn.Dropout(0.1),
    )
    self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
    for m in self.modules():
      if m not in self.backbone.modules():
    #    if isinstance(m, nn.Conv2d):
    #      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, batchnorm):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
    if cfg.MODEL_FREEZEBN:
      self.freeze_bn()

  def forward(self, x, getf=False, interpolate=True):
    N,C,H,W = x.size()
    l1, l2, l3, l4 = self.backbone(x)
    feature_aspp = self.aspp(l4)
    #feature_aspp = self.dropout1(feature_aspp)

    feature_shallow = self.shortcut_conv(l1)
    n,c,h,w = feature_shallow.size()
    feature_aspp = F.interpolate(feature_aspp, (h, w),
      mode='bilinear', align_corners=True)

    feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
    feature = self.cat_conv(feature_cat)
    result = self.cls_conv(feature)
    result = F.interpolate(result, (H, W),
      mode='bilinear', align_corners=True)

    if getf:
      if interpolate:
        feature = F.interpolate(feature, (H, W),
          mode='bilinear', align_corners=True)
      return result, feature
    else:
      return result

  def freeze_bn(self):
    for m in self.modules():
      if isinstance(m, self.batchnorm):
        m.eval()
        
  def unfreeze_bn(self):
    for m in self.modules():
      if isinstance(m, self.batchnorm):
        m.train()


class deeplabv3plus2d(deeplabv3plus):
  def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
    super(deeplabv3plus2d, self).__init__(cfg, batchnorm=batchnorm, **kwargs)
    self.compress_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, 2, 1, 1, padding=0, bias=False)
    self.cls_conv = nn.Conv2d(2, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0, bias=False)
    for m in self.modules():
      if m not in self.backbone.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, batchnorm):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
    if cfg.MODEL_FREEZEBN:
      self.freeze_bn()

  def forward(self, x, getf=False, interpolate=True):
    N,C,H,W = x.size()
    l1, l2, l3, l4 = self.backbone(x)
    feature_aspp = self.aspp(l4)
    #feature_aspp = self.dropout1(feature_aspp)

    feature_shallow = self.shortcut_conv(l1)
    n,c,h,w = feature_shallow.size()
    feature_aspp = F.interpolate(feature_aspp,(h,w),mode='bilinear',align_corners=True)

    feature_cat = torch.cat([feature_aspp,feature_shallow],1)
    feature = self.cat_conv(feature_cat)
    feature = self.compress_conv(feature)
    result = self.cls_conv(feature)
    result = F.interpolate(result, (H,W), mode='bilinear',align_corners=True)

    if getf:
      if interpolate:
        feature = F.interpolate(feature, (H,W), mode='bilinear', align_corners=True)
      return result, feature
    else:
      return result


class deeplabv3plusInsNorm(deeplabv3plus):
  def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
    super(deeplabv3plusInsNorm, self).__init__(cfg, batchnorm, **kwargs)
    self.cat_conv = nn.Sequential(
        nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=False),
        nn.InstanceNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=False),
        nn.InstanceNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
        nn.ReLU(inplace=True),
    )
    for m in self.modules():
      if m not in self.backbone.modules():
        if isinstance(m, (batchnorm, nn.InstanceNorm2d)):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
    if cfg.MODEL_FREEZEBN:
      self.freeze_bn()


class deeplabv3plusAux(deeplabv3plus):
  def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
    super(deeplabv3plusAux, self).__init__(cfg, batchnorm, **kwargs)
    input_channel = self.backbone.OUTPUT_DIM
    self.seghead2 = nn.Sequential(
        nn.Conv2d(input_channel//4, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=False),
        batchnorm(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
    )
    self.seghead3 = nn.Sequential(
        nn.Conv2d(input_channel//2, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=False),
        batchnorm(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
    )
    self.seghead4 = nn.Sequential(
        nn.Conv2d(input_channel, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=False),
        batchnorm(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
    )
    #self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0, bias=False)
    for m in self.modules():
      if m not in self.backbone.modules():
    #    if isinstance(m, nn.Conv2d):
    #      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, batchnorm):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
    if cfg.MODEL_FREEZEBN:
      self.freeze_bn()

  def forward(self, x, getf=False, interpolate=True):
    N,C,H,W = x.size()
    l1, l2, l3, l4 = self.backbone(x)
    feature_aspp = self.aspp(l4)

    feature_shallow = self.shortcut_conv(l1)
    n,c,h,w = feature_shallow.size()
    feature_aspp = F.interpolate(feature_aspp,(h,w),mode='bilinear',align_corners=True)

    feature_cat = torch.cat([feature_aspp,feature_shallow],1)
    feature = self.cat_conv(feature_cat)
    result = self.cls_conv(feature)
    result = F.interpolate(result, (H,W), mode='bilinear',align_corners=True)

    seg2 = F.interpolate(self.seghead2(l2), (H,W), mode='bilinear', align_corners=True)
    seg3 = F.interpolate(self.seghead3(l3), (H,W), mode='bilinear', align_corners=True)
    seg4 = F.interpolate(self.seghead4(l4), (H,W), mode='bilinear', align_corners=True)

    if getf:
      if interpolate:
        feature = F.interpolate(feature, (H,W), mode='bilinear', align_corners=True)
      return [result, seg2, seg3, seg4], feature
    else:
      return [result, seg2, seg3, seg4]

  def orth_init(self):
    self.cls_conv.weight = torch.nn.Parameter(torch.eye(n=self.cfg.MODEL_NUM_CLASSES, m=self.cfg.MODEL_ASPP_OUTDIM).unsqueeze(-1).unsqueeze(-1))
    self.seghead2[-1].weight = torch.nn.Parameter(torch.eye(n=self.cfg.MODEL_NUM_CLASSES, m=self.cfg.MODEL_ASPP_OUTDIM).unsqueeze(-1).unsqueeze(-1))
    self.seghead3[-1].weight = torch.nn.Parameter(torch.eye(n=self.cfg.MODEL_NUM_CLASSES, m=self.cfg.MODEL_ASPP_OUTDIM).unsqueeze(-1).unsqueeze(-1))
    self.seghead4[-1].weight = torch.nn.Parameter(torch.eye(n=self.cfg.MODEL_NUM_CLASSES, m=self.cfg.MODEL_ASPP_OUTDIM).unsqueeze(-1).unsqueeze(-1))
    print('deeplabv3plusAux orth_init() finished')

  def orth_reg(self):
    module_list = [self.cls_conv, self.seghead2[-1], self.seghead3[-1], self.seghead4[-1]]
    loss_reg = 0
    for m in module_list:
      w = m.weight.squeeze(-1).squeeze(-1)
      w_norm = torch.norm(w, dim=1, keepdim=True)
      w = w/w_norm
      matrix = torch.matmul(w, w.transpose(0,1))
      loss_reg += torch.mean(matrix*(1-torch.eye(self.cfg.MODEL_NUM_CLASSES).to(0)))
    return loss_reg


class deeplabv3plusAuxSigmoid(deeplabv3plusAux):
  def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
    super(deeplabv3plusAuxSigmoid, self).__init__(cfg, batchnorm, **kwargs)
    for m in self.modules():
      if m not in self.backbone.modules() and isinstance(m, nn.ReLU):
        m = nn.Sigmoid()


class deeplabv3plusAuxReLUSigmoid(deeplabv3plusAux):
  def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
    super(deeplabv3plusAuxReLUSigmoid, self).__init__(cfg, batchnorm, **kwargs)
    for m in self.modules():
      if isinstance(m, nn.ReLU):
        m = nn.Sequential(
          nn.ReLU(inplace=True),
          nn.Sigmoid()
        )


class deeplabv3plusNorm(deeplabv3plus):
  def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
    super(deeplabv3plusNorm, self).__init__(cfg, batchnorm, **kwargs)
    self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0, bias=False)

  def forward(self, x, getf=False, interpolate=True):
    N,C,H,W = x.size()
    l1, l2, l3, l4 = self.backbone(x)
    feature_aspp = self.aspp(l4)
    #feature_aspp = self.dropout1(feature_aspp)

    feature_shallow = self.shortcut_conv(l1)
    n,c,h,w = feature_shallow.size()
    feature_aspp = F.interpolate(feature_aspp,(h,w),mode='bilinear',align_corners=True)

    feature_cat = torch.cat([feature_aspp,feature_shallow],1)
    feature = self.cat_conv(feature_cat)
    feature_norm = torch.norm(feature, dim=1, keepdim=True).detach()
    feature = feature/feature_norm
    conv_norm = torch.norm(self.cls_conv.weight, dim=1, keepdim=True).detach()
    conv_norm = conv_norm.permute(1,0,2,3)
    result = self.cls_conv(feature)/conv_norm

    result = F.interpolate(result, (H,W), mode='bilinear',align_corners=True)

    if getf:
      if interpolate:
        feature = F.interpolate(feature, (H,W), mode='bilinear', align_corners=True)
      return result, feature
    else:
      return result