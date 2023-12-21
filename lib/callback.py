import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from lib.visualizer import viz_SE, segviz_torch
from lib.op import *


class CheckpointCallback(pl.Callback):
  def __init__(self, ckpt_func, interval):
    self.func = ckpt_func
    self.interval = interval
  
  def on_batch_end(self, trainer, pl_module):
    if trainer.global_step % self.interval == 0:
      print(f"=> Saving model on {trainer.global_step}")
      self.func(pl_module)


class WeightVisualizerCallback(pl.Callback):
  def __init__(self, size=200, interval=100):
    self.history = []
    self.size = size
    self.count = 0
    self.interval = interval

  def on_batch_end(self, trainer, pl_module):
    self.count += 1
    if self.count % self.interval != 0:
      return
    weight = pl_module.model._calc_layer_weight().detach().cpu().numpy()
    self.history.append(weight)
    if len(self.history) > self.size:
      del self.history[0]
    
    N, M = len(self.history), weight.shape[0]
    data = np.stack(self.history)
    cdata = np.cumsum(data, 1)
    fig = plt.figure(figsize=(10, 5))
    for i in range(M):
      plt.bar(range(N), data[:, i],
        bottom=None if i == 0 else cdata[:, i - 1])
      plt.legend([f"layer {i}" for i in range(M)])
    trainer.logger.experiment.add_figure(f"Layer Weight",
      fig, self.count // self.interval)
    plt.close()


class ImageVisualizerCallback(pl.Callback):
  def __init__(self, z):
    super().__init__()
    self.z = z
    self.shape = self.z.shape
    self.count = 0
  
  def on_batch_end(self, trainer, pl_module):
    self.count += 1
    if self.count % 1000 != 0:
      return
    tensorboard = trainer.logger.experiment
    with torch.no_grad():
      r, _ = pl_module(self.z.view(*self.shape), reverse=True)
    if r.shape[1] == 1:
      r = r.repeat(1, 3, 1, 1)
    disp_images = vutils.make_grid(r, nrow=4)
    tensorboard.add_image('generated samples',
      disp_images.clamp(0, 1), global_step=self.count // 1000)


class TrainingEvaluationCallback(pl.Callback):
  def __init__(self):
    super().__init__()
    self.count = 0
    self.vals = []
  
  def on_epoch_end(self, trainer, pl_module):
    self.count += 1
    tensorboard = trainer.logger.experiment
    table = pl_module.train_evaluation
    pixelacc = torch.Tensor([entry[0] for entry in table])
    pixelacc = pixelacc.mean()
    IoU = torch.stack([entry[1] for entry in table], 1) # IoU: (C, N)
    c_IoU = torch.zeros((IoU.shape[0]))

    for i in range(IoU.shape[0]):
      v = IoU[IoU > -0.1]
      c_IoU[i] = -1 if len(v) == 0 else v.mean()
      if hasattr(pl_module.P, "labels"):
        labels = pl_module.P.labels
        tensorboard.add_scalar(f'val/{labels[i]}_IoU',
          c_IoU[i], self.count)
    mIoU = c_IoU[c_IoU > -1].mean()
    tensorboard.add_scalar('val/mIoU', mIoU, self.count)
    tensorboard.add_scalar('val/pixelacc', pixelacc, self.count)
    self.vals.append([mIoU, pixelacc, c_IoU])
    torch.save(self.vals, pl_module.save_dir + "/train_evaluation.pth")
    pl_module.train_evaluation = []


class SEVisualizerCallback(pl.Callback):
  def __init__(self, wp, P=None, interval=1000):
    super().__init__()
    self.wp = wp
    self.P = P
    self.interval = interval
    self.count = 0

  def on_batch_end(self, trainer, pl_module):
    self.count += 1
    if self.count % self.interval != 0:
      return

    tensorboard = trainer.logger.experiment
    images, seg_vizs, label_vizs, layer_vizs, hmaps = viz_SE(
      pl_module.G, pl_module.model, self.P, self.wp, size=256)
    disp_layer = vutils.make_grid(torch.cat([
      layer_vizs[0], images[0:1]]), nrow=5)
    tensorboard.add_image('Layer-wise Semantics',
      disp_layer, global_step=self.count // self.interval)   
    if label_vizs is not None:
      disp = vutils.make_grid(torch.cat([images, seg_vizs, label_vizs]),
        nrow=images.shape[0])
      tensorboard.add_image('Image / Output / Label',
        disp, global_step=self.count // self.interval) 
      disp_hmap = vutils.make_grid(torch.cat([
        images[1:2], label_vizs[1:2], seg_vizs[1:2], hmaps[1]]), nrow=5)
      tensorboard.add_image('Class score heatmap',
        disp_hmap, global_step=self.count // self.interval)  
    else:
      disp = vutils.make_grid(torch.cat([images, seg_vizs]),
        nrow=images.shape[0])
      tensorboard.add_image('Image / Output',
        disp, global_step=self.count // self.interval) 
      disp_hmap = vutils.make_grid(torch.cat([
        images[1:2], seg_vizs[1:2], hmaps[1]]), nrow=5)
      tensorboard.add_image('Class score heatmap',
        disp_hmap, global_step=self.count // self.interval)


class TrainSegVizCallback(pl.Callback):
  def __init__(self, ds, num=8, name="Train", interval=100):
    super().__init__()
    self.ds = ds
    self.idx = 0
    self.num = num
    self.name = name
    self.count = 0
    self.interval = interval

  def on_batch_end(self, trainer, pl_module):
    self.count += 1
    if self.count % self.interval != 0:
      return
    tensorboard = trainer.logger.experiment
    image_viz, pred_viz, label_viz = [], [], []
    for i in range(self.num):
      x, y = self.ds[self.idx]
      x, y = x.unsqueeze(0), y.unsqueeze(0)
      self.idx += 1
      with torch.no_grad():
        pred = pl_module(x.cuda()).argmax(1)
        pred_viz.append(segviz_torch(pred.cpu()))
      label_viz.append(segviz_torch(y))
      image_viz.append((x.clamp(-1, 1) + 1) / 2)
    image_viz = torch.cat(image_viz)
    pred_viz, label_viz = torch.cat(pred_viz), torch.cat(label_viz)
    disp = vutils.make_grid(
      torch.cat([image_viz, label_viz, pred_viz]),
      nrow=image_viz.shape[0])
    tensorboard.add_image(f'{self.name} / Image / Label / Pred',
      disp, global_step=self.count)   



class SegVizCallback(pl.Callback):
  def __init__(self, image, label, name="Validation", interval=100):
    super().__init__()
    self.image = image
    self.name = name
    self.image_viz = ((image.clamp(-1, 1) + 1) / 2).cpu()
    self.label_viz = segviz_torch(label)
    self.count = 0
    self.interval = interval

  def on_batch_end(self, trainer, pl_module):
    self.count += 1
    if self.count % self.interval != 0:
      return
    tensorboard = trainer.logger.experiment
    pred_viz = []
    for i in range(self.image.shape[0]):
      with torch.no_grad():
        pred = pl_module.model(self.image[i:i+1].cuda())
        pred_viz.append(segviz_torch(pred.argmax(1).cpu()))
    pred_viz = torch.cat(pred_viz)
    disp = vutils.make_grid(
      torch.cat([self.image_viz, self.label_viz, pred_viz]),
      nrow=self.image.shape[0])
    tensorboard.add_image(f'{self.name} / Image / Label / Pred',
      disp, global_step=self.count)   


class DualSegVizCallback(pl.Callback):
  def __init__(self, latent, interval=100):
    super().__init__()
    self.latent = latent
    self.count = 0
    self.interval = interval

  def on_batch_end(self, trainer, pl_module):
    self.count += 1
    if self.count % self.interval != 0:
      return
    tensorboard = trainer.logger.experiment
    with torch.no_grad():
      image, G_seg, P_seg = pl_module(self.latent)
    image = (image.clamp(-1, 1) + 1) / 2
    G_viz = segviz_torch(G_seg.argmax(1))
    P_viz = segviz_torch(P_seg.argmax(1))
    disp = vutils.make_grid(
      torch.cat([image.cpu(), G_viz, P_viz]),
      nrow=image.shape[0])
    tensorboard.add_image('Image / G Pred / P Pred',
      disp, global_step=self.count // self.interval)
