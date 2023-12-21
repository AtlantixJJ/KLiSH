"""Pytorch Lightning learner."""
from models.helper import build_generator
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import jaccard_index

from lib.evaluate import aggregate_iou
from lib.op import bu
from models.helper import generate_image, sample_latent
from models.helper import build_generator


class ImageSegmentationLearner(pl.LightningModule):
    """Learning a image segmentation network."""

    def __init__(self, model, lr, reduce_lr_at_epoch=1):
        super().__init__()
        self.model = model
        self.reduce_lr_at_epoch = reduce_lr_at_epoch
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, label = batch
        seg = self.model(image)
        ce_loss = F.cross_entropy(seg, label)
        return {"loss": ce_loss}

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"].mean()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        seg = self.model(image)
        IoU = jaccard_index(
            seg.argmax(1),
            label,
            absent_score=-1,
            reduction="none",
            num_classes=self.model.n_class,
        )
        return {"IoU": IoU}

    def validation_epoch_end(self, validation_step_outputs):
        N = len(validation_step_outputs)
        IoU = torch.zeros(N, self.model.n_class)
        for i in range(N):
            IoU[i] = validation_step_outputs[i]["IoU"]
        mIoU, cIoU = aggregate_iou(IoU)
        for i in range(cIoU.shape[0]):
            self.log(f"val/cIoU/{i:02d}", cIoU[i])
        self.log("val/mIoU", mIoU)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), self.lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=[self.reduce_lr_at_epoch]
        )
        return [optim], [sched]


class DualLearner(pl.LightningModule):
    def __init__(self, G_name, SE, P, resolution=256, batch_size=4, lr=0.001):
        super().__init__()
        self.G, self.SE, self.P = build_generator(G_name).net, SE, P
        self.resolution, self.batch_size = resolution, batch_size
        self.lr = lr

    def forward(self, latent):
        with torch.no_grad():
            image, feat = generate_image(self.G, latent, generate_feature=True)
            image = bu(image, self.resolution)
        G_seg = self.SE(feat, self.resolution)[-1]
        P_seg = self.P(image)
        return image, G_seg, P_seg

    def training_step(self, batch, batch_idx):
        _, G_seg, P_seg = self(sample_latent(self.G, self.batch_size))
        GtoP = F.cross_entropy(G_seg, P_seg.argmax(1).detach())
        PtoG = F.cross_entropy(P_seg, G_seg.argmax(1).detach())
        if batch_idx % 2 == 0:
            GtoP = GtoP.detach()
        else:
            PtoG = PtoG.detach()
        loss = GtoP + PtoG
        self.log("main/GtoP", GtoP)
        self.log("main/PtoG", PtoG)
        self.log("main/loss", PtoG)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), self.lr)
        return optim
