"""Train an image segmentation network (DeepLabV3 with ResNet50 backbone, no ImageNet pretraining) on a dataset.
"""
import argparse
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_logger
from pytorch_lightning.callbacks import LearningRateMonitor

from lib.callback import SegVizCallback, TrainSegVizCallback
from lib.misc import set_cuda_devices
from predictors.helper import build_predictor
from lib.learner import ImageSegmentationLearner
from lib.dataset import ImageSegmentationDataset, CelebAMaskDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/CelebAMask-HQ",
        help="The experiment directory.",
    )
    parser.add_argument(
        "--label-set", type=str, default="c15", help="The experiment directory."
    )
    parser.add_argument(
        "--expr", type=str, default="expr/image_seg", help="The experiment directory."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="The initial learning rate."
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Image resize.")
    parser.add_argument("--n-epoch", type=int, default=20, help="The number of epoches")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="GPUs to use. (default: %(default)s)",
    )
    args = parser.parse_args()
    n_gpu = set_cuda_devices(args.gpu_id)

    ind = args.label_set.rfind("c")
    n_class = int(args.label_set[ind + 1 :])
    name = args.data_dir[args.data_dir.rfind("/") + 1 :]
    log_dir = f"{args.expr}/{name}/{args.label_set}"
    logger = pl_logger.TensorBoardLogger(log_dir)
    net = build_predictor("deeplabv3+", n_class=n_class)
    learner = ImageSegmentationLearner(
        net, args.lr, reduce_lr_at_epoch=args.n_epoch - 5
    )
    if "CelebAMask-HQ" in args.data_dir:
        dataset_class = CelebAMaskDataset
    else:
        dataset_class = ImageSegmentationDataset
    train_ds = dataset_class(
        args.data_dir,
        image_folder="image",
        label_folder=f"label_{args.label_set}",
        use_split="train",
    )
    val_ds = dataset_class(
        args.data_dir,
        image_folder="image",
        label_folder=f"label_{args.label_set}",
        use_split="val",
    )
    train_dl = DataLoader(
        train_ds, args.batch_size, shuffle=False, num_workers=16
    )  # The shuffling is done in dataset
    val_dl = DataLoader(
        val_ds, args.batch_size, shuffle=False, num_workers=0
    )  # Make sure deterministic behavior
    for (
        val_image,
        val_label,
    ) in val_dl:  # Take a single batch out from val for visualization
        break
    seg_cb = SegVizCallback(val_image[:10].cpu(), val_label[:10].cpu(), interval=1000)
    tseg_cb = TrainSegVizCallback(train_ds, interval=1000)
    lr_monitor = LearningRateMonitor("step")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.n_epoch,
        log_every_n_steps=5,
        progress_bar_refresh_rate=1,
        callbacks=[seg_cb, tseg_cb, lr_monitor],
        gpus=n_gpu,
    )
    trainer.fit(learner, train_dl, val_dl)
    torch.save(net.state_dict(), f"{log_dir}/deeplabv3+_c{n_class}.pth")
