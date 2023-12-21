"""Training few-shot LSE.
"""
import torch, argparse, os
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_logger
import torchvision.utils as vutils

from models.semantic_extractor import *
from models.helper import build_semantic_extractor, save_semantic_extractor
from models.stylegan2_generator import StyleGAN2Generator
from lib.visualizer import segviz_torch
from lib.callback import SEVisualizerCallback
from lib.evaluate import evaluate_SE, write_results
from lib.misc import set_cuda_devices, imread
from lib.dataset import NoiseDataModule
from lib.op import bu, sample_latent
from predictors.face_segmenter import FaceSegmenter


def get_features(synthesis, wp, resolution, P=None, is_large_mem=False):
    """Get features."""
    images, features, labels = [], [], []
    with torch.no_grad():
        for i in range(wp.shape[0]):
            image, feature = synthesis(wp[i : i + 1], generate_feature=True)
            if P:
                labels.append(P(image, size=resolution).long())
            if is_large_mem:
                feature = [f.cpu() for f in feature]
            features.append(feature)
            images.append(image)
    features = [
        torch.cat([feats[i] for feats in features]) for i in range(len(features[0]))
    ]
    images = bu(torch.cat(images), resolution)
    images = ((images.clamp(-1, 1) + 1) / 2).cpu()
    if P:
        labels = torch.cat(labels)
        return images, features, labels
    return images, features


class UpdateDataCallback(pl.Callback):
    """Update the data of few-shot trainer."""
    def __init__(self, wp=None, resolution=256, is_large_mem=False):
        super().__init__()
        self.resolution = resolution
        self.is_large_mem = is_large_mem
        self.wp = wp

    def on_epoch_end(self, trainer, pl_module):
        """Update on the epoch end."""
        del pl_module.feature
        images, features = get_features(
            pl_module.G.synthesis, self.wp, self.resolution, is_large_mem=self.is_large_mem
        )
        pl_module.feature = features


def main():
    """Entrance."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expr", type=str, default="expr/", help="The experiment directory"
    )
    parser.add_argument(
        "--gpu-id", type=str, default="0", help="GPUs to use. (default: %(default)s)"
    )
    parser.add_argument("--model", type=str, default="LSE", help="LSE, NSE-1")
    parser.add_argument(
        "--label-set", type=str, default="human", help="human or deeplabv3"
    )
    parser.add_argument(
        "--num-sample",
        type=int,
        default=1,
        help="The total number of few shot samples.",
    )
    parser.add_argument("--repeat-ind", type=int, default=0, help="The repeat index.")
    parser.add_argument("--seed", type=int, default=514, help="The repeat index.")
    parser.add_argument(
        "--G-name",
        type=str,
        default="stylegan2_ffhq",
        help="The model type of generator",
    )
    args = parser.parse_args()
    set_cuda_devices(args.gpu_id)

    is_large_mem = "ffhq" in args.G_name and args.num_sample >= 4
    is_face = "ffhq" in args.G_name
    n_class = 15 if is_face else 9
    resolution = 512
    DATA_DIR = f"data/{args.G_name}_fewshot"
    DIR = f"{args.expr}/fewshot_{args.label_set}/{args.G_name}_{args.model}_fewshot"
    prefix = f"{DIR}/{args.model}_r{args.repeat_ind}_n{args.num_sample}"
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    else:
        print("=> Found existing file, stop training.")
        exit()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    G = StyleGAN2Generator(model_name=args.G_name, randomize_noise=True)

    # get the dims and layers
    features = G(G.easy_sample(1))["feature"]
    G = G.net
    dims = [s.shape[1] for s in features]
    layers = list(range(len(dims)))
    logger = pl_logger.TensorBoardLogger(prefix)
    z = torch.randn(6, 512).cuda()
    train_wp, label = [], []
    for i in range(args.num_sample):
        st = args.num_sample * args.repeat_ind
        train_wp.append(torch.load(f"{DATA_DIR}/latent/wp_{st + i:02d}.npy"))
        label.append(
            imread(f"{DATA_DIR}/{args.label_set}_label/{st + i:02d}.png")[..., 0]
        )
    train_wp = torch.cat(train_wp).cuda()
    label = torch.from_numpy(np.stack(label)).long().cuda()
    if args.G_name == "stylegan2_car":
        zeros = torch.zeros(label.shape[0], 64, label.shape[2]).to(label)
        label = torch.cat([zeros, label, zeros], 1)
    print(label.min(), label.max())
    label_viz = segviz_torch(label)
    images, features = get_features(
        G.synthesis, train_wp, resolution, is_large_mem=is_large_mem
    )
    disp = []
    for i in range(images.shape[0]):
        disp.extend([images[i], label_viz[i]])
    vutils.save_image(torch.stack(disp), f"{prefix}_training_data.png", nrow=6)

    wp = sample_latent(G, 6, "trunc-wp")
    wp[:1] = train_wp[:1]
    udc = UpdateDataCallback(resolution=resolution, is_large_mem=is_large_mem)
    udc.wp = train_wp
    sevc = SEVisualizerCallback(wp, interval=1000)
    callbacks = [udc, sevc]
    dm = NoiseDataModule(train_size=128, batch_size=1)

    SE = build_semantic_extractor(
        lw_type="none", model_name=args.model, n_class=n_class, dims=dims, layers=layers
    ).cuda()
    SE.train()
    learner = SEFewShotLearner(
        model=SE,
        G=G,
        optim="adam-0.001",
        loss_type="normal",
        resolution=resolution,
        latent_strategy="trunc-wp",
        save_dir=prefix,
    )

    learner.feature, learner.label = features, label
    trainer = pl.Trainer(
        logger=logger,
        accumulate_grad_batches=args.num_sample,
        max_epochs=50,
        progress_bar_refresh_rate=1,
        log_every_n_steps=1,
        callbacks=callbacks,
        gpus=1,
    )

    torch.save(train_wp.cpu(), f"{prefix}/wp.pth")
    trainer.fit(learner, dm)
    disp = []
    SE = SE.cuda()
    with torch.no_grad():
        for i in range(features[0].shape[0]):
            feature = [f[i : i + 1].cuda() for f in features]
            label = SE(feature, size=resolution)[-1].argmax(1)
            disp.extend([images[i], label_viz[i]])
    vutils.save_image(torch.stack(disp), f"{prefix}/result.png", nrow=6)
    save_semantic_extractor(SE, f"{prefix}/model.pth")

    P = FaceSegmenter() if is_face else exit()
    mIoU, c_ious = evaluate_SE(SE, G, P, resolution, 10000, "trunc-wp")
    name = (
        f"{args.G_name}_{args.model}_r{args.repeat_ind}_n{args.num_sample}_elstrunc-wp"
    )
    write_results(f"results/fewshot_real_extractor/{name}.txt", mIoU, c_ious)


if __name__ == "__main__":
    main()
