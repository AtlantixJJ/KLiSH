"""Draw the figure of USCS results from web collected masks."""
import sys, torch, argparse, pickle, glob, os

sys.path.insert(0, "thirdparty/spade")
sys.path.insert(0, ".")
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from pixmodels.pix2pix_model import Pix2PixModel
from lib.misc import set_cuda_devices, listkey_convert, imread
from lib.visualizer import segviz_torch


edit_file_dir = "data/web_collection/edit_select"


def load_opt(args):
    """Load the option file for SPADE model."""
    CKPT_DIR = f"thirdparty/spade/checkpoints/"
    Gn1, Gn2, serial, resolution = args.mtd.split("_")
    G_name = f"{Gn1}_{Gn2}"
    label_name = glob.glob(f"data/generated/{G_name}_s1113/label_{serial}_c*")[0]
    # n_class = int(label_name[label_name.rfind("_") + 2 :])

    opt = pickle.load(open(f"{CKPT_DIR}/{args.mtd}/opt.pkl", "rb"))
    opt.checkpoints_dir = CKPT_DIR
    opt.name = args.mtd
    opt.dataset_mode = "custom"
    opt.semantic_nc = opt.label_nc
    opt.gpu_ids = [0] if args.gpu_id != "-1" else []
    opt.batchSize = 1
    opt.isTrain = False
    opt.serial_batches = True

    opt.label_dir = label_name
    opt.image_dir = f"expr/data/{G_name}_s1113/image"
    return opt


def process(z_name, model, zs, cap_ds_name, label_vizs, images):
    """Process a single file."""
    z_name = z_name.strip()
    file_prefix = f"data/web_collection/{cap_ds_name}/{z_name}"
    label = torch.from_numpy(imread(f"{file_prefix}_label.png")[..., 0])
    label = label.long().unsqueeze(0).unsqueeze(0)

    label_viz = segviz_torch(label[0])
    label_vizs.append(label_viz.cpu())
    for j in range(args.R):
        image = model({"label": label, "z": zs[j : j + 1]}, mode="inference")
        image = (image.clamp(-1, 1) + 1) / 2
        if cap_ds_name == "MetFace":  # making the image look brighter
            image = image * 0.8 + 0.2
        images[j].append(image.cpu())


def save_results(images, label_vizs, count, t):
    """Save results to file."""
    disp = []
    for x, y in zip(images[0], label_vizs):
        disp.extend([y, x])
    disp = torch.cat(disp)
    D = 20
    disp = vutils.make_grid(disp, nrow=2, padding=D, pad_value=1)
    disp = disp[:, D:-D, D:-D]
    vutils.save_image(
        disp, f"results/plot/{args.mtd}_pix2pix_edit{count}{t}_vertical.png"
    )

    disp = label_vizs + images[0]
    disp = torch.cat(disp)
    disp = vutils.make_grid(disp, nrow=len(label_vizs), padding=D, pad_value=1)
    disp = disp[:, D:-D, D:-D]
    vutils.save_image(
        disp, f"results/plot/{args.mtd}_pix2pix_edit{count}{t}_horizontal.png"
    )


def worker():
    """Worker."""
    opt = load_opt(args)
    model = Pix2PixModel(opt)
    model.eval()
    ds_name = listkey_convert(
        args.mtd, ["metface", "car", "ffhq", "dog", "cat", "wild"]
    )
    cap_ds_name = listkey_convert(
        args.mtd,
        ["metface", "car", "ffhq", "dog", "cat", "wild"],
        ["MetFace", "Car", "Face", "Dog", "Cat", "Wild"],
    )
    for t in [""]:
        if t == "_final":
            args.R = 1
        fp = open(f"{edit_file_dir}/{ds_name}_uscs{t}.txt", "r", encoding="ascii")
        z_names = [l.strip() for l in fp.readlines()] + [""]
        zs = torch.randn(args.R, opt.z_dim).to(device)
        count = 0
        label_vizs, images = [], [[] for _ in range(args.R)]
        with torch.no_grad():
            for z_name in z_names:
                if len(z_name) == 0:
                    save_results(images, label_vizs, count, t)
                    zs = torch.randn(args.R, opt.z_dim).to(device)
                    count += 1
                    label_vizs, images = [], [[] for _ in range(args.R)]
                    continue
                process(z_name, model, zs, cap_ds_name, label_vizs, images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtd", default="all")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--R", type=int, default=1, help="The number of samples.")
    args = parser.parse_args()
    set_cuda_devices(args.gpu_id)
    device = "cuda" if args.gpu_id != "-1" else "cpu"

    if args.mtd == "all":
        for mtd in [
            "stylegan2_ffhq_klish_512",
            "stylegan2_car_klish_512",
            "ada_wild_klish_512",
            "ada_cat_klish_512",
            "ada_metface_klish_512",
            "ada_dog_klish_512",
        ]:
            args.mtd = mtd
            worker()
    else:
        worker()
