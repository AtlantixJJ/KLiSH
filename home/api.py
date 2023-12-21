"""API for interaction between GAN and django views.
"""
import sys, pickle, json, torch, glob

sys.path.insert(0, ".")
sys.path.insert(0, "thirdparty/spade")
import numpy as np

from pixmodels.pix2pix_model import Pix2PixModel
from home.utils import *
from models.helper import *
from lib.op import torch2image, torch2numpy
from lib.misc import imwrite, imread
from lib.visualizer import segviz_numpy


def load_opt(mc):
    """Load options."""
    CKPT_DIR = mc["model_dir"]
    name = mc["model_name"]
    gpu_id = mc["gpu"]
    # n_class = mc["n_class"]

    Gn1, Gn2, serial, _ = name.split("_")
    G_name = f"{Gn1}_{Gn2}"
    label_name = glob.glob(f"data/generated/{G_name}_s1113/label_{serial}_c*")[0]

    opt = pickle.load(open(f"{CKPT_DIR}/{name}/opt.pkl", "rb"))
    opt.checkpoints_dir = CKPT_DIR
    opt.name = name
    opt.gpu_ids = [gpu_id] if gpu_id >= 0 else []
    opt.batchSize = 1
    # opt.label_nc = n_class
    opt.semantic_nc = opt.label_nc
    opt.isTrain = False
    opt.label_dir = label_name
    opt.image_dir = f"data/generated/{G_name}_s1113/image"
    return opt


class ModelAPI(object):
    """Initialize and manage models."""

    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self.config = json.load(f)
        self.init_model()

    def init_model(self):
        """Loading pretrained models."""
        self.pix2pix = {}
        self.model_config = {}
        for name, mc in self.config["models"].items():
            print(name, mc)
            self.model_config[name] = load_opt(mc)
            self.pix2pix[name] = Pix2PixModel(self.model_config[name])
            data_dir = mc["data_dir"]
            if not os.path.exists(f"{data_dir}/{name}"):
                os.makedirs(f"{data_dir}/{name}")


class EditAPI(object):
    """Respond to editing request."""

    def __init__(self, model_api):
        self.model_api = model_api

    def has_model(self, model_name):
        """Check if the model is present."""
        return model_name in self.model_api.model_config

    def generate_image_given_stroke(
        self, model_name, z, orig_label, label_stroke, label_mask
    ):
        """Generate image given stroke."""
        G = self.model_api.pix2pix[model_name]
        G_opt = self.model_api.model_config[model_name]
        G_config = self.model_api.config["models"][model_name]
        device = f"cuda:{G_opt.gpu_ids[0]}"

        if z is None:
            z = torch.randn(1, G_opt.z_dim)
        else:
            z = np.array(z, dtype=np.float32).reshape((1, G_opt.z_dim))
            z = torch.from_numpy(z).float()
        orig_label = np.array(orig_label, dtype=np.float32)
        time_str = get_time_str()
        data_dir = G_config["data_dir"]
        p = f"{data_dir}/{model_name}/{time_str}"
        np.save(f"{p}_z.npy", z)
        imwrite(f"{p}_label-stroke.png", label_stroke)
        imwrite(f"{p}_label-mask.png", label_mask)

        image_size = G_config["image_size"]
        label_stroke = preprocess_label(
            label_stroke, G_opt.label_nc, image_size
        ).unsqueeze(0)
        orig_label = torch.from_numpy(orig_label).view_as(label_stroke)
        label_mask = preprocess_mask(label_mask, image_size).squeeze(1)

        # if model_name == "Car":
        #    label_mask[:, :64] = label_mask[:, -64:] = 0
        #    label_stroke[:, :, :64] = label_stroke[:, :, -64:] = 0
        label = (orig_label * (1 - label_mask) + label_stroke * label_mask).long()
        label_arr = torch2numpy(label)
        label_viz = segviz_numpy(label_arr)
        with torch.no_grad():
            image = torch2image(
                G({"z": z.to(device), "label": label.to(device)}, mode="inference")
            )[0]

        # if model_name == "Car":
        #    alpha = np.ones((image.shape[0], image.shape[1], 1), dtype="uint8") * 255
        #    alpha[:64] = alpha[-64:] = 0
        #    image = np.concatenate([image, alpha], 2)
        #    label_viz = np.concatenate([label_viz, alpha], 2)

        imwrite(f"{p}_image.png", image)
        imwrite(f"{p}_label.png", label_arr[0, 0])
        imwrite(f"{p}_labelviz.png", label_viz)
        label_arr = label_arr.reshape(-1).tolist()
        z_arr = torch2numpy(z.view(-1)).tolist()
        return image, label_viz, label_arr, z_arr

    def generate_new_image(self, model_name):
        G = self.model_api.pix2pix[model_name]
        G_opt = self.model_api.model_config[model_name]
        device = f"cuda:{G_opt.gpu_ids[0]}"

        label_files = glob.glob(f"{G_opt.label_dir}/*.png")
        label_path = np.random.choice(label_files, (1,))[0]
        label = imread(label_path)[None, None, :, :, 0]
        label = torch.from_numpy(label).long().to(device)
        z = torch.randn(1, G_opt.z_dim).to(device)
        image = G({"z": z, "label": label}, mode="inference")
        image = torch2image(image).astype("uint8")[0]
        label = torch2numpy(label)
        label_viz = segviz_numpy(label)
        z = z.detach().cpu().view(-1).numpy().tolist()

        # if model_name == "Car":
        #    alpha = np.ones((image.shape[0], image.shape[1], 1), dtype="uint8") * 255
        #    alpha[:64] = alpha[-64:] = 0
        #    image = np.concatenate([image, alpha], 2)
        #    label_viz = np.concatenate([label_viz, alpha], 2)

        return image, label_viz, label.reshape(-1).tolist(), z
