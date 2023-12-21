import sys, torch, argparse, pickle, glob
sys.path.insert(0, "thirdparty/spade")
sys.path.insert(0, ".")
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import data
from pixmodels.pix2pix_model import Pix2PixModel
from lib.op import bu
from lib.misc import set_cuda_devices
from lib.dataset import SimpleDataset
from lib.visualizer import segviz_torch
from predictors.helper import P_from_name


def load_opt(args):
  CKPT_DIR = f"thirdparty/spade/checkpoints/"
  Gn1, Gn2, serial, _ = args.name.split("_")
  G_name = f"{Gn1}_{Gn2}"
  label_name = glob.glob(f"expr/data/{G_name}_s1113/label_{serial}_c*")[0]
  n_class = int(label_name[label_name.rfind("_") + 2:])

  opt = pickle.load(open(
    f"{CKPT_DIR}/{args.name}/opt.pkl", "rb"))
  opt.checkpoints_dir = CKPT_DIR
  opt.name = args.name
  opt.dataset_mode = "custom"
  opt.label_nc = n_class
  opt.semantic_nc = n_class
  opt.gpu_ids = [0] if args.gpu_id != "-1" else []
  opt.batchSize = 1
  opt.isTrain = False
  opt.serial_batches = True
  
  opt.label_dir = label_name
  opt.image_dir = f"expr/data/{G_name}_s1113/image"
  return opt


def main(args):
  opt = load_opt(args)
  dataloader = data.create_dataloader(opt)
  model = Pix2PixModel(opt)
  model.eval()

  if "stylegan2_ffhq" in args.name:
    #indice = [1] # used for teaser
    #torch.manual_seed(1)
    indice = [4, 5, 9, 10] #[1, 5]
    torch.manual_seed(2)
  elif "stylegan2_car" in args.name:
    #indice = [1, 4, 10] # used for teaser
    #torch.manual_seed(1)
    indice = [12, 15, 19, 20] # used in paper
    torch.manual_seed(3)
  elif "ada_cat" in args.name:
    indice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    torch.manual_seed(3)
  elif "ada_dog" in args.name:
    indice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    torch.manual_seed(3)
  elif "ada_wild" in args.name:
    indice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    torch.manual_seed(3)

  zs = torch.randn(args.R, opt.z_dim).to(device)
  N = args.R + 1
  label_vizs, images = [], []
  with torch.no_grad():
    for i, data_i in enumerate(dataloader):
      if i > max(indice):
        break
      if i not in indice:
        continue
      label_viz = segviz_torch(data_i['label'].squeeze(1).long())
      label_vizs.append(label_viz.cpu())
      images.append(label_viz.cpu())
      for j in range(zs.shape[0]):
        data_i['z'] = zs[j:j+1]
        image = model(data_i, mode='inference')
        image = (image.clamp(-1, 1) + 1) / 2
        images.append(image.cpu())

  images = torch.cat(images)
  D = int(10 * (images.shape[2] / 512))
  if "car" in args.name:
    L = images.shape[2] // 8
    images = images[:, :, L:-L]
  disp = vutils.make_grid(images,
    nrow=N, padding=D, pad_value=1)[:, D:-D, D:-D]
  vutils.save_image(disp,
    f"results/plot/{args.name}_pix2pix_parallel_{len(indice)}.png")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", default="all")
  parser.add_argument("--size", type=int, default=512)
  parser.add_argument("--gpu-id", type=str, default="0")
  parser.add_argument('--R', type=int, default=3,
    help='The number of samples.')
  args = parser.parse_args()

  set_cuda_devices(args.gpu_id)
  device = "cuda" if args.gpu_id != "-1" else "cpu"

  if args.name == "all":
    #for name in ["stylegan2_ffhq_us0_512", "stylegan2_car_us1_512", "ada_wild_us0_512"]:
    for name in ["ada_cat_us0_512", "ada_dog_us0_512"]:
      args.name = name
      main(args)
  else:
    main(args)
