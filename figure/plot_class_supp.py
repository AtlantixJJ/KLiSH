"""Plot supplementary figures of KLiSH results."""
import sys
sys.path.insert(0, ".")
import sys, argparse, torch
sys.path.insert(0, ".")
from torchvision import utils as vutils

from lib.visualizer import segviz_torch
from lib.misc import set_cuda_devices
from lib.op import sample_layer_feature, bu, cat_mut_iou
from models.helper import build_generator


def consistent_label(labels):
  M, N, H, W = labels.shape
  new_labels = labels.clone()
  color_idx = labels[0].max() + 1
  for i in range(1, labels.shape[0]):
    IoU = cat_mut_iou(
      labels[i].view(N, -1),
      labels[i - 1].view(N, -1))[0]
    for j in range(IoU.shape[0]):
      max_ind = IoU[j].argmax()
      mask = labels[i] == j
      if IoU[j, max_ind] > 0.7:
        new_idx = int(new_labels[i - 1][mask].mode().values)
      else:
        new_idx = color_idx
        color_idx += 1
      new_labels[i][mask] = new_idx
  return new_labels

  
def visualize(image, feat, w0, kmeans_ws, klish_ws):
  N, H, W, C = feat.shape
  feat = feat.view(-1, C)
  image = bu(image, (H, W))
  kmeans_labels = []
  klish_labels = []
  with torch.no_grad():
    for w in [w0] + kmeans_ws:
      seg = torch.matmul(feat.view(-1, C), w.permute(1, 0))
      kmeans_labels.append(seg.argmax(1).view(N, H, W).cpu())
    for w in [w0] + klish_ws:
      seg = torch.matmul(feat.view(-1, C), w.permute(1, 0))
      klish_labels.append(seg.argmax(1).view(N, H, W).cpu())
  kmeans_labels = consistent_label(torch.stack(kmeans_labels))
  klish_labels = consistent_label(torch.stack(klish_labels))

  kmeans_label_viz = segviz_torch(
    kmeans_labels.view(-1, H, W)).view(-1, N, 3, H, W)
  klish_labels_viz = segviz_torch(
    klish_labels.view(-1, H, W)).view(-1, N, 3, H, W)[1:]
  cls_img = []
  D = 10
  ones = torch.ones((1, 3, image.shape[2], D))
  for i in range(image.shape[0]):
    disp = [image[i:i+1], ones, ones, kmeans_label_viz[:1, i], ones]
    for j in range(1, kmeans_label_viz.shape[0]):
      disp.extend([ones, kmeans_label_viz[j:j+1, i], ones, klish_labels_viz[j-1:j, i], ones])
    if kmeans_label_viz.shape[0] == klish_labels_viz.shape[0] - 1:
      disp.extend([ones, klish_labels_viz[-1:, i]])
    cls_img.append(torch.cat(disp, 3)[0])
  return cls_img


def main(args, feat, image):  
  klish_wfile = torch.load(f"expr/cluster/klish/{args.G_name}_ltrunc-wp_iauto_256_1997_tree.pth")
  kmeans_w = [torch.load(f"expr/cluster/kmeans/{args.G_name}_lwp_iauto_K{K}_N256_S256_1991_arccos.pth") for K in [100, 50, 30, 20]]
  select_classes = [50, 30, 20]
  klish_ws = {}
  for i in list(klish_wfile.keys()):
    w = klish_wfile[i]["W"]
    if w.shape[0] in select_classes:
      klish_ws[w.shape[0]] = torch.from_numpy(w).float().cuda()
  s = " ".join([str(v.shape[0]) for k, v in klish_ws.items()])
  print(f"=> Processing {G_name}: {s}")
  
  klish_ws = [klish_ws[l] for l in select_classes]
  cls_img = visualize(image, feat,
    kmeans_w[0], kmeans_w[1:], klish_ws)
  for i in range(len(cls_img)):
    if "stylegan2_car" == args.G_name:
      cls_img[i] = cls_img[i][:, 32:-32]
    vutils.save_image(cls_img[i], f"results/plot/{args.G_name}_mergeclass{i}_supp.png")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # experiment name
  parser.add_argument("--expr", default="expr/cluster")
  parser.add_argument("--out-dir", default="results/plot")
  # architecture
  parser.add_argument("--G-name", default="all")
  parser.add_argument("--layer-idx", default="auto", type=str)
  parser.add_argument("--w-path",
    default="expr/cluster/klish", type=str)
  parser.add_argument("--gpu-id", default="0", type=str)
  parser.add_argument("--N", default=4, type=int)
  args = parser.parse_args()
  n_gpu = set_cuda_devices(args.gpu_id)

  if args.G_name == "all":
    #G_names = ["stylegan2_ffhq", "stylegan2_car"]
    G_names = ["ada_cat", "ada_dog", "ada_wild", "ada_metface",
    "pggan_celebahq", "pggan_bedroom", "pggan_church",
    "stylegan_celebahq", "stylegan_bedroom", "stylegan_church", 
    "stylegan2_ffhq", "stylegan2_car", "stylegan2_bedroom", "stylegan2_church"]
  else:
    G_names = [args.G_name]
  
  for G_name in G_names:
    args.G_name = G_name
    G = build_generator(args.G_name).net.cuda()
    if "ffhq" in args.G_name or "car" in args.G_name:
      wps = []
      # do not replicate other figure
      for i in range(25, 25 + args.N):
        wps.append(torch.load(
          f"data/{args.G_name}_fewshot/latent/wp_{i:02d}.npy"))
      wps = torch.cat(wps).cuda()
      image, feat = sample_layer_feature(G, args.N,
        wps=wps, layer_idx=args.layer_idx, latent_type="trunc-wp")
    else:
      image, feat = sample_layer_feature(G, args.N,
        layer_idx=args.layer_idx, latent_type="trunc-wp")
    image = bu(image, feat.shape[2])
    main(args, feat, image)
    del G, feat, image
    torch.cuda.empty_cache()

