"""Mean Shift clustering KMeans weights."""
import sys
sys.path.insert(0, ".")
import glob, torch
import numpy as np

from sklearn.cluster import MeanShift
from models.helper import build_generator
from lib.misc import listkey_convert
from lib.op import sample_layer_feature
from script.weight_analysis import visualize


if __name__ == "__main__":
  expr_dir = "expr/cluster/kmeans"
  kmeans_weights = glob.glob(f"{expr_dir}/*lauto*.pth")
  N_viz = 16

  for w_path in kmeans_weights:
    G_name = listkey_convert(w_path,
      ["bedroom", "church", "ffhq", "car"],
      ["stylegan2_bedroom", "stylegan2_church", "stylegan2_ffhq", "stylegan2_car"])
    G = build_generator(G_name).net
    image, feat = sample_layer_feature(
      G, N_viz, latent_type="trunc-wp")
    w = torch.load(w_path, map_location="cpu").numpy()
    print(w_path)
    
    for bandwidth in [0.1, 0.2, 0.5, 0.8, 1.0]:
      res = MeanShift(bandwidth=bandwidth).fit_predict(w)
      indice = np.unique(res)
      nw = []
      for ind in indice:
        v = w[res == ind].mean(0)
        nw.append(v / np.linalg.norm(v, ord=2))
      nw = torch.from_numpy(np.stack(nw)).float().cuda()
      visualize(image, feat, nw, w_path.replace(
        ".pth", f"ms{bandwidth}"))

        