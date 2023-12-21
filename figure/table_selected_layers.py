"""Print the selected layers to a table."""
import torch, sys

sys.path.insert(0, ".")
from models.helper import (
    build_generator,
    sample_latent,
    auto_layer_selection,
)


def worker(G_name):
    """Worker."""
    G = build_generator(G_name).net.cuda()
    with torch.no_grad():
        wps = sample_latent(G, 1).cuda()
        if hasattr(G, "synthesis"):
            _, features, layer_indice = G.synthesis(
                wps, generate_feature=True, generate_layer=True
            )
        else:
            _, features, layer_indice = G(
                wps, generate_feature=True, generate_layer=True
            )
        shapes = [f.shape for f in features]
        layer_index = auto_layer_selection(shapes)
        global_indice = [layer_indice[i] for i in layer_index]
        dimension = sum([shapes[i][1] for i in layer_index])
    return global_indice, dimension


if __name__ == "__main__":
    G_names = [
        "stylegan2_bedroom",
        "stylegan2_church",
        "stylegan2_ffhq",
        "stylegan2_car",
        "stylegan_bedroom",
        "stylegan_church",
        "stylegan_celebahq",
        "ada_metface",
        "ada_cat",
        "ada_dog",
        "ada_wild",
        "pggan_celebahq",
        "pggan_bedroom",
        "pggan_church",
    ]
    with open("figure/selected_layers.csv", "w", encoding="ascii") as f:
        for G_name in G_names:
            layer_index, dimension = worker(G_name)
            f.write(f"{G_name},{layer_index},{dimension}\n")
            torch.cuda.empty_cache()
