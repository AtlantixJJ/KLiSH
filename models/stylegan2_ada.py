"""StyleGAN2-ADA"""
import sys, torch, os

# for psp use cases
sys.path.insert(0, "../../thirdparty/stylegan2_ada")
sys.path.insert(0, "thirdparty/stylegan2_ada")
import legacy, dnnlib

BASE_DIR = os.path.dirname(os.path.relpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "thirdparty", "stylegan2_ada", "pretrained")


class StyleGAN2ADAWrapper(object):
    """The wrapper for StyleGAN2 ADA model."""

    def __init__(self, name):
        self.name = name
        self.net = StyleGAN2ADA(name)


class StyleGAN2ADA(torch.nn.Module):
    """StyleGAN2 ADA model"""

    def __init__(self, name, truncation_psi=0.7):
        super().__init__()
        gan_type, model_type = name.split("_")
        self.is_ada = True
        self.truncation_psi = truncation_psi
        network_pkl = {
            "cat": f"{MODEL_DIR}/afhqcat.pkl",
            "dog": f"{MODEL_DIR}/afhqdog.pkl",
            "wild": f"{MODEL_DIR}/afhqwild.pkl",
            "brecahad": f"{MODEL_DIR}/brecahad.pkl",
            "metface": f"{MODEL_DIR}/metfaces.pkl",
            "cifar10": f"{MODEL_DIR}/cifar10.pkl",
        }[model_type]
        with dnnlib.util.open_url(network_pkl) as f:
            self.net = legacy.load_network_pkl(f)["G_ema"]
        self.w_avg = self.net.mapping.w_avg
        self.num_layers = len(self.net.synthesis.block_resolutions)

    def set_device(self, device):
        """Set the device."""
        self.net = self.net.to(device)
        return self

    def forward(
        self,
        z,
        c=None,
        size=None,
        input_is_latent=False,
        generate_feature=False,
        generate_layer=False,
    ):
        """Get the image and features from the generator."""
        if input_is_latent:
            ws = z
        else:
            if self.net.c_dim > 0 and c is None:
                n_class = self.net.c_dim
                c = torch.zeros(z.shape[0], n_class)
                indice = torch.randint(0, n_class, size=(z.shape[0], 1))
                c = c.scatter_(1, indice, 1).to(z)
            ws = self.net.mapping(z, c, truncation_psi=self.truncation_psi)
        # print(self.net.mapping(torch.randn(1, 512).cuda(), None).shape)
        block_ws = []
        syn = self.net.synthesis
        with torch.autograd.profiler.record_function("split_ws"):
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in syn.block_resolutions:
                block = getattr(syn, f"b{res}")
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv
        x = img = None
        feats, layer_indice = [], []
        for idx, (res, cur_ws) in enumerate(zip(syn.block_resolutions, block_ws)):
            block = getattr(syn, f"b{res}")
            x, img = block(x, img, cur_ws)
            if generate_feature:
                feats.append(x)
                layer_indice.append(2 * idx + 1)
        if generate_layer:
            return img, feats, layer_indice
        if generate_feature:
            return img, feats
        return img
