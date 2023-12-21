"""StyleGAN3."""
import sys, torch

sys.path.insert(0, "thirdparty/stylegan3")
import legacy, dnnlib


class StyleGAN3Wrapper(object):
    """StyleGAN3Wrapper"""

    def __init__(self, name):
        self.name = name
        self.net = StyleGAN3(name)


class StyleGAN3(torch.nn.Module):
    """StyleGAN3"""

    def __init__(self, name, truncation_psi=0.7):
        super().__init__()
        gan_type, model_type = name.split("_")
        self.is_ada = True
        self.truncation_psi = truncation_psi
        PKL_PATH = "thirdparty/stylegan3/pretrained"
        network_pkl = {
            "afhq": f"{PKL_PATH}/stylegan3-r-afhqv2-512x512.pkl",
            "ffhq": f"{PKL_PATH}/stylegan3-r-ffhq-1024x1024.pkl",
            "metface": f"{PKL_PATH}/stylegan3-r-metfaces-1024x1024.pkl",
        }[model_type]
        with dnnlib.util.open_url(network_pkl) as f:
            self.net = legacy.load_network_pkl(f)["G_ema"]
        self.num_layers = self.net.synthesis.num_layers

    def set_device(self, device):
        """Set device"""
        self.net = self.net.to(device)
        return self

    def forward(self, z, size=None, generate_feature=False):
        """Get image and feature."""
        ws = self.net.mapping(z, None, self.truncation_psi, None, False)
        ws = ws.to(torch.float32).unbind(dim=1)
        syn = self.net.synthesis
        x = syn.input(ws[0])
        feat = []
        for name, w in zip(syn.layer_names, ws[1:]):
            x = getattr(syn, name)(x, w)
            if generate_feature and name != syn.layer_names[-1]:
                if x.shape[3] in [64, 128, 256, 512, 1024]:
                    feat.append(x)
                else:
                    s = syn.margin_size
                    feat.append(x[:, :, s:-s, s:-s])
        if syn.output_scale != 1:
            x = x * syn.output_scale
        x = x.to(torch.float32)
        if generate_feature:
            return x, feat
        return x
