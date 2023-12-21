# python 3.7
# pylint: disable=too-many-arguments,invalid-name,too-many-locals
"""Helper functions."""

import torch

from lib.op import bu, multigpu_map
from lib.misc import GeneralThread
from .semantic_extractor import EXTRACTOR_POOL
from .model_settings import MODEL_POOL
from .pggan_generator import PGGANGenerator
from .pggan_discriminator import PGGANDiscriminator
from .stylegan_generator import StyleGANGenerator
from .stylegan_discriminator import StyleGANDiscriminator
from .stylegan_encoder import StyleGANEncoder
from .stylegan2_generator import StyleGAN2Generator
from .stylegan2_discriminator import StyleGAN2Discriminator
from .perceptual_model import PerceptualModel
from .biggan_generator import BigGANWrapper

__all__ = [
    "G_from_name",
    "build_generator",
    "build_discriminator",
    "build_encoder",
    "build_perceptual",
    "build_semantic_extractor",
    "load_semantic_extractor",
    "save_semantic_extractor",
]


def sample_condition_vector(n_samples, n_class):
    """Sample condition vector."""
    c = torch.zeros(n_samples, n_class)
    indice = torch.randint(0, n_class, size=(n_samples, 1))
    return c.scatter_(1, indice, 1)


def sample_generator_feature(
    g_name,
    latent_type="trunc-wp",
    truncation=0.5,
    layer_idx="auto",
    randomize_noise=True,
    n_samples=256,
    cs=None,
    size=256,
    device_ids=None,
    cpu=False,
    seed=None,
):
    """Sample the feature block for clustering.
    Args:
    Returns:
        mimage: the images, a list of CPU Tensors of (B, C, H, W);
        mfeat: the features, a list of GPU Tensors (in different devices) of (B, C, H, W). If the device_ids is None, then it will return a CPU Tensor.
    """
    devices = [f"cuda:{d}" for d in device_ids] if device_ids else ["cpu"]
    n_gpu = len(devices)
    d_size = n_samples // n_gpu
    generators = [
        build_generator(
            g_name, randomize_noise=randomize_noise, truncation_psi=truncation
        ).net.to(d)
        for d in devices
    ]
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    with torch.no_grad():
        wps = sample_latent(generators[0], n_samples, latent_type)
        wps = [wps[d_size * i : d_size * (i + 1)].to(d) for i, d in enumerate(devices)]
        feats = generate_image(generators[0], wps[0][:1], generate_feature=True)[1]
    if layer_idx == "auto":
        layers = auto_layer_selection([f.shape for f in feats])
    else:
        layers = [int(i) for i in layer_idx.split(",")]
    print(f"=> Selected layers: (out of {len(feats)} layers)")
    for idx in layers:
        print(f"=> {idx:02d} {feats[idx].shape}")
    del feats
    args = [generators, None, size, wps, cs, layer_idx, latent_type, "NHWC", cpu]
    mimage, mfeat = multigpu_map(sample_layer_feature, args, "append")
    del generators
    torch.cuda.empty_cache()
    return mimage, mfeat


def get_mixwp(G, N=1):
    """Sample a mixed W+ latent vector. Available for StyleGANs."""
    L = G.num_layers
    zs = torch.randn(N * L, 512).to(next(G.parameters()))
    return G.mapping(zs).view(N, L, -1)


def get_wp(G, N=1):
    """Sample a W+ latent vector. Available for StyleGANs."""
    z = torch.randn(N, 512).to(next(G.parameters()))
    L = G.num_layers
    return G.mapping(z).unsqueeze(1).repeat(1, L, 1)


def get_trunc_wp(G, N=1):
    """Sample a W+ latent vector. Available for StyleGANs."""
    z = torch.randn(N, 512).to(next(G.parameters()))
    return G.truncation(G.mapping(z))


def sample_latent(G, N, latent_type="trunc-wp"):
    """If G is a StyleGAN, return a mixed wp.
    If G is PGGAN, return a normal latent vector.

    Args:
      G : The generator.
      N : The number of samples.
    Returns:
      N sampled mixed wp or normal latent code.
    """
    if hasattr(G, "mapping"):
        if latent_type == "mixwp":
            return get_mixwp(G, N)
        elif latent_type == "wp":
            return get_wp(G, N)
        elif latent_type == "trunc-wp":
            z = get_trunc_wp(G, N)
            return z
    if hasattr(G, "sample_latent"):
        return G.sample_latent(N)
    device = next(G.parameters()).device
    return torch.randn(N, 512).to(device)  # hardcode


def generate_image(G, wp, c=None, generate_feature=False):
    """Handling forward for StyleGAN and non-StyleGAN"""
    if hasattr(G, "is_ada"):
        # note that this wp should be z
        return G(wp, c, generate_feature=generate_feature)
    if hasattr(G, "synthesis"):
        return G.synthesis(wp, generate_feature=generate_feature)
    return G(wp, generate_feature=generate_feature)


def auto_layer_selection(shapes):
    """Automatically select the features used in clustering."""
    layers = []
    for i in range(len(shapes) - 1):
        if shapes[i][2] < shapes[i + 1][2]:
            layers.append(i)
    if shapes[-1][2] > shapes[layers[-1]][2]:
        layers.append(len(shapes) - 1)
    while sum([shapes[l][1] for l in layers]) > 1000:
        del layers[0]
    return layers


def sample_layer_feature(
    G,
    N=None,
    S=256,
    wps=None,
    cs=None,
    layer_idx="auto",
    latent_type="trunc-wp",
    order="NHWC",
    cpu=False,
):
    """Sample feature maps of generator.
    Args:
        G: The generator.
        N: The total sample number.
        n_batch: The size of minibatch during each forward. Default is 1,
                choose higher values for higher efficiency.
        layer_idx: The indices of target feature maps. str type. Can be a
                    single number, multiple numbers separated by colons,
                    and 'auto'.
        order: The channel order. Can be 'NHWC' or 'NCHW'.
        latent_type: Can be 'mixwp' or 'wp'. The wp is used in original StyleGAN.
    Returns:
        image (in [0, 1]), feature block
    """
    image, feat = [], []
    device = next(G.parameters()).device
    N = N if N is not None else len(wps)
    with torch.no_grad():
        for i in range(N):
            wp = (
                sample_latent(G, 1, latent_type)
                if wps is None
                else wps[i : i + 1].to(device)
            )
            c = None if cs is None else cs[i : i + 1].to(device)
            image_, feature_ = generate_image(G, wp, c, generate_feature=True)
            if len(feat) == 0:  # set up feature storation
                if layer_idx != "auto":
                    indice = [int(l) for l in layer_idx.split(",")]
                else:
                    indice = auto_layer_selection([f.shape for f in feature_])
                C = sum([feature_[l].shape[1] for l in indice])
                shape = (N, S, S, C) if order == "NHWC" else (N, C, S, S)
                if not cpu:
                    feat = torch.cuda.FloatTensor(*shape, device=device)
                else:
                    feat = torch.Tensor(*shape)
            if len(image) == 0:  # set up image storation
                image = torch.zeros(N, 3, *image_.shape[2:])
            c = 0
            for l in indice:
                d = feature_[l].shape[1]
                if order == "NHWC":
                    feat[i : i + 1, ..., c : c + d].copy_(
                        bu(feature_[l], S).permute(0, 2, 3, 1), non_blocking=True
                    )
                else:
                    feat[i : i + 1, c : c + d].copy_(
                        bu(feature_[l], S), non_blocking=True
                    )
                c += d
            image[i : i + 1].copy_(image_)
        image = (1 + image.clamp(-1, 1)) / 2
    return image, feat


def G_from_name(fpath):
    """Build generator from name."""
    name = fpath.split("/")[-2]
    model_name = "_".join(name.split("_")[:2])
    return model_name, build_generator(model_name).net


def load_semantic_extractor(fpath):
    """Load semantic extractor from a pth path."""
    data = torch.load(fpath)
    try:
        SE_type = data["arch"]["type"]
    except KeyError:  # a bug in code
        print(f"!> {fpath} data incomplete, use LSE data.")
        lse_data = torch.load(fpath.replace("NSE-2", "LSE"))
        data["arch"] = {
            "ksize": 3,
            "type": "NSE-2",
            "n_class": lse_data["arch"]["n_class"],
            "dims": lse_data["arch"]["dims"],
            "layers": lse_data["arch"]["layers"],
        }
        torch.save(data, fpath)
        SE_type = data["arch"]["type"]
    SE = EXTRACTOR_POOL[SE_type](**data["arch"])
    SE.load_state_dict(data["param"])
    return SE


def save_semantic_extractor(SE, fpath):
    """Save semantic extractor and architecture information."""
    data = {"arch": SE.arch_info(), "param": SE.state_dict()}
    torch.save(data, fpath)


def build_semantic_extractor(model_name, n_class, dims, layers, **kwargs):
    """Builds semantic extractor by model name."""
    if model_name not in EXTRACTOR_POOL:
        raise ValueError(
            f"Model `{model_name}` is not registered in "
            f"`EXTRACTOR_POOL` in `semantic_extractor.py`!"
        )
    return EXTRACTOR_POOL[model_name](
        n_class=n_class, dims=dims, layers=layers, **kwargs
    )


def build_generator(model_name, logger=None, **kwargs):
    """Builds generator module by model name."""
    gan_type, _ = model_name.split("_")
    if gan_type in ["stylegan3"]:
        from .stylegan3 import StyleGAN3Wrapper

        return StyleGAN3Wrapper(model_name)
    if gan_type in ["pggan", "pgganinv"]:
        return PGGANGenerator(model_name, logger=logger, **kwargs)
    if gan_type in ["stylegan", "styleganinv"]:
        return StyleGANGenerator(model_name, logger=logger, **kwargs)
    if gan_type in ["stylegan2", "stylegan2inv"]:
        return StyleGAN2Generator(model_name, logger=logger, **kwargs)
    if gan_type in ["biggan"]:
        return BigGANWrapper(256)
    if gan_type in ["ada"]:
        from .stylegan2_ada import StyleGAN2ADAWrapper

        return StyleGAN2ADAWrapper(model_name)
    raise NotImplementedError(f"Unsupported GAN type `{gan_type}`!")


def build_discriminator(model_name, logger=None):
    """Builds discriminator module by model name."""
    if model_name not in MODEL_POOL:
        raise ValueError(
            f"Model `{model_name}` is not registered in "
            f"`MODEL_POOL` in `model_settings.py`!"
        )

    gan_type = model_name.split("_")[0]
    if gan_type in ["pggan", "pgganinv"]:
        return PGGANDiscriminator(model_name, logger=logger)
    if gan_type in ["stylegan", "styleganinv"]:
        return StyleGANDiscriminator(model_name, logger=logger)
    if gan_type in ["stylegan2", "stylegan2inv"]:
        return StyleGAN2Discriminator(model_name, logger=logger)
    raise NotImplementedError(f"Unsupported GAN type `{gan_type}`!")


def build_encoder(model_name, logger=None):
    """Builds encoder module by model name."""
    if model_name not in MODEL_POOL:
        raise ValueError(
            f"Model `{model_name}` is not registered in "
            f"`MODEL_POOL` in `model_settings.py`!"
        )

    gan_type = model_name.split("_")[0]
    if gan_type == "styleganinv":
        return StyleGANEncoder(model_name, logger=logger)
    raise NotImplementedError(f"Unsupported GAN type `{gan_type}`!")


build_perceptual = PerceptualModel
