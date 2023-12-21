"""Easy execution script.
Running scripts at different gpu slots:
python submit.py --gpu 0/1/2/3/...

Running scripts at different multi-gpu slots:
python submit.py --gpu 0,1/2,3/4,5/...

Running scripts with default gpu specified in the script:
python submit.py --gpu -1
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--func", default="test", type=str)
parser.add_argument("--gpu", default="x")
args = parser.parse_args()


def calc_class_centroids():
    cmds = []
    cmd = "python -m script.calc_class_centroid --G-name {G_name}"
    for G_name in [
        "stylegan2_ffhq",
        "stylegan2_bedroom",
        "stylegan2_church",
        "stylegan_celebahq",
        "stylegan_bedroom",
        "stylegan_church",
        "pggan_celebahq",
        "pggan_bedroom",
        "pggan_church",
    ]:
        cmds.append(cmd.format(G_name=G_name))
    return cmds


def train_se_fewshot():
    """For results in Table.3."""
    cmds = []
    evalcmd = "python3 train_fewshot.py --G {G} --num-sample {num_sample} --repeat-ind {repeat_ind} --label-set deeplabv3"
    for repeat_ind in range(5):
        for G in ["stylegan2_ffhq"]:
            for num_sample in [1, 10]:
                cmds.append(
                    evalcmd.format(G=G, num_sample=num_sample, repeat_ind=repeat_ind)
                )
    return cmds


def train_se_full():
    """For results in Table.1. Compare LSE and SVM loss."""

    cmds = []
    evalcmd = "python train_full.py --G {G} --loss-type {loss} --SE {model}"
    G_names = [
        "stylegan2_ffhq",
        "stylegan2_bedroom",
        "stylegan2_church",
        "stylegan_celebahq",
        "stylegan_bedroom",
        "stylegan_church",
        "pggan_celebahq",
        "pggan_bedroom",
        "pggan_church",
    ]
    # for loss in ["normal"]:
    #    for model in ["LSE", "NSE-1", "NSE-2"]:
    #        for G in G_names:
    #            cmds.append(evalcmd.format(G=G, loss=loss, model=model))
    for loss in ["svm-1", "normal"]:
        for model in ["LSE"]:
            for G in G_names:
                cmds.append(evalcmd.format(G=G, loss=loss, model=model))
    return cmds


def train_image_seg():
    """Train image segmentation network."""
    cmds = []
    cmd = "python train_image_seg.py --gpu-id 0,1,2,3,4,5,6,7 --batch-size 4 --data-dir data/generated/{G_name}_s1113 --label-set {label_set} --n-epoch 10"

    # for repeat_ind in range(5):
    #    for num_sample in [1, 10]:
    #        label_set = f"deeplabv3_LSE_r{repeat_ind}_n{num_sample}_c15"
    #        cmds.append(cmd.format(G_name="stylegan2_ffhq", label_set=label_set))

    # cmds.append(cmd.format(G_name="stylegan2_ffhq", label_set="klish_c26"))
    # cmds.append(cmd.format(G_name="stylegan2_car", label_set="klish_c12"))
    cmds.append(cmd.format(G_name="stylegan2_ffhq", label_set="ahc_c26"))

    return cmds


def test_image_seg():
    """Test image segmentation networks."""
    cmds = []
    cmd = "python test.py --calc 1 --model expr/image_seg/{G_name}_s1113/{type}_LSE_r{repeat_ind}_n{num_sample}_c{n_class}/deeplabv3+_c{n_class}.pth --out-dir results/predictions_face/{type}"

    G_name = "stylegan2_ffhq"
    n_class = {"stylegan2_ffhq": 15}[G_name]

    type = "deeplabv3"
    for num_sample in [1, 10]:
        for repeat_ind in range(5):
            cmds.append(
                cmd.format(
                    G_name=G_name,
                    repeat_ind=repeat_ind,
                    type=type,
                    n_class=n_class,
                    num_sample=num_sample,
                )
            )

    # unsupervised
    cmd = "python3 test.py --calc 1 --model expr/image_seg/{G_name}_s1113/us{ind}_c{n_class}/deeplabv3+_c{n_class}.pth --out-dir results/predictions_face/unsupervised"
    for ind, n_class in [[0, 28]]:
        cmds.append(cmd.format(G_name=G_name, ind=ind, n_class=n_class))

    return cmds


def cluster_klish():
    """Run KLiSH clustering."""
    cmds = []
    cmd = "python mlsc.py --G-name {G_name} --seed {seed} --gpu-id {gpu} --skip-existing 1 --k-init {K}"
    count = 0

    for G_name in [
        # "stylegan2_car",
        # "stylegan2_ffhq",
        # "pggan_celebahq",
        # "stylegan_celebahq",
        "ada_cat",
        "ada_dog",
        "ada_wild",
        "ada_metface",
        # "stylegan2_bedroom",
        # "stylegan2_church",
        # "pggan_bedroom",
        # "pggan_church",
        # "stylegan_bedroom",
        # "stylegan_church",
    ]:
        for seed in range(1990, 1995):
            if "ada" in G_name:
                K = 40
            else:
                K = 100
            gpu = "0,1,2,3,4,5,6,7"
            cmds.append(cmd.format(G_name=G_name, seed=seed, gpu=gpu, K=K))
            count += 1

    return cmds


def cluster_mld():
    """Run KLiSH clustering."""
    cmds = []
    cmd = "python mlsc.py --G-name {G_name} --seed {seed} --metric {metric} --objective {objective}-l1 --l2-coef 1 --l1-coef 0 --k-init {K} --svm-coef {svm_coef} --n-samples {n_samples} --name {objective}-{svm_coef} --skip-existing 0"
    count = 0

    for objective in ["newcbmcsvc", "mcmld"]:  # ["cbmcsvc", "mcsvc"]:
        metric = objective
        for seed in range(1990, 1992):
            for svm_coef in [1, 10]:
                for G_name in [
                    "stylegan2_ffhq",
                    "stylegan2_car",
                ]:
                    n_samples = 256
                    cmds.append(
                        cmd.format(
                            G_name=G_name,
                            seed=seed,
                            K=100,
                            objective=objective,
                            metric=metric,
                            svm_coef=svm_coef,
                            n_samples=n_samples,
                        )
                    )
                    count += 1

    return cmds


def cluster_smovr():
    """Run KLiSH clustering."""
    cmds = []
    cmd = "python mlsc.py --G-name {G_name} --seed {seed} --metric sm-mcmld --objective ovrsvc-{loss_type} --l2-coef 1 --l1-coef 0 --k-init {K} --svm-coef {svm_coef} --n-samples {n_samples} --name smovrsvc-{svm_coef} --skip-existing 0"
    count = 0

    for loss_type in ["l2", "l1"]:
        for seed in range(1990, 1991):
            for svm_coef in [1, 10, 100, 1000]:
                for G_name in [
                    "stylegan2_ffhq",
                    # "stylegan2_car",
                ]:
                    n_samples = 128
                    cmds.append(
                        cmd.format(
                            G_name=G_name,
                            seed=seed,
                            K=100,
                            loss_type=loss_type,
                            svm_coef=svm_coef,
                            n_samples=n_samples,
                        )
                    )
                    count += 1

    return cmds


def cluster_smovrl1l2reg():
    """Run KLiSH clustering."""
    cmds = []
    cmd = "python mlsc.py --G-name {G_name} --seed {seed} --metric sm-mcmld --objective ovrsvc-{loss_type} --l2-coef 1 --l1-coef 1 --k-init {K} --svm-coef {svm_coef} --n-samples {n_samples} --name smovrsvc-{svm_coef}-l1l2reg --skip-existing 0"
    count = 0

    for loss_type in ["l1"]:
        for seed in range(1990, 1992):
            for svm_coef in [1, 10, 100, 1000]:
                for G_name in [
                    "stylegan2_ffhq",
                    "stylegan2_car",
                ]:
                    n_samples = 256
                    cmds.append(
                        cmd.format(
                            G_name=G_name,
                            seed=seed,
                            K=100,
                            loss_type=loss_type,
                            svm_coef=svm_coef,
                            n_samples=n_samples,
                        )
                    )
                    count += 1

    return cmds


def cluster_kmeans():
    """Run K-means++ clustering."""
    cmds = []
    cmd = (
        "python -m script.cluster_kmeans --seed {seed} --G-name {G_name} --gpu-id {gpu}"
    )
    count = 0
    for seed in range(1990, 1995):
        for G_name in [
            "stylegan2_car",
            # "stylegan2_ffhq",
            # "pggan_celebahq",
            # "stylegan_celebahq",
            # "ada_cat",
            # "ada_dog",
            # "ada_wild",
            # "ada_metface",
            # "stylegan2_bedroom",
            # "stylegan2_church",
            # "pggan_bedroom",
            # "pggan_church",
            # "stylegan_bedroom",
            # "stylegan_church",
        ]:
            gpu = "0,1,2,3,4,5,6,7"
            cmds.append(cmd.format(G_name=G_name, seed=seed, gpu=gpu))
            count += 1
    return cmds


def cluster_kmeans_cifar():
    """Run K-means++ clustering."""
    cmds = []
    cmd = "python -m script.cluster_kmeans --G-name ada_cifar10 --n-samples 256 --resolution 128 --ALL-K 40 --seed {seed} --skip-existing 0 --dist euclidean --class-number 10 --class-idx {class_idx}"
    count = 0
    for seed in range(1990, 1992):
        for class_idx in range(10):
            cmds.append(cmd.format(seed=seed, class_idx=class_idx))
            count += 1
    return cmds


def cluster_klish_cifar():
    """Run KLiSH clustering."""
    cmds = []
    cmd = "python mlsc.py --G-name ada_cifar10 --seed {seed} --k-init {K} --n-samples 256 --resolution 128 --skip-existing 0 --class-number 10 --class-idx {class_idx}"
    count = 0

    for seed in range(1990, 1992):
        for class_idx in range(10):
            K = 40
            cmds.append(cmd.format(seed=seed, K=K, class_idx=class_idx))
            count += 1

    return cmds


def cluster_ahc():
    """Run Spectral Clustering."""
    cmds = []
    cmd = "python -m script.cluster_ahc --seed {seed} --G-name {G_name} --dist {dist}"

    for seed in range(1990, 1995):
        for G_name in [
            "ada_cat",
            "ada_dog",
            "ada_wild",
            "ada_metface",
            "stylegan2_ffhq",
            "stylegan2_car",
            "stylegan2_bedroom",
            "stylegan_bedroom",
            "pggan_bedroom",
            "stylegan2_church",
            "pggan_celebahq",
            "pggan_church",
            "stylegan_celebahq",
            "stylegan_church",
        ]:
            for dist in ["euclidean", "arccos"]:
                cmds.append(cmd.format(G_name=G_name, seed=seed, dist=dist))
    return cmds


def eval_cluster():
    """Evaluate clustering results."""
    cmds = []
    cmd = "python -m script.eval_cluster --G-name {G_name} --train-seed {seed} --eval-name {eval_name}"

    for g_name in ["stylegan2_ffhq", "stylegan_celebahq", "pggan_celebahq"]:
        for eval_name in ["klish", "kmeans", "ahc", "kasp"]:
            for seed in range(1990, 1995):
                cmds.append(cmd.format(seed=seed, G_name=g_name, eval_name=eval_name))
    return cmds


def eval_klish_variant():
    """Run ARI evaluation."""
    cmds = []
    cmd = "python -m script.eval_cluster --G-name {G_name} --train-seed {seed} --eval-name {eval_name} --out-dir expr/eval_clustering_iter5_{svm_coef}svm --in-dir expr/cluster/iter5_{svm_coef}svm"

    for svm_coef in [2000, 3000, 4000, 5000]:
        for seed in range(1990, 1994):
            cmds.append(
                cmd.format(
                    seed=seed,
                    svm_coef=svm_coef,
                    G_name="stylegan2_ffhq",
                    eval_name="klish",
                )
            )
    return cmds


def fig_class():
    """Draw figure."""
    cmds = []
    cmd = "python -m figure.fig_class --G-name {G_name}"
    for G_name in [
        "pggan_church",
        "pggan_bedroom",
        "stylegan2_bedroom",
        "stylegan2_church",
        "stylegan_bedroom",
        "stylegan_church",
        "ada_metface",
        "ada_cat",
        "ada_dog",
        "ada_wild",
        "stylegan2_ffhq",
        "stylegan2_car",
        "stylegan_celebahq",
        "pggan_celebahq",
    ]:
        cmds.append(cmd.format(G_name=G_name))
    return cmds


def one_stage():
    return cluster_smovrl1l2reg()
    # return cluster_kmeans() + cluster_klish()


def klish_layers_ablation():
    """Run KLiSH clustering."""
    cmds = []
    cmd = "python mlsc.py --name klish_layers_ablation --n-samples {n_sample} --G-name {G_name} --seed {seed} --gpu-id {gpu} --skip-existing 1 --k-init 100 --layer-idx {layers}"
    count = 0

    gpu = "0,1,2,3,4,5,6,7"
    orig_layers = [9, 11, 13, 15, 17]
    layer_pat_dict = {
        "all": [str(i) for i in range(8, 18)],
        "m1": [str(i - 1) for i in orig_layers],
        # "m2": [str(i - 2) for i in orig_layers],
    }
    for G_name in [
        "stylegan2_ffhq",
        "stylegan_celebahq",
        "pggan_celebahq",
    ]:
        for layer_pat, layers in layer_pat_dict.items():
            layers = ",".join(layers)
            n_sample = 128 if layer_pat == "all" else 256
            for seed in range(1990, 1995):
                cmds.append(
                    cmd.format(
                        G_name=G_name,
                        n_sample=n_sample,
                        seed=seed,
                        gpu=gpu,
                        layers=layers,
                    )
                )
                count += 1

    return cmds


def kmeans_layers_ablation():
    """Run K-means++ clustering for layer ablation experiment."""
    cmds = []
    cmd = "python -m script.cluster_kmeans --n-sample {n_sample} --layer-idx {layers} --name kmeans_layers_ablation --seed {seed} --G-name {G_name} --gpu-id {gpu}"
    count = 0
    gpu = "1,3,4,5,6,7,8,9"
    orig_layers = [9, 11, 13, 15, 17]
    layer_pat_dict = {
        "all": [str(i) for i in range(8, 18)],
        "m1": [str(i - 1) for i in orig_layers],
        # "m2": [st
        # r(i - 2) for i in orig_layers],
    }

    for G_name in [
        "stylegan2_ffhq",
        "stylegan_celebahq",
        "pggan_celebahq",
    ]:
        for layer_pat, layers in layer_pat_dict.items():
            layers = ",".join(layers)
            n_sample = 128 if layer_pat == "all" else 256
            for seed in range(1990, 1995):
                cmds.append(
                    cmd.format(
                        G_name=G_name,
                        seed=seed,
                        gpu=gpu,
                        layers=layers,
                        n_sample=n_sample,
                    )
                )
                count += 1
    return cmds


def cluster_ahc_ablation():
    """Run Spectral Clustering."""
    cmds = []
    cmd = "python -m script.cluster_ahc --name ahc_layers_ablation --seed {seed} --G-name {G_name} --dist {dist} --layer-idx {layers}"
    orig_layers = [9, 11, 13, 15, 17]
    layer_pat_dict = {
        "all": [str(i) for i in range(8, 18)],
        "m1": [str(i - 1) for i in orig_layers],
    }
    for dist in ["euclidean"]:
        for G_name in [
            "stylegan2_ffhq",
            "pggan_celebahq",
            "stylegan_celebahq",
        ]:
            for layer_pat, layers in layer_pat_dict.items():
                layers = ",".join(layers)
                for seed in range(1990, 1995):
                    cmds.append(
                        cmd.format(G_name=G_name, seed=seed, dist=dist, layers=layers)
                    )
    return cmds


def eval_ablation():
    """Run ARI evaluation."""
    cmds = []
    cmd = "python -m script.eval_cluster --G-name {G_name} --train-seed {seed} --eval-name {eval_name} --out-dir expr/eval_clustering_ablation --in-dir expr/cluster/{eval_name}_layers_ablation --layer-idx {layers}"

    orig_layers = [9, 11, 13, 15, 17]
    layer_pat_dict = {
        "all": [str(i) for i in range(8, 18)],
        "m1": [str(i - 1) for i in orig_layers],
        # "m2": [str(i - 2) for i in orig_layers],
    }

    for eval_name in ["klish", "kmeans", "ahc"]:
        for G_name in [
            "stylegan2_ffhq",
            "stylegan_celebahq",
            "pggan_celebahq",
        ]:
            for layer_pat, layers in layer_pat_dict.items():
                layers = ",".join(layers)
                for seed in range(1990, 1995):
                    cmds.append(
                        cmd.format(
                            seed=seed, layers=layers, G_name=G_name, eval_name=eval_name
                        )
                    )
    return cmds


funcs = {
    # layer choice ablation study
    "cluster_ahc_ablation": cluster_ahc_ablation,
    "klish_layers_ablation": klish_layers_ablation,
    "kmeans_layers_ablation": kmeans_layers_ablation,
    "eval_ablation": eval_ablation,
    "ccc": calc_class_centroids,
    "train_se_full": train_se_full,
    "train_se_fewshot": train_se_fewshot,
    #
    "train_image_seg": train_image_seg,
    "test_image_seg": test_image_seg,
    # clustering
    "one_stage": one_stage,
    "cluster_klish": cluster_klish,
    "cluster_mld": cluster_mld,
    "cluster_smovr": cluster_smovr,
    "cluster_kmeans": cluster_kmeans,
    "cluster_kmeans_cifar": cluster_kmeans_cifar,
    "cluster_klish_cifar": cluster_klish_cifar,
    "cluster_ahc": cluster_ahc,
    # evaluate
    "eval_cluster": eval_cluster,
    "eval_klish_variant": eval_klish_variant,
    # plot
    "fig_class": fig_class,
}

gpus = args.gpu.split("/")
slots = [[] for _ in gpus]
for i, cmd_out in enumerate(funcs[args.func]()):
    gpu = gpus[i % len(gpus)]
    cmd = f"{cmd_out} --gpu-id {gpu}" if gpu != "x" else cmd_out
    slots[i % len(gpus)].append(cmd)
for s in slots:
    cmd_out = " && ".join(s) + " &"
    print(cmd_out)
    os.system(cmd_out)
