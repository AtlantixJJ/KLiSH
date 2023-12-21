"""Train all psp models."""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--func", default="test", type=str)
parser.add_argument("--gpu", default="x")
args = parser.parse_args()


def seg2image():
    n_class_dic = {
        "stylegan2_ffhq": 26,
        "stylegan2_car": 12,
        "ada_cat": 7,
        "ada_dog": 9,
        "ada_metface": 18,
        "ada_wild": 22,
    }
    n_styles_dic = {
        "stylegan2_ffhq": 18,
        "stylegan2_car": 16,
        "ada_cat": 16,
        "ada_dog": 16,
        "ada_metface": 16,
        "ada_wild": 16,
    }
    output_size_dic = {
        "stylegan2_ffhq": 1024,
        "stylegan2_car": 512,
        "ada_cat": 512,
        "ada_dog": 512,
        "ada_metface": 1024,
        "ada_wild": 512,
    }
    cmds = []
    for G_name in n_class_dic.keys():
        ds_name = f"{G_name}_klish"
        n_class = n_class_dic[G_name]
        # n_style = n_styles_dic[G_name]
        output_size = output_size_dic[G_name]
        cmd = f"python scripts/train.py --dataset_type={ds_name} --exp_dir=../../expr/psp/{ds_name} --workers=2 --batch_size=2 --test_batch_size=2 --test_workers=2 --val_interval=2500 --image_interval=1000 --checkpoint_path ../../expr/psp/{G_name}_encode/checkpoints/best_model.pt --encoder_type=GradualStyleEncoder --lpips_lambda=1 --l2_lambda=1 --start_from_latent_avg --id_lambda=0 --label_nc={n_class} --input_nc={n_class} --output_size={output_size} --G-name={G_name}"
        cmds.append(cmd)
    return cmds


def image2image():
    output_size_dic = {
        "stylegan2_ffhq": 1024,
        "stylegan2_car": 512,
        "ada_cat": 512,
        "ada_dog": 512,
        "ada_metface": 1024,
        "ada_wild": 512,
    }
    cmds = []
    for G_name in output_size_dic.keys():
        ds_name = f"{G_name}_encode"
        # n_class = n_class_dic[G_name]
        # n_style = n_styles_dic[G_name]
        output_size = output_size_dic[G_name]
        cmd = f"python scripts/train.py --dataset_type={ds_name} --exp_dir=../../expr/psp/{ds_name} --workers=2 --batch_size=2 --test_batch_size=2 --test_workers=2 --val_interval=2500 --image_interval=1000 --encoder_type=GradualStyleEncoder --lpips_lambda=1 --l2_lambda=1 --id_lambda=0 --start_from_latent_avg --output_size={output_size} --G-name={G_name}"
        cmds.append(cmd)
    return cmds


def all():
    return image2image()


funcs = {
    "all": all,
    "seg2image": seg2image,
}

gpus = args.gpu.split("/")
slots = [[] for _ in gpus]
for i, cmd_out in enumerate(funcs[args.func]()):
    gpu = gpus[i % len(gpus)]
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} {cmd_out}" if gpu != "x" else cmd_out
    slots[i % len(gpus)].append(cmd)
for s in slots:
    cmd_out = " && ".join(s) + " &"
    print(cmd_out)
    os.system(cmd_out)
