"""KLiSH algorithm.
Usage: python klish.py --G-name <the model name> --gpu-id <your gpu> --N <the number of images>
"""
# pylint: disable=wrong-import-position,wrong-import-order,multiple-imports,invalid-name
import os, sys, argparse, torch
import numpy as np
from torchvision import utils as vutils
from tqdm import tqdm
sys.path.insert(0, ".")

from lib.op import GeneralThread, bu, torch2numpy, delete_index, multigpu_get_svm_label
from lib.misc import DictRecorder
from lib.misc import set_cuda_devices
from lib.visualizer import segviz_torch, VideoRecorder, plot_dict
from models import helper

EPS = 1e-9


def calc_metric_worker(feat, svm_label, w):
    """The worker for calculating merging metrics in a thread.
    Args:
      feat: (L, C)
      w: (K, C)
    """
    dm, dm_c, mm, mm_c, N = 0, 0, 0, 0, 8
    B = feat.shape[0] // N
    with torch.no_grad():
        for i in range(N):
            st, ed = B * i, B * (i + 1)
            seg = torch.matmul(feat[st:ed], w.permute(1, 0))

            # SVM accuracy (IoU)
            gt = svm_label[st:ed] > 0
            dt = seg > 0
            dm += (dt & gt).sum(0).float()
            dm_c += (dt | gt).sum(0).float()

            # ECoS
            dt = (seg.clamp(-1, 1) + 1) / 2
            mm += torch.matmul(dt.permute(1, 0), dt)
            mm_c += torch.square(dt).sum(0)
    return dm, dm_c, mm, mm_c


def calculate_metric(mfeat, msvm_label, w):
    """Calculate merging metrics with multiple threads.
    """
    dm, dm_c, mm, mm_c, threads = 0, 0, 0, 0, []
    for d_id in range(len(mfeat)):
        threads.append(GeneralThread(calc_metric_worker,
                                     mfeat[d_id], msvm_label[d_id],
                                     w.clone().to(f"cuda:{d_id}")))
        threads[-1].start()
    for d_id in range(len(mfeat)):
        threads[d_id].join()
        dm += threads[d_id].res[0].cuda()
        dm_c += threads[d_id].res[1].cuda()
        mm += threads[d_id].res[2].cuda()
        mm_c += threads[d_id].res[3].cuda()
    dm /= dm_c.clamp(min=EPS)
    norm = torch.sqrt(mm_c)
    mm /= (norm.unsqueeze(1) * norm.unsqueeze(0)).clamp(min=EPS)
    return dm, mm


def loss_func_batched_l2svc(feat, w, svm_label):
    """Calculate the L2 SVC loss with multiple gradient accumulation (to save GPU memory).
    """
    acc_svm_loss, N = 0, 8
    B = feat.shape[0] // N
    for i in range(N):
        st, ed = B * i, B * (i + 1)
        seg = torch.matmul(feat[st:ed], w.permute(1, 0))  # (L, K)
        svm_loss = torch.square(
            (1 - svm_label[st:ed] * seg).clamp(min=0)).mean() / N * 10
        acc_svm_loss += svm_loss.detach()
        svm_loss.backward()  # accumulate grad
    return acc_svm_loss


def train_svm_l2reg_l2svc(mfeat, w, optim, msvm_label, mlabel=None):
    """Train the SVM for a single iteration."""
    optim.zero_grad()
    threads, svm_loss = [], 0
    for d_id in range(len(mfeat)):
        threads.append(GeneralThread(loss_func_batched_l2svc,
                                     mfeat[d_id], w.clone().to(f"cuda:{d_id}"), msvm_label[d_id]))
        threads[-1].start()
    for d_id in range(len(mfeat)):
        threads[d_id].join()
        svm_loss += threads[d_id].res.cuda() / len(mfeat)
    reg_loss = 0.5 * torch.square(w).sum(1).mean()
    reg_loss.backward()
    total_loss = reg_loss + svm_loss
    return total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="expr/cluster",
                        help="The directory of experiments.")
    parser.add_argument("--name", default="klish",
                        help="The name of the experiment.")
    parser.add_argument("--G-name", default="stylegan2_ffhq",
                        help="The name of generator, should be in models/pretrained/pytorch folder.")
    parser.add_argument("--layer-idx", default="auto", type=str,
                        help="The layer indice used for collecting features, use ``auto'' for using the default layer selection process.")
    parser.add_argument("--N", default=256, type=int,
                        help="The number of samples.")
    parser.add_argument("--latent-type", default="trunc-wp", type=str,
                        help="The latent type of StyleGANs, can be either one of [trunc-wp, wp, mix-wp].")
    parser.add_argument("--gpu-id", default="0,1,2,3,4,5,6,7", type=str)
    parser.add_argument("--svm-iter", default=100, type=int,
                        help="The maximum training iteration of SVM.")
    parser.add_argument("--resample-interval", default=200, type=int,
                        help="When to re-sample all the features. Use 200 for no resampling (default).")
    parser.add_argument("--kmeans-seed", default=1991, type=int)
    parser.add_argument("--seed", default=1997, type=int)
    args = parser.parse_args()
    n_gpu = set_cuda_devices(args.gpu_id)
    devices = list(range(n_gpu))

    # decide the initial cluster numbers
    if "ada_cat" in args.name or "ada_dog" in args.name or "ada_wild" in args.name:
        K = 50
    else:
        K = 100
    truncation = 0.5

    B = args.N // n_gpu  # The batch size per GPU
    # resolution, number of images used for visualization
    S, N_viz = 256, min(B, 16)
    name = f"{args.G_name}_l{args.latent_type}_i{args.layer_idx}_{args.N}_{args.seed}"
    kmeans_wpath = f"{args.expr}/kmeans/{args.G_name}_l{args.latent_type}_i{args.layer_idx}_K{K}_N256_S256_{args.kmeans_seed}_arccos.pth"
    prefix = f"{args.expr}/{args.name}/{name}"
    if not os.path.exists(f"{args.expr}/{args.name}"):
        os.makedirs(f"{args.expr}/{args.name}")

    print(f"=> Preparing {args.N} samples in {S} resolution ...")
    Gs = [helper.build_generator(args.G_name, randomize_noise=True,
                          truncation_psi=truncation).net.to(f"cuda:{d}") for d in devices]
    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        rng = np.random.RandomState(args.seed)
    with torch.no_grad():
        wps = helper.sample_latent(Gs[0], args.N, args.latent_type)
        wps = [wps[B*i:B*(i+1)].to(f"cuda:{i}") for i in range(n_gpu)]
    # sample the feature block, by default the shape is (N, H, W, C)
    mimage, mfeat = helper.multigpu_sample_layer_feature(
        Gs=Gs, S=S, N=None, wps=wps,
        layer_idx=args.layer_idx, latent_type=args.latent_type)
    B, H, W, C = mfeat[0].shape
    for i in range(n_gpu):  # flatten features
        mfeat[i] = mfeat[i].view(-1, C)
    print(f"=> GPU Feature: {mfeat[0].shape} ({B}, {H}, {W}, {C})")

    w = torch.load(kmeans_wpath).cuda().requires_grad_(True)
    mlabel, msvm_label = multigpu_get_svm_label(mfeat, w)
    print(f"=> Init from {kmeans_wpath}: {w.shape[0]} classes.")

    # visualize the initial clustering
    viz_feat = mfeat[0].view(B, H, W, C)[:N_viz].detach().view(-1, C)
    mimage = [bu(x[:N_viz], (H, W)) for x in mimage]
    labels = torch.matmul(viz_feat, w.permute(1, 0)).argmax(1)
    pred_viz = segviz_torch(labels.view(N_viz, H, W).cpu())
    disp = []
    for i in range(mimage[0].shape[0]):
        disp.extend([mimage[0][i], pred_viz[i]])
    vutils.save_image(torch.stack(disp), f"{prefix}_kmeans.png",
                      nrow=8, padding=2, pad_value=0)
    del viz_feat, pred_viz, disp

    optim = torch.optim.LBFGS([w])
    record = DictRecorder()
    video_recorder = VideoRecorder(prefix, N_viz)
    data_count, count = args.N, 0
    tree, cur_label = {}, w.shape[0]
    idx2label = [i for i in range(w.shape[0])]
    for _ in tqdm(range(K - 1)):
        # train the SVM
        def closure():
            """A training closure."""
            return train_svm_l2reg_l2svc(mfeat, w, optim, msvm_label, mlabel)
        for j in range(args.svm_iter):
            w_prev = w.clone().detach()
            total_loss = optim.step(closure)
            w_diff = float((w - w_prev).abs().max())
            record.add("Max W L2 Norm", w.norm(p=2, dim=1).max())
            record.add("Total Loss", min(total_loss, 10))
            record.add("K", w.shape[0])
            record.add("Data count", data_count)
            record.add("W/difference", min(w_diff, 0.1))
            if w_diff < 1e-4:
                break

        # calculate deleting and merging metrics
        with torch.no_grad():
            dm, mm = calculate_metric(mfeat, msvm_label, w)
        p = int(dm.argmin())
        mm[p, p] = 0
        q = int(mm[p].argmax())
        record.add("Maximum CIoU", dm[p])
        record.add("Minimum ECoS", mm[p, q])

        # resample every n iterations
        if (i + 1) % args.resample_interval == 0:
            data_count += args.N
            video_recorder.clean_sync()
            del mfeat, mimage, msvm_label, mlabel
            torch.cuda.empty_cache()
            mimage, mfeat = helper.multigpu_sample_layer_feature(
                Gs=Gs, S=S, N=args.N,
                layer_idx=args.layer_idx, latent_type=args.latent_type)
            mimage = [bu(x[:N_viz], (H, W)) for x in mimage]
            for i in range(n_gpu):
                mfeat[i] = mfeat[i].view(-1, C)
            mlabel, msvm_label = multigpu_get_svm_label(mfeat, w)
            continue

        # visualize
        video_recorder(None, w, None, mimage[0], mfeat[0],
                       mlabel[0].view(B, H, W), None, None, None, q, p)

        # save merge tree
        tree[count] = {
            "W": torch2numpy(w),
            "Merging Metric": torch2numpy(mm[p, q]),
            "Deleting Metric": torch2numpy(dm[p])}
        tree[count]["merge"] = [idx2label[q], idx2label[p], cur_label]
        cur_label += 1
        torch.save(tree, f"{prefix}_tree.pth")
        idx2label[q] = count  # p -> q, q -> cur_label
        del idx2label[p]
        count += 1

        # merge labels
        new_w = delete_index(w, p, 0)
        new_w = new_w.clone().detach().requires_grad_(True)
        new_optim = torch.optim.LBFGS([new_w])
        del optim, w
        optim, w = new_optim, new_w
        for j in range(len(mlabel)):  # merge label
            mask = msvm_label[j][:, p] > 0
            msvm_label[j][:, q][mask] += 2
            t = msvm_label[j].cpu().detach()
            t = delete_index(t, p, 1)
            msvm_label[j] = None
            torch.cuda.empty_cache()
            msvm_label[j] = t.to(f"cuda:{j}")
            mlabel[j] = msvm_label[j].argmax(1)
        plot_dict(record, f"{prefix}_record.png")
        if w.shape[0] <= 1:
            break

    torch.save(tree, f"{prefix}_tree.pth")
    plot_dict(record, f"{prefix}_record.png")
    video_recorder.clean_sync()
