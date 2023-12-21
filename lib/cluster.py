"""Clustering algorithms implementation."""
# pylint: disable=invalid-name,no-member,too-many-instance-attributes,consider-using-f-string
import json
import math
import os
import glob
import torch
import torchvision.utils as vutils
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
from lib.misc import GeneralThread, DictRecorder, imwrite
from lib.visualizer import (
    visualize_segmentation,
    SimpleVideoRecorder,
    PlannarVideoRecorder,
    plot_dict,
    segviz_torch,
)
from lib.op import (
    bu,
    pairwise_dist,
    torch2numpy,
    copy_tensor,
    upper_triangle_minmax,
    delete_index,
    delete_index_reorder,
    modify_optim,
    multigpu_map,
    torch2image,
)
from models.semantic_extractor import SimpleLSE


EPS = 1e-9
SL1_BETA = 1e-4


class LinearClassifier:
    """Linear classifier on a (flattened) feature block."""

    def __init__(self, w, b=None):
        self.weight, self.bias = w, b

    def to(self, device):
        """Move parameters to device."""
        self.weight = self.weight.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)
        return self

    def cuda(self):
        """Move parameters to GPU."""
        self.weight = self.weight.cuda()
        if self.bias is not None:
            self.bias = self.bias.cuda()
        return self

    def clone(self):
        """Return a new instance with cloned parameters (gradient shared)."""
        if self.bias is None:
            return LinearClassifier(self.weight.clone())
        return LinearClassifier(self.weight.clone(), self.bias.clone())

    def detach(self):
        """Return a new instance with detached parameters."""
        if self.bias is None:
            return LinearClassifier(self.weight.detach())
        return LinearClassifier(self.weight.detach(), self.bias.detach())

    def __call__(self, x):
        """Linear transformation."""
        seg = torch.matmul(x, self.weight.permute(1, 0))
        if self.bias is not None:
            seg = seg + self.bias.unsqueeze(0)
        return seg

    def parameters(self):
        """Get trainable parameters."""
        return [self.weight] if self.bias is None else [self.weight, self.bias]

    def state_dict(self):
        """Return standard state dictionary."""
        if self.bias is None:
            return {"weight": self.weight}
        return {"weight": self.weight, "bias": self.bias}

    def set_param(self, w, b=None):
        """Set the parameters of the linear classifier."""
        self.weight = copy_tensor(w, True)
        if b is not None:
            self.bias = copy_tensor(b, True)

    @staticmethod
    def load_as_lc(bias_file, nobias_file, n_clusters):
        """Load the saved results to Linear Classifier models."""
        dic = {"bias": {}, "nobias": {}}
        for n_cluster in n_clusters:
            if bias_file is not None:
                w = bias_file[n_cluster]["weight"]
                b = bias_file[n_cluster]["bias"]
                dic["bias"][n_cluster] = LinearClassifier(w, b).cuda()
            if nobias_file is not None:
                w = nobias_file[n_cluster]["weight"]
                dic["nobias"][n_cluster] = LinearClassifier(w).cuda()
        return dic


class MultiGPUKMeansPP:
    """Multi-GPU implementation of KMeans++ using PyTorch.
    Args:
        n_class: The number of clusters;
        dist: The metric of K-means. Can be euclidean and arccos.
        n_repeat: The total repeat number to find a best solution.
        tol: The tolerance measured in max-norm (L-inf).
        max_iter: The default maximum clustering iteration.
    """

    def __init__(
        self,
        n_class=100,
        dist="euclidean",
        n_repeat=1,
        max_iter=20,
        seed=1990,
    ):
        self.n_class = n_class
        self.seed = seed
        self.n_repeat = n_repeat
        self.dist = dist
        self.max_iter = max_iter
        self.fitted = False
        self.verbose = False
        self.w, self.bias = None, None
        self.count, self.unormed_w = None, None
        self.record = None
        self.repeat_record = []

    def log(self, msg):
        """Logging with verbose control."""
        if self.verbose:
            print(msg)

    def fit(self, mfeat, mx2=None, verbose=False):
        """Cluster on mx.
        Args:
            mx: A multi-GPU list of Tensosr in shape (N, C).
            mx2: The squared norm of mX, a list of multi-GPU Tensor of shape (N,).
                Used for accelerating the calculation in euclidean metric.
        """
        self.verbose = verbose
        self.log(f"=> MultiGPU KMeans++ clustering on {self.dist} metric.")
        rng = np.random.RandomState(self.seed)
        for i in range(self.n_repeat):
            self.log(f"=> Repeat {i+1}/{self.n_repeat} K={self.n_class} initializing.")
            w_init = multigpu_kmeanspp_init(mfeat, self.n_class, self.dist, mx2, rng)
            torch.cuda.empty_cache()
            self.log(f"=> Repeat {i+1}/{self.n_repeat} on {self.dist} metric...")
            res = multigpu_kmeans(
                mfeat,
                w_init,
                n_iter=self.max_iter,
                dist=self.dist,
                mx2=mx2,
            )
            self.repeat_record.append(res)
            torch.cuda.empty_cache()
        icdists = torch.Tensor([x[3] for x in self.repeat_record])
        icdists = torch.nan_to_num(icdists, 1e10)
        self.log(
            f"=> In {self.n_repeat} repeats, intra-class distances of {self.dist} range between [{icdists.min():.6f}, {icdists.max():.6f}]."
        )
        best_ind = torch.argmin(icdists)
        w, unormed_w, self.count, _, self.record = self.repeat_record[best_ind]
        if self.dist == "euclidean":
            self.bias = -0.5 * (w * w).sum(1)
            self.w = w
            self.unormed_w = w
        elif self.dist == "arccos":
            self.w = w
            self.unormed_w = unormed_w
            self.bias = torch.zeros(w.shape[0])
        self.fitted = True
        self.log("=> Done.")

    def predict(self, x):
        """Predict the cluster assignments of a Tensor.
        Args:
          x : Tensor of shape (N, C).
        Returns:
          cluster assignments of shape (N,).
        """
        if not self.fitted:
            self.log("!> Not fitted.")
            return
        score = torch.matmul(x, self.w.to(x).permute(1, 0))
        score = score + self.bias.to(x).view(1, -1)
        return score.argmax(1)

    @property
    def param(self):
        """Get the information to be saved."""
        if not self.fitted:
            print("!> Not fitted.")
            return {}
        return {
            "weight": self.w,
            "bias": self.bias,
            "unormed_W": self.unormed_w,
            "count": self.count,
        }


class KASP:
    """Fast Approximate Spectral Clustering."""

    def __init__(self, kmeans_model, min_clusters=2, max_clusters=100):
        self.kmeans_model = kmeans_model
        self.K = self.kmeans_model.weight.shape[0]
        self.n_clusters = list(range(min_clusters, max_clusters))
        self.fitted = False
        self.merge_record = {}
        self.merge_matrix = {}

    def restore(self, merge_record):
        """Set the merge record."""
        self.merge_record = {int(k): v for k, v in merge_record.items()}
        self.calc_permute_matrix()
        self.fitted = True
        return self

    def calc_permute_matrix(self):
        """Calculate the permutation matrix for each step."""
        self.merge_matrix = {
            k: permute_matrix(v, k)[1] for k, v in self.merge_record.items()
        }

    def fit(self):
        """Fit the kmeans model."""
        w_np = self.kmeans_model.weight.cpu().numpy()
        for n_cluster in tqdm(self.n_clusters):
            alg = SpectralClustering(
                n_cluster, affinity="nearest_neighbors", assign_labels="discretize"
            )
            label_perm = alg.fit_predict(w_np)
            self.merge_record[n_cluster] = label_perm.tolist()
        self.fitted = True

    def predict(self, x, n_cluster):
        """Predict cluster assignments."""
        if not self.fitted:
            return 0
        kmeans_label = self.kmeans_model(x).argmax(1)  # (either NCHW or NC)
        label_perm = np.array(self.merge_record[n_cluster]).astype("uint8")
        old2new = [np.where(label_perm == i)[0] for i in range(n_cluster)]
        new_label = torch.zeros_like(kmeans_label)
        for new_y, sub in enumerate(old2new):
            for v in sub:
                new_label[kmeans_label == v] = new_y
        return new_label


class MLSClustering:
    """Maximum Linear Separability Clustering.
    Args:
        w_init: K-means weight.
    """

    def __init__(
        self,
        w_init,
        use_bias=True,
        n_viz=16,
        image_shape=None,
        metric="heuristic",
        objective="ovrsvc-l2",  # ovrsvc, mcsvc, mcmld; l1, l2
        max_iter=20,
        svm_coef=1.0,
        l1_coef=0.0,
        l2_coef=1.0,
        save_prefix="expr/",
    ):
        self.svm_coef = svm_coef
        self.l1_coef, self.l2_coef = l1_coef, l2_coef
        self.w_init, self.use_bias = w_init, use_bias
        self.K_init, self.D = self.w_init.shape
        self.K = self.K_init  # current number of clusters
        self.max_iter, self.n_viz = max_iter, n_viz
        self.image_shape, self.feat_shape = image_shape, None
        self.save_prefix = save_prefix
        self.record, self.video_recorder = DictRecorder(), None
        self.merge_record = {}
        # used for strict minimization mode
        self.merge_table, self.merge_weights, self.merge_bias = None, None, None
        self.cur_label = self.K_init
        self.idx2label = list(range(self.K_init))
        # define the name of data first
        self.mimage, self.mx, self.mx2, self.my, self.mby = [], [], [], [], []
        # training stored variables
        self.svm_loss, self.svm_closs = 0, 0
        self.viz_image, self.viz_feat = None, None
        self.n_gpu, self.coef, self.class_size = -1, 1.0, None
        self.fitted = False

        if use_bias:
            b = -0.5 * (w_init**2).sum(1)
            div = 2 * (-b).max()
            w_init = copy_tensor((w_init / div).cuda(), True)
            b_init = copy_tensor((b / div).cuda(), True)
            self.model = LinearClassifier(w_init, b_init)
        else:
            w_init = copy_tensor(w_init.cuda(), True)
            self.model = LinearClassifier(w_init)
        self.model_reps = []
        self.metric, self.objective = metric, objective

        self.optim = torch.optim.LBFGS(self.model.parameters(), max_iter=5)
        self.has_new_model = False
        self.MEM_UNIT = 200 * 26 * 256 * 256

    def _train_topk_candidate(self, mat):
        """Train model on the top-k candidates of the matrix."""
        sorted_idx = mat.view(-1).argsort(descending=True)
        M, K = self.mby[0].shape[1], self.model.weight.shape[0]
        # the sorting is in ascending order
        sorted_idx = [(int(idx) // K, int(idx) % K) for idx in sorted_idx]
        sorted_idx = [x for x in sorted_idx if x[0] < x[1]]  # remove lower triangle
        indice = [x for x in sorted_idx if self.merge_table[x[0], x[1]] < -0.9]
        # train some weights again if there are spares
        comp_len = M - len(indice)
        if comp_len > 0:
            sorted_idx = self.merge_table.view(-1).argsort(descending=True)
            sorted_idx = [(int(idx) // K, int(idx) % K) for idx in sorted_idx]
            sorted_idx = [x for x in sorted_idx if x[0] < x[1]]
            indice.extend([x for x in sorted_idx if x not in indice])
        M = min(len(indice), M)
        self.qp_indice = indice[:M]
        for d_id, by in enumerate(self.mby):
            cby = by.cpu().clone().detach()
            for i, (q, p) in enumerate(self.qp_indice):
                self.mby[d_id][:, i].copy_(cby[:, q] + cby[:, p])
        # initialize the merged pairs
        old_losses = torch.zeros(M)
        w = copy_tensor(self.model.weight)
        ws = []
        if self.use_bias:
            b = copy_tensor(self.model.bias)
            bs = []
        for i, (q, p) in enumerate(self.qp_indice):
            alpha = self.class_size[p] / self.class_size[[p, q]].sum()
            ws.append(alpha * w[p] + (1 - alpha) * w[q])
            if self.use_bias:
                bs.append(alpha * b[p] + (1 - alpha) * b[q])
            old_losses[i] = self.svm_closs[[p, q]].sum()
        # train the merged clusters
        orig_model = self.model.clone().detach()
        del self.optim, self.model, self.model_reps, w, b
        w = copy_tensor(torch.stack(ws), True)
        if self.use_bias:
            b = copy_tensor(torch.stack(bs), True)
            self.model = LinearClassifier(w, b)
        else:
            self.model = LinearClassifier(w)
        self.optim = torch.optim.LBFGS(self.model.parameters(), max_iter=5)
        self._train_model()
        # collect results for merged clusters
        loss_dec = old_losses - self.svm_closs
        for i, (q, p) in enumerate(self.qp_indice):
            self.merge_table[q, p] = loss_dec[i]
            self.merge_weights[q, p].copy_(self.model.weight[i])
            if self.use_bias:
                self.merge_bias[q, p].copy_(self.model.bias[i])
        # find the best merging clusters
        best_ind = self.merge_table.view(-1).argmax()
        q, p = best_ind // K, best_ind % K
        best_dec = float(self.merge_table[q, p])
        best_loss = float(self.svm_closs[loss_dec.argmax()])
        # print(self.merge_table)
        # print(q, p, best_dec, best_loss)
        # after merging, the new cluster has no merging data
        self.merge_table[q].fill_(-1)
        # apply the merging to the original model
        w = copy_tensor(orig_model.weight)
        w[q].copy_(self.merge_weights[q, p])
        w = copy_tensor(delete_index(w, p, 0), True)
        self.merge_table = delete_index(delete_index(self.merge_table, p, 0), p, 1)
        self.merge_weights = delete_index(delete_index(self.merge_weights, p, 0), p, 1)
        if self.use_bias:
            b = copy_tensor(orig_model.bias)
            b[q].copy_(self.merge_bias[q, p])
            b = copy_tensor(delete_index(b, p, 0), True)
            self.new_model = LinearClassifier(w, b)
            self.merge_bias = delete_index(delete_index(self.merge_bias, p, 0), p, 1)
        else:
            self.new_model = LinearClassifier(w)
        self.model = orig_model
        self.has_new_model = True
        return best_dec, q, p

    def _find_merging_clusters(self):
        self.has_new_model = False
        svm_type = self.objective.split("-")[1]
        t = self.class_size
        K = t.shape[0]
        args = [self.model_reps, self.mx, self.my, self.mby, self.class_size, svm_type]
        if self.metric == "sm-mcmld":
            if self.merge_table is None:
                self.merge_table = -torch.ones(K, K)  # maximum
                self.merge_weights = torch.zeros(K, K, self.model.weight.shape[1])
                self.merge_bias = torch.zeros(K, K)
            closs = multigpu_map(MLSClustering.mcmld_nograd, args + [2])[0]
            closs /= (t.unsqueeze(0) + t.unsqueeze(1)).clamp(min=1)
            closs = closs + closs.T
            max_closs, q, p = self._train_topk_candidate(closs)
            self.record.add("Maximum Cluster Loss", max_closs)
            return closs, max_closs, max_closs, q, p
        elif self.metric in ["newcbmcsvc", "cbmcsvc", "mcsvc", "mcmld"]:
            metric_func = getattr(MLSClustering, f"{self.metric}_nograd")
            closs = multigpu_map(metric_func, args)[0]
            if self.metric == "mcmld":
                closs /= (t.unsqueeze(0) + t.unsqueeze(1)).clamp(min=1)
            closs = closs + closs.T
            max_closs, (q, p) = upper_triangle_minmax(closs, "max")
            self.record.add("Maximum Cluster Loss", max_closs)
            return closs, max_closs, max_closs, q, p
        elif self.metric == "mcld":
            ld_mat = multigpu_map(MLSClustering.mcld_nograd, args + [1])[0]
            max_ld, (q, p) = upper_triangle_minmax(ld_mat, "max")
            self.record.add("Maximum Linear Density", max_ld)
            return ld_mat, max_ld, max_ld, q, p
        elif "heuristic" in self.metric:  # heuristic or heuristic-filter
            dm, dm_c, mm, mm_c = multigpu_map(MLSClustering.heuristic_nograd, args)
            dm /= dm_c.clamp(min=EPS)
            norm = torch.sqrt(mm_c)
            mm /= (norm.unsqueeze(1) * norm.unsqueeze(0)).clamp(min=EPS)
            p = int(dm.argmin())
            mm[p, p] = 0
            q = int(mm[p].argmax())
            self.dm = dm
            self.record.add("Minimum Deleting Heuristic", dm[p])
            self.record.add("Maximum Mergeing Heuristic", mm[p, q])
            return mm, dm[p], mm[p, q], q, p
        print(f"!> {self.metric} not implemented!")
        exit(0)

    def _closure(self):
        func_name, svm_type = self.objective.split("-")
        fb_func = getattr(MLSClustering, func_name)
        if "mcld" == func_name:
            self.l2_coef = 0.0
        self.model_reps = [self.model.clone().to(x) for x in self.mx]

        self.optim.zero_grad()
        args = [
            self.model_reps,
            self.mx,
            self.my,
            self.mby,
            self.class_size,
            svm_type,
            self.svm_coef,
        ]
        self.svm_loss, self.svm_closs = multigpu_map(fb_func, args)  # CPU value
        w = self.model_reps[0].weight
        reg_loss = 0
        if self.l2_coef > 0:
            reg_loss = reg_loss + self.l2_coef * torch.square(w).sum() / self.K / 2
        if self.l1_coef > 0:
            reg_loss = reg_loss + self.l1_coef * w.clamp(min=SL1_BETA).sum() / self.K
        reg_loss.backward()

        # recording losses
        grad_norm = self.model.weight.grad.norm(p=2, dim=1)
        self.record.add("Max L2 Norm W/Grad", min(1.0, grad_norm.max()))
        if self.use_bias:
            self.record.add("Max W/bias", self.model.bias.max())
        self.record.add("SVM Loss", min(self.svm_loss, 1.0))
        self.record.add("Reg Loss", reg_loss)
        return self.svm_loss + reg_loss.cpu()

    def _train_model(self):
        # cannot skip training, because svm closs need to be calculated
        for _ in range(self.max_iter):
            w = self.model.weight
            w_prev = copy_tensor(w)
            self.optim.step(self._closure)
            w_diff = float((w - w_prev).abs().max())
            sr = (w < 1e-4).float().sum() / (w.shape[0] * w.shape[1])
            self.record.add("K", w.shape[0])
            self.record.add("Max L2 Norm of W", w.norm(p=2, dim=1).max())
            self.record.add("L-inf Norm of delta_W", min(w_diff, 0.1))
            self.record.add("Sparse Ratio", sr)
            if w_diff < 1e-4:
                break

    def _training_step(self):
        self._train_model()
        full_metric, dm, mm, q, p = self._find_merging_clusters()

        # save merge tree
        n_cluster = int(self.model.weight.shape[0])
        self.merge_record[n_cluster] = {
            "Merge Metric": float(torch2numpy(mm)),
            "Delete Metric": float(torch2numpy(dm)),
            "Full Metric": full_metric,
            # preserved, removed, new label index
            "Merge Index": [q, p],
            "Merge Cluster Index": [
                self.idx2label[q],
                self.idx2label[p],
                self.cur_label,
            ],
        }
        state_dict = self.model.clone().detach().to("cpu").state_dict()
        self.merge_record[n_cluster].update(state_dict)
        torch.save(self.merge_record, f"{self.save_prefix}_tree.pth")
        self.idx2label[q] = self.cur_label  # p -> q, q -> cur_label
        del self.idx2label[p]
        self.cur_label += 1
        self.K -= 1

        # visualize to video
        if self.image_shape is not None:
            viz_label = self.my[-1].view(*self.feat_shape[:-1])[: self.n_viz]
            self.video_recorder(
                self.model.clone().to("cpu"),
                self.viz_image,
                self.viz_feat,
                viz_label.detach().cpu(),
                q,
                p,
            )
        else:
            self.video_recorder(self)

        # merge weights
        if not self.has_new_model:
            alpha = self.class_size[p] / self.class_size[[p, q]].sum()
            new_w = copy_tensor(self.model.weight)
            new_w[q] = new_w[p] * alpha + new_w[q] * (1 - alpha)
            new_w = delete_index(new_w, p, 0)
            if self.use_bias:
                new_b = copy_tensor(self.model.bias)
                new_b[q] = new_b[p] * alpha + new_b[q] * (1 - alpha)
                new_b = delete_index(new_b, p, 0)
                self.model.set_param(new_w, new_b)
            else:
                self.model.set_param(new_w)
        else:
            self.model = self.new_model
            self.model_reps = [self.model.clone().to(x) for x in self.mx]

        for d_id, label in enumerate(self.my):
            self.mby[d_id].fill_(0).scatter_(1, label.unsqueeze(1), 1)
            self.mby[d_id][:, q].add_(self.mby[d_id][:, p])
            delete_index_reorder(self.mby[d_id], p, 1)
            self.mby[d_id][:, self.model.weight.shape[0] :].fill_(0)
            self.my[d_id] = self.mby[d_id].argmax(1)
        self.class_size = sum([y.sum(0).cpu() for y in self.mby])
        self.class_size = self.class_size[: self.model.weight.shape[0]]

        optim = torch.optim.LBFGS(self.model.parameters(), max_iter=5)
        modify_optim(self.optim, optim, p)
        self.optim = optim
        plot_dict(self.record, f"{self.save_prefix}_record.png")

    def _calc_label(self):
        """Calculate label from self.model."""
        with torch.no_grad():
            model_reps = [self.model.clone().to(x) for x in self.mx]
            n_cluster = self.model.weight.shape[0]
            d_sample = self.mx[0].shape[0]
            if n_cluster * d_sample > self.MEM_UNIT:
                self.my = []
                for d_id, x in enumerate(self.mx):
                    model = model_reps[d_id]
                    L = x.shape[0] // 8
                    t = []
                    for i in range(8):
                        st, ed = L * i, L * (i + 1)
                        t.append(model(x[st:ed]).argmax(1).cpu())
                    self.my.append(torch.cat(t))
            else:
                self.my = [model(x).argmax(1) for model, x in zip(model_reps, self.mx)]
            self.mby = [
                torch.zeros(d_sample, n_cluster, device=y.device).scatter_(
                    1, y.unsqueeze(1), 1
                )
                for y in self.my
            ]
            self.class_size = sum([y.sum(0).cpu() for y in self.mby])
            # re-order the clusters according to its size (at first)
            reorder_indice = self.class_size.argsort(descending=True)
            self.class_size = self.class_size[reorder_indice]
            for d_id, by in enumerate(self.mby):
                cby = by.cpu().clone().detach()
                self.mby[d_id].copy_(cby[:, reorder_indice])
                self.my[d_id] = self.mby[d_id].argmax(1)
            w = copy_tensor(self.model.weight[reorder_indice], True)
            if self.use_bias:
                b = copy_tensor(self.model.bias[reorder_indice], True)
                self.model = LinearClassifier(w, b)
            else:
                self.model = LinearClassifier(w)
            self.optim = torch.optim.LBFGS(self.model.parameters(), max_iter=5)

    def fit(self, mx, mx2=None, mimage=None):
        """Cluster the data.
        Args:
            mx: A list of multigpu tensors.
            mimage: A list of image to be visualized.
        """
        self.n_gpu = len(mx)
        self.mx, self.mx2, self.mimage = mx, mx2, mimage

        self._calc_label()
        if self.metric == "heuristic-filter":
            print("=> Finding too unseparable clusters...")
            self._train_model()
            self._find_merging_clusters()
            dm = self.dm.detach().cpu().double()
            logit = torch.log((dm / (1 - dm).clamp(min=EPS)).clamp(min=EPS))
            # tau = logit.mean() + scipy.stats.norm.ppf(0.1) * logit.std()
            tau = logit.mean() - logit.std()
            keep_mask = logit > tau
            remove_indice = torch.nonzero(~keep_mask).view(-1).numpy().tolist()
            print(f"=> Removing: {remove_indice}")
            w = self.w_init[keep_mask, :]
            del self.my, self.mby, self.class_size
            torch.cuda.empty_cache()
            if self.use_bias:
                res = multigpu_kmeans(self.mx, w.cuda(), dist="euclidean", mx2=self.mx2)
                w = res[0].cuda()
                b = -0.5 * (w**2).sum(1)
                div = 2 * (-b).max()
                w = copy_tensor(w / div, True)
                b = copy_tensor(b / div, True)
                self.model = LinearClassifier(w, b)
            else:
                res = multigpu_kmeans(self.mx, w.cuda(), dist="arccos")
                w = copy_tensor(res[0].cuda(), True)
                self.model = LinearClassifier(w)
            self._calc_label()
            self.optim = torch.optim.LBFGS(self.model.parameters(), max_iter=5)
            self.K = self.K_init = w.shape[0]

        with torch.no_grad():
            # initial visualization
            if self.image_shape is not None:
                B = mx[0].shape[0] // (self.image_shape[0] * self.image_shape[1])
                self.feat_shape = [B] + self.image_shape + [self.D]
                self.n_viz = min(self.n_viz, B)
                self.viz_image = mimage[-1][: self.n_viz]
                self.viz_feat = mx[-1].view(*self.feat_shape)[: self.n_viz]
                self.viz_feat = self.viz_feat.view(-1, self.D).detach().cpu()
                model = self.model.clone().to("cpu")
                labels = model(self.viz_feat).argmax(1)
                labels = labels.view(-1, *self.feat_shape[1:-1])
                disp_img = visualize_segmentation(self.viz_image, labels)
                imwrite(f"{self.save_prefix}_kmeans.png", disp_img)
                is_norm = "ovr" not in self.objective
                self.video_recorder = SimpleVideoRecorder(
                    prefix=self.save_prefix, n_viz=self.n_viz, is_norm=is_norm
                )
            else:
                self.video_recorder = PlannarVideoRecorder(mx[0], self.save_prefix)

        for _ in tqdm(range(self.K_init - 1)):
            self._training_step()

        torch.save(self.merge_record, f"{self.save_prefix}_tree.pth")
        self.video_recorder.clean_sync()
        self.fitted = True

    def predict(self, x, n_cluster, ret_raw=False):
        """Predict the cluster assignments of a Tensor.
        Args:
          x : Tensor of shape (N, C).
          n_cluster : The number of clusters.
        Returns:
          cluster assignments of shape (N,).
        """
        if not self.fitted:
            self.log("!> Not fitted.")
            return

        dic = self.merge_record[n_cluster]
        if self.use_bias:
            model = LinearClassifier(dic["weight"], dic["bias"]).to(x)
        else:
            model = LinearClassifier(dic["weight"]).to(x)
        score = model(x)
        if ret_raw:
            return score
        return score.argmax(1)

    @staticmethod
    def cbmcsvc(model, feat, label, bl, class_size, loss_type, coef):
        """Class-Balanced Multi-Class Support Vector Classification."""
        acc_loss, acc_closs, N = 0, 0, 8
        B = feat.shape[0] // N
        t = class_size.to(feat)
        for i in range(N):
            st, ed = B * i, B * (i + 1)
            seg = model(feat[st:ed])
            s_true = torch.gather(seg, 1, label[st:ed].unsqueeze(1))  # (L, 1)
            seg = seg + 1 - bl[st:ed]
            max_values = seg.max(dim=1, keepdim=True).values
            margin = max_values - s_true
            if loss_type == "l2":
                svm_loss = torch.square(margin)
            svm_closs = (margin * bl[st:ed]).sum(0) / t.clamp(min=1)
            svm_loss = svm_closs.sum() / seg.shape[1] * coef
            svm_loss.backward()
            acc_loss += svm_loss.detach()
            acc_closs += svm_closs.detach()
        return acc_loss, acc_closs

    @staticmethod
    def cbmcsvc_nograd(model, feat, label, bl, class_size, loss_type):
        """Multi-Class Support Vector Classification."""
        acc_closs, N, K = 0, 4, model.weight.shape[0]
        B = feat.shape[0] // N
        t = class_size.to(feat).clamp(min=1)
        max_mask = torch.cuda.FloatTensor(B, K, device=feat.device)
        with torch.no_grad():
            for i in range(N):
                st, ed = B * i, B * (i + 1)
                seg = model(feat[st:ed])
                s_true = torch.gather(seg, 1, label[st:ed].unsqueeze(1))  # (L, 1)
                seg.add_(1).sub_(bl[st:ed])
                res = seg.max(dim=1, keepdim=True)
                max_indices, max_values = res.indices, res.values
                max_mask.fill_(0).scatter_(1, max_indices, 1)
                svm_loss = max_values - s_true
                if loss_type == "l2":
                    svm_loss.square_()
                svm_loss = svm_loss * max_mask  # (L, M) loss for each class
                # (M, M) loss for each class, decomposed into every classes
                acc_closs += torch.matmul(svm_loss.T, bl[st:ed]) / t.unsqueeze(0)
        return (acc_closs,)

    @staticmethod
    def newcbmcsvc(model, feat, label, bl, class_size, loss_type, coef):
        """Class-Balanced Multi-Class Support Vector Classification."""
        acc_loss, acc_closs, N = 0, 0, 8
        B = feat.shape[0] // N
        K = model.weight.shape[0]
        t = class_size.to(feat)
        div = (t.unsqueeze(0) + t.unsqueeze(1)).clamp(min=1)
        for i in range(N):
            st, ed = B * i, B * (i + 1)
            seg = model(feat[st:ed])
            s_true = torch.gather(seg, 1, label[st:ed].unsqueeze(1))  # (L, 1)
            seg = seg + 1 - bl[st:ed, :K]
            max_values = seg.max(dim=1, keepdim=True).values
            margin = max_values - s_true
            if loss_type == "l2":
                svm_loss = torch.square(margin)
            svm_closs = (margin * bl[st:ed, :K]).sum(0) / div
            svm_loss = 2 * svm_closs.sum() / seg.shape[1] * coef
            svm_loss.backward()
            acc_loss += svm_loss.detach()
            acc_closs += svm_closs.detach()
        return acc_loss, acc_closs

    @staticmethod
    def newcbmcsvc_nograd(model, feat, label, bl, class_size, loss_type):
        """Multi-Class Support Vector Classification."""
        acc_closs, N, K = 0, 4, model.weight.shape[0]
        B = feat.shape[0] // N
        K = model.weight.shape[0]
        t = class_size.to(feat)
        div = (t.unsqueeze(0) + t.unsqueeze(1)).clamp(min=1)
        max_mask = torch.cuda.FloatTensor(B, K, device=feat.device)
        with torch.no_grad():
            for i in range(N):
                st, ed = B * i, B * (i + 1)
                seg = model(feat[st:ed])
                s_true = torch.gather(seg, 1, label[st:ed].unsqueeze(1))  # (L, 1)
                seg.add_(1).sub_(bl[st:ed, :K])
                res = seg.max(dim=1, keepdim=True)
                max_indices, max_values = res.indices, res.values
                max_mask.fill_(0).scatter_(1, max_indices, 1)
                svm_loss = max_values - s_true
                if loss_type == "l2":
                    svm_loss.square_()
                svm_loss = svm_loss * max_mask  # (L, M) loss for each class
                # (M, M) loss for each class, decomposed into every classes
                acc_closs += torch.matmul(svm_loss.T, bl[st:ed, :K]) / div
        return (acc_closs,)

    @staticmethod
    def mcsvc(model, feat, label, bl, class_size, loss_type, coef):
        """Multiclass Support Vector Classification."""
        acc_loss, N = 0, 8
        B = feat.shape[0] // N
        K = model.weight.shape[0]
        total_size = class_size.sum()
        for i in range(N):
            st, ed = B * i, B * (i + 1)
            seg = model(feat[st:ed])
            s_true = torch.gather(seg, 1, label[st:ed].unsqueeze(1))  # (L, 1)
            seg = seg + 1 - bl[st:ed, :K]
            max_values = seg.max(dim=1, keepdim=True).values
            if loss_type == "l1":
                svm_loss = (max_values - s_true).sum()
            elif loss_type == "l2":
                svm_loss = torch.square(max_values - s_true).sum()
            elif loss_type == "sl1":
                svm_loss = F.smooth_l1_loss(
                    max_values, s_true, reduction="sum", beta=SL1_BETA
                )
            svm_loss = svm_loss / total_size * coef
            svm_loss.backward()
            acc_loss += svm_loss.detach()
        return acc_loss, acc_loss.clone()

    @staticmethod
    def mcsvc_nograd(model, feat, label, bl, class_size, loss_type):
        """Multi-Class Support Vector Classification."""
        acc_closs, N, K = 0, 4, model.weight.shape[0]
        B = feat.shape[0] // N
        max_mask = torch.cuda.FloatTensor(B, K, device=feat.device)
        with torch.no_grad():
            for i in range(N):
                st, ed = B * i, B * (i + 1)
                seg = model(feat[st:ed]).add_(1).sub_(bl[st:ed, :K])
                res = seg.max(dim=1, keepdim=True)
                max_indices, max_values = res.indices, res.values
                max_mask.fill_(0).scatter_(1, max_indices, 1)
                s_true = torch.gather(seg, 1, label[st:ed].unsqueeze(1))  # (L, 1)
                svm_loss = max_values - s_true
                if loss_type == "l2":
                    svm_loss.square_()
                svm_loss = svm_loss * max_mask  # (L, M) loss for each class
                # (M, M) loss for each class, decomposed into every classes
                acc_closs = acc_closs + torch.matmul(svm_loss.T, bl[st:ed, :K])
        return (acc_closs,)

    @staticmethod
    def ovrsvc(model, feat, label, bl, class_size, loss_type, coef):
        """One v.s. Rest binary Support Vector Classification."""
        acc_loss, acc_closs, N = 0, 0, 8
        B = feat.shape[0] // N
        K = model.weight.shape[0]
        total_size = class_size.sum()
        for i in range(N):
            st, ed = B * i, B * (i + 1)
            seg = model(feat[st:ed])
            with torch.no_grad():
                bl_ = bl[st:ed, :K]
                if feat.device != bl.device:
                    bl_ = bl_.to(feat.device)
                svm_label = bl_ * 2 - 1
            margin = 1 - svm_label * seg
            if loss_type == "l1":
                margin = margin.clamp(min=SL1_BETA)
            elif loss_type == "l2":
                margin = torch.square(margin.clamp(min=0))
            elif loss_type == "sl1":
                margin = F.smooth_l1_loss(
                    margin, torch.zeros_like(margin), reduction="none", beta=SL1_BETA
                )
            svm_closs = margin.sum(0)
            svm_loss = svm_closs.sum() / total_size * coef / K
            svm_loss.backward()
            acc_loss += svm_loss.detach()
            acc_closs += svm_closs.detach()
        return acc_loss, acc_closs

    @staticmethod
    def cbovrsvc(model, feat, label, bl, class_size, loss_type, coef):
        """One v.s. Rest Binary Minimum Linear Distance."""
        acc_loss, acc_closs, N = 0, 0, 8
        B = feat.shape[0] // N
        K = model.weight.shape[0]
        t = class_size.to(feat)
        total_size = t.sum()

        def worker(margin, div):
            if loss_type == "l2":
                svm_loss = torch.square(margin)
            closs = svm_loss.sum(0) / div
            svm_loss = coef / margin.shape[1] * closs.sum()
            svm_loss.backward()
            return svm_loss.detach(), closs.detach()

        for i in range(N):
            st, ed = B * i, B * (i + 1)
            seg = model(feat[st:ed])
            # positive margin
            margin = (1 - bl[st:ed, :K] * seg).clamp(min=SL1_BETA)
            svm_loss_p, closs_p = worker(margin, t.clamp(min=1))

            # negative margin
            seg = model(feat[st:ed])
            margin = (1 + (1 - bl[st:ed, :K]) * seg).clamp(min=SL1_BETA)
            svm_loss_n, closs_n = worker(margin, (total_size - t).clamp(min=1))
            acc_loss += svm_loss_p + svm_loss_n
            acc_closs += closs_p + closs_n
        return acc_loss, acc_closs

    @staticmethod
    def mcmld(model, feat, label, bl, class_size, loss_type, coef):
        """Multi-Class Maximum Linear Distance."""
        N = 8
        B = feat.shape[0] // N
        K = model.weight.shape[0]
        t = class_size.to(feat)
        div = (t.unsqueeze(0) + t.unsqueeze(1)).clamp(min=1)
        acc_loss, acc_closs = 0, 0
        for i in range(N):
            st, ed = B * i, B * (i + 1)
            seg = model(feat[st:ed])
            s_true = torch.gather(seg, 1, label[st:ed].unsqueeze(1))  # (L, 1)
            margin = 1 - bl[st:ed, :K] + seg - s_true  # (L, M)
            if loss_type == "l2":
                margin = torch.square(margin.clamp(min=0))
            elif loss_type == "sl1":
                margin = F.smooth_l1_loss(
                    margin, torch.zeros_like(margin), reduction="none", beta=SL1_BETA
                )
            elif loss_type == "l1":
                margin = margin.clamp(min=SL1_BETA)
            closs = torch.matmul(margin.permute(1, 0), bl[st:ed, :K]) / div
            svm_loss = 2 / seg.shape[1] * closs.sum()
            acc_loss += svm_loss.detach()
            acc_closs += closs.detach()
            svm_loss.backward()
        return acc_loss, acc_closs

    @staticmethod
    def mcmld_nograd(model, feat, label, bl, class_size, loss_type, width=1):
        """Multi-Class Maximum Linear Distance."""
        N, K = 2, model.weight.shape[0]
        B = feat.shape[0] // N
        closs = torch.zeros(K, K).to(feat)  # class linear margin
        with torch.no_grad():
            for i in range(N):
                st, ed = B * i, B * (i + 1)
                seg = model(feat[st:ed])
                s_true = torch.gather(seg, 1, label[st:ed].unsqueeze(1))  # (L, 1)
                seg.sub_(s_true).sub_(bl[st:ed, :K] * width).add_(width).clamp_(min=0)
                if loss_type == "l2":
                    seg.square_()
                closs.add_(torch.matmul(seg.permute(1, 0), bl[st:ed, :K]))
        return (closs,)

    @staticmethod
    def mcld(model, feat, label, bl, class_size, loss_type, coef):
        """Multi-Class Minimum Linear Density Classification."""
        N = 8
        B = feat.shape[0] // N
        K = model.weight.shape[0]
        t = class_size.to(feat).clamp(min=1)
        acc_loss = 0
        w = model.weight
        min_width = 1.0 / 10
        target = torch.zeros(w.shape[0], w.shape[0]).fill_(min_width).to(w)
        for i in range(N):
            st, ed = B * i, B * (i + 1)
            seg = model(feat[st:ed])
            s_true = torch.gather(seg, 1, label[st:ed].unsqueeze(1))  # (L, 1)
            margin = (1 - bl[st:ed, :K] + seg - s_true).clamp(min=0)
            diff_w_norm = ((w.unsqueeze(1) - w.unsqueeze(0)) ** 2).sum(-1)
            if loss_type == "l2":
                margin = torch.square(margin)
            elif loss_type == "sl1":
                margin = F.smooth_l1_loss(
                    margin, torch.zeros_like(margin), reduction="none", beta=SL1_BETA
                )
                diff_w_norm = torch.sqrt(diff_w_norm.clamp(min=EPS))  # (K, K)
            class_loss = torch.matmul(margin.permute(1, 0), bl[st:ed, :K])
            diff_norm = (
                F.smooth_l1_loss(
                    diff_w_norm.clamp(min=min_width),
                    target,
                    beta=SL1_BETA,
                    reduction="none",
                )
                + min_width
            )
            svm_loss = (class_loss / t.unsqueeze(0) * diff_norm).sum()
            svm_loss.backward()
            acc_loss += svm_loss.detach()
        return acc_loss, acc_loss.clone()

    @staticmethod
    def heuristic_nograd(model, feat, label, bl, class_size, loss_type):
        """The worker for calculating merging metrics in a thread.
        Args:
        feat: (L, C)
        w: (K, C)
        """
        K = model.weight.shape[0]
        dm, dm_c, mm, mm_c, N = 0, 0, 0, 0, 8
        B = feat.shape[0] // N
        with torch.no_grad():
            for i in range(N):
                st, ed = B * i, B * (i + 1)
                seg = model(feat[st:ed])
                bl_ = bl[st:ed, :K]
                bl_ = bl_.to(seg.device) if seg.device != bl.device else bl_
                # SVM accuracy (IoU)
                gt = bl_ > 0
                dt = seg > 0
                dm += (dt & gt).sum(0).float()
                dm_c += (dt | gt).sum(0).float()
                # ECoS
                dt = (seg.clamp(-1, 1) + 1) / 2
                mm += torch.matmul(dt.permute(1, 0), dt)
                mm_c += torch.square(dt).sum(0)
        return dm, dm_c, mm, mm_c

    @staticmethod
    def ld_nograd(model, feat, label, bl, class_size, loss_type, coef):
        """Calculate Linear Density per class."""
        N = 8
        B = feat.shape[0] // N
        acc_loss = 0
        K = model.weight.shape[0]
        t = class_size.to(feat).clamp(min=1)
        with torch.no_grad():
            w = model.weight
            diff_w_norm = ((w.unsqueeze(1) - w.unsqueeze(0)) ** 2).sum(-1)
            for i in range(N):
                st, ed = B * i, B * (i + 1)
                seg = model(feat[st:ed])
                s_true = torch.gather(seg, 1, label[st:ed].unsqueeze(1))  # (L, 1)
                margin = (1 - bl[st:ed, :K] + seg - s_true).clamp(min=0)  # (L, K)
                if loss_type == "l2":
                    diff_w_norm = diff_w_norm.clamp(min=0.1)
                    margin = torch.square(margin)
                elif loss_type == "sl1":
                    diff_w_norm = torch.sqrt(diff_w_norm).clamp(min=0.1)
                    margin = F.smooth_l1_loss(
                        margin,
                        torch.zeros_like(margin),
                        reduction="none",
                        beta=SL1_BETA,
                    )
                elif loss_type == "l0":
                    diff_w_norm = torch.sqrt(diff_w_norm).clamp(min=0.1)
                    margin = (margin > 0).float()
                class_loss = torch.matmul(margin.permute(1, 0), bl[st:ed])
                acc_loss = acc_loss + class_loss / t.unsqueeze(0) * diff_w_norm
        return [acc_loss]


class MLSSC(MLSClustering):
    """Using Spectral Clustering with linear separability heuristic."""

    def _find_merging_clusters(self):
        """Find a pair of clusters to be merged."""
        self.has_new_model = False
        svm_type = self.objective.split("-")[1]
        t = self.class_size
        args = [self.model_reps, self.mx, self.my, self.mby, self.class_size, svm_type]
        if self.metric == "ncut":
            closs = multigpu_map(MLSClustering.mcmld_nograd, args + [2])[0]
            closs /= (t.unsqueeze(0) + t.unsqueeze(1)).clamp(min=1)
            self.similarity = torch2numpy(closs + closs.T)
            self.scalg = SpectralClustering(
                self.final_K, affinity="precomputed", assign_labels="discretize"
            )
            self.y_perm = self.scalg.fit_predict(self.similarity)
            self.y_perm_w = permute_matrix(self.y_perm)[1]
            return self.similarity
        print(f"!> {self.metric} not implemented!")

    def _train_model(self):
        # cannot skip training, because svm closs need to be calculated
        for _ in tqdm(range(self.max_iter)):
            w = self.model.weight
            w_prev = copy_tensor(w)
            self.optim.step(self._closure)
            w_diff = float((w - w_prev).abs().max())
            sr = (w < 1e-4).float().sum() / (w.shape[0] * w.shape[1])
            self.record.add("K", w.shape[0])
            self.record.add("Max L2 Norm of W", w.norm(p=2, dim=1).max())
            self.record.add("L-inf Norm of delta_W", min(w_diff, 0.1))
            self.record.add("Sparse Ratio", sr)
            if w_diff < 1e-4:
                break

    def fit(self, mx, mx2=None, mimage=None):
        """Cluster the data.
        Args:
            mx: A list of multigpu tensors.
            mimage: A list of image to be visualized.
        """
        self.n_gpu = len(mx)
        self.mx, self.mx2, self.mimage = mx, mx2, mimage

        self._calc_label()
        self._train_model()

        with torch.no_grad():
            # initial visualization
            if self.image_shape is not None:
                B = mx[0].shape[0] // (self.image_shape[0] * self.image_shape[1])
                self.feat_shape = [B] + self.image_shape + [self.D]
                self.n_viz = min(self.n_viz, B)
                viz_image = mimage[-1][: self.n_viz]
                viz_feat = mx[-1].view(*self.feat_shape)[: self.n_viz]
                viz_feat = viz_feat.view(-1, self.D).detach().cpu()
                model = self.model.clone().to("cpu")
                labels = model(viz_feat).argmax(1)
                labels = labels.view(-1, *self.feat_shape[1:-1])
                disp_img = visualize_segmentation(viz_image, labels)
                imwrite(f"{self.save_prefix}_kmeans.png", disp_img)

        for K in [30, 40, 50]:
            print(K)
            self.final_K = K
            self._calc_label()
            self._find_merging_clusters()
            with torch.no_grad():
                w = self.y_perm_w
                for d_id, by in enumerate(self.mby):
                    self.mby[d_id] = torch.matmul(by, w.to(by))
                    self.my[d_id] = self.mby[d_id].argmax(1)
            label = self.my[-1].view(*self.feat_shape[:-1])
            label_viz = segviz_torch(label[: self.n_viz].clone().detach().cpu())

            # class map of visualized images
            disp = []
            for i in range(min(viz_image.shape[0], 15)):
                disp.extend([viz_image[i : i + 1], label_viz[i : i + 1]])
            disp = torch.cat(bu(disp, 128))
            frame = vutils.make_grid(disp, nrow=6, padding=2, pad_value=0)
            frame = torch2image(frame, "[0,1]")
            imwrite(f"{self.save_prefix}_{self.final_K}.png", frame)
            del (
                self.my,
                self.mby,
                self.class_size,
                self.similarity,
                self.y_perm,
                self.y_perm_w,
            )
            del label, label_viz, disp, by, frame
            torch.cuda.empty_cache()
            # input()
        plot_dict(self.record, f"{self.save_prefix}_record.png")


def load_as_slse(method, layers, resolution, expr_dir, g_name, seed, layer_idx="auto", custom_path=False):
    """Load clustering models as SLSE."""

    if method == "kmeans":
        prefix = expr_dir if custom_path else f"{expr_dir}/kmeans"
        kmeans_path = glob.glob(f"{prefix}/{g_name}*_i{layer_idx}_*{seed}.pth")
        if len(kmeans_path) == 0:
            print(f"=> K-means file {g_name}*_i{layer_idx}_*{seed} not found, skip.")
            return
        #print(kmeans_path)
        kmeans_file = torch.load(kmeans_path[0])
        return SimpleLSE.load_as_lse_bias(
            kmeans_file["euclidean"], kmeans_file["arccos"], layers, resolution
        )

    if method == "klish":
        args = f"i{layer_idx}_b%d_heuristic_ovrsvc-l2"
        prefix = expr_dir if custom_path else f"{expr_dir}/klish"
        file_format = f"{prefix}/{g_name}_{args}_{seed}_tree.pth"
        nobias_file = (
            torch.load(file_format % 0) if os.path.exists(file_format % 0) else None
        )
        bias_file = (
            torch.load(file_format % 1) if os.path.exists(file_format % 1) else None
        )
        if nobias_file is None and bias_file is None:
            print(f"=> KLiSH file {file_format} not found, skip.")
            return
        return SimpleLSE.load_as_lse_bias(bias_file, nobias_file, layers, resolution)

    if method == "ahc":
        prefix = expr_dir if custom_path else f"{expr_dir}/ahc"
        file_format = f"{prefix}/{g_name}_ltrunc-wp_i{layer_idx}_N32_S64_%s_{seed}.pth"
        print(file_format)
        nobias_file = (
            torch.load(file_format % "arccos")
            if os.path.exists(file_format % "arccos")
            else None
        )
        bias_file = (
            torch.load(file_format % "euclidean")
            if os.path.exists(file_format % "euclidean")
            else None
        )
        if not nobias_file and not bias_file:
            print(f"=> AHC file: {file_format} not found, skip.")
            return
        return SimpleLSE.load_as_lse_bias(bias_file, nobias_file, layers, resolution)

    if method == "kasp":
        kmeans_path = glob.glob(f"{expr_dir}/kmeans/{g_name}*{seed}.pth")
        kasp_path = glob.glob(f"{expr_dir}/kasp/{g_name}*{seed}.json")
        if len(kmeans_path) == 0 or len(kasp_path) == 0:
            print(f"=> KASP {g_name}-{seed} not found, skip.")
            return
        kmeans_file = torch.load(kmeans_path[0])
        with open(kasp_path[0], "r", encoding="ascii") as f:
            kasp_dic = json.load(f)
        init_k = 100
        w = kmeans_file["euclidean"][init_k]["weight"]
        b = kmeans_file["euclidean"][init_k]["bias"]
        alg_bias = KASP(SimpleLSE(w, layers, resolution, b).cuda())
        alg_bias.restore(kasp_dic["euclidean"])
        w = kmeans_file["arccos"][init_k]["weight"]
        alg_nobias = KASP(SimpleLSE(w, layers, resolution).cuda())
        alg_nobias.restore(kasp_dic["arccos"])
        return {"bias": alg_bias, "nobias": alg_nobias}


def permute_matrix(label_perm, n_cluster=-1):
    """Get a permute matrix from a label permutation list.
    Args:
        label_perm: A list of numbers. The i-th element indicate its new label.
    Returns:
        old2new: The subclass of each new label;
        mat: (N_OLD, N_NEW) the permutation matrix.
    """
    label_perm = torch.Tensor(label_perm).long()
    n_cluster = label_perm.max() + 1 if n_cluster < 0 else n_cluster
    pmat = torch.stack([(label_perm == i).float() for i in range(n_cluster)], 1)
    old2new = [np.where(label_perm == i)[0] for i in range(n_cluster)]
    return old2new, pmat


def multigpu_kmeans(
    mx: list, w: Tensor, n_iter=2000, tol=1e-3, dist="arccos", mx2=None, record=None
):
    """EM training of hard assignment KMeans with arccos.
    Args:
        mx: A list of Tensor of shape (BHW, C)
        w: A Tensor of shape (num_classes, num_features)
        n_iter: The maximum iteration number.
        tol: The tolerance measured in max-norm (L-inf).
        dist: The metric of K-means. Can be euclidean and arccos.
        mx2: The squared norm of mx, a list of Tensors of shape (BHW,)
        record: Whether to append training record to an old one.
    Returns:
        w: Tensor in CPU, the centroids.
        unormed_w: Tensor in CPU, the unormalized centroids for arccos.
                    unormed_w will be the same as w for euclidean metric.
        count: The number of samples belonging to each centroid.
        record: The record of K-means.
    """
    w = copy_tensor(w)  # deep copy
    with torch.no_grad():
        record = DictRecorder()
        for _ in tqdm(range(n_iter)):
            threads, icdist, ws, count = [], [], [], []
            for d, x in enumerate(mx):
                x2 = None if mx2 is None else mx2[d]
                threads.append(
                    GeneralThread(
                        _hard_kmeans_em_step,
                        X=x,
                        W=copy_tensor(w).to(x),
                        dist=dist,
                        X2=x2,
                    )
                )
                threads[-1].start()
            for d in range(len(mx)):
                threads[d].join()
                w_, count_, icdist_ = threads[d].res[:3]
                count.append(count_.cuda())
                ws.append(w_.cuda())
                icdist.append(icdist_.cuda())
            icdist = torch.stack(icdist)  # (N_GPU, K)
            ws = torch.stack(ws)
            count = torch.stack(count)
            dc_count = count.sum(0)  # (K,)
            dc_coef = count / dc_count.unsqueeze(0).clamp(min=EPS)
            icdist = (icdist * dc_coef).sum(0).mean()
            unormed_w = (ws * dc_coef.unsqueeze(2)).sum(0)

            if dist == "arccos":
                norm = unormed_w.norm(p=2, dim=1, keepdim=True)
                normed_w = unormed_w / norm.clamp(min=EPS)
                max_cshift = (normed_w - w).abs().max()
                w.copy_(normed_w)
            else:
                max_cshift = (unormed_w - w).abs().max()
                w.copy_(unormed_w)

            record.add("Intra-class Distance", icdist)
            record.add("Center Shift", max_cshift)

            if max_cshift < tol:  # reach tolerance and early stop
                break
    return w.cpu(), unormed_w.cpu(), count.cpu(), icdist.cpu(), record


def _centroids_from_assignments(mx, mby, dist):
    """Generate centriods from cluster assignments."""

    def _worker(x, by):
        with torch.no_grad():
            w = torch.zeros(by.shape[1], x.shape[1]).to(x)
            count = torch.zeros(by.shape[1])
            for i in range(by.shape[1]):
                indice = torch.nonzero(by[:, i]).squeeze(1).to(x.device)
                count[i] = indice.shape[0]
                if indice.shape[0] < 1:
                    continue
                B, T = 8192, indice.shape[0]
                v = 0
                for j in range(T // B + 1):
                    st, ed = B * j, min(B * (j + 1), T)
                    if st >= T:
                        break
                    v += x[indice[st:ed]].mean(0) * (ed - st) / T
                # w[i] = torch.index_select(x, 0, indice).mean(0)
                w[i].copy_(v)
        return w, count

    threads, ws, count = [], [], []
    for x, by in zip(mx, mby):
        threads.append(GeneralThread(_worker, x, by))
        threads[-1].start()
    for d in range(len(mx)):
        threads[d].join()
        w, count = threads[d].res
        count.append(count.cuda())
        ws.append(w.cuda())
    ws, count = torch.stack(ws), torch.stack(count)
    coef = count / count.sum(0, keepdim=True).clamp(min=1)
    unormed_w = (ws * coef.unsqueeze(2)).sum(0)
    if dist == "arccos":
        norm = unormed_w.norm(p=2, dim=1, keepdim=True).clamp(min=EPS)
        w = unormed_w / norm
    return w.detach().cpu()


def _hard_kmeans_em_step(X: Tensor, W: Tensor, dist: str, X2=None):
    """Perform an EM step of hard KMeans.
    Args:
      X: The data to be clustered, Tensor in GPU of shape (N, C).
      W: The centroids, Tensor in GPU of shape (K, C).
      dist: The distance type. Can be arccos and euclidean.
      X2: To accelerate the calculation of euclidean distance.
          Set to None will make the algorithm slower.
    Returns:
      W: The updated centroids;
      count: The samples belonging to each centroid;
      L: The intra-cluster distance for each cluster.
      D: The distance matrix.
    """
    icdist, count = torch.zeros(W.shape[0]), torch.zeros(W.shape[0])
    with torch.no_grad():
        if dist == "euclidean":
            D = torch.matmul(X, W.permute(1, 0)).mul_(-2)
            if X2 is None:
                X2 = (X * X).sum(1)
            D.add_(X2.unsqueeze(1))
            D.add_((W * W).sum(1).unsqueeze(0))
            D.sqrt_()
        else:
            D = pairwise_dist(X, W, dist)
        y_pred = D.argmin(1)
        # res = D.min(1)
        # minD2, min_indice = res.values, res.indices
        # by = torch.cuda.
        # by[:, : idx + 1].fill_(0).scatter_(1, min_indice.unsqueeze(1), 1)

        for i in range(W.shape[0]):
            indice = torch.nonzero(y_pred == i).squeeze()
            if len(indice.shape) == 0 or indice.shape[0] < 1:
                continue
            count[i] = indice.shape[0]
            icdist[i] = D[indice, i].sum() / count[i].clamp(min=1)
            B, T = 2**16, indice.shape[0]
            v = 0
            for j in range(T // B + 1):
                st, ed = B * j, min(B * (j + 1), T)
                if st >= T:
                    break
                v += (ed - st) / T * X[indice[st:ed]].mean(0)
            W[i].copy_(v)
    return W, count, icdist, D


def _intra_cluster_distance(X: Tensor, W: Tensor, dist: str, X2=None):
    """Calculate the intra-cluster distance.
    Should be executed within nograd environment.
    Args:
      X: The data to be clustered, Tensor in GPU of shape (N, C).
      W: The centroids, Tensor in GPU of shape (K, C).
      dist: The distance type. Can be arccos and euclidean.
      X2: To accelerate the calculation of euclidean distance.
          Set to None will make the algorithm slower.
    Returns:
      icdist: The intra cluster distance;
      count: The samples belonging to each centroid.
    """
    if dist == "euclidean":
        D = torch.matmul(X, W.permute(1, 0)).mul_(-2)
        if X2 is None:
            X2 = (X * X).sum(1)
        D.add_(X2.unsqueeze(1))
        D.add_((W * W).sum(1).unsqueeze(0))
        D.clamp_(min=EPS).sqrt_()
    else:
        D = pairwise_dist(X, W, dist)
    y_bl = torch.zeros(X.shape[0], W.shape[0]).to(X.device)
    y_bl.scatter_(1, D.argmin(1, keepdim=True), 1)
    count = y_bl.sum(0)
    icdist = torch.einsum("ij,ij->j", y_bl, D) / count.clamp(min=1)
    return icdist, count, D


def multigpu_kmeanspp_init(
    mX: list, K: int, dist="euclidean", mX2=None, rng=None, n_repeat=1
):
    """Find an initialization using K-means++.
    Args:
        mX: The data to be clustered, a list of multi-GPU Tensor of shape (N, C);
        K: The number of clusters;
        mX2: The squared norm of mX, a list of multi-GPU Tensor of shape (N,).
             Used for accelerating the calculation in euclidean metric.
        dist: The metric of K-means. Can be euclidean and arccos.
        rng: Set the np.random.RandomState instance for reproducibility.
        n_repeat: trials of each sampling.
    Returns:
        w: The initialized centroids, Tensor in GPU:0 of shape (K, C).
    """
    N = sum([X.shape[0] for X in mX])
    B, C = mX[0].shape  # assume all X have the same shape
    w = torch.zeros(K, C).to(mX[0])
    minD2 = torch.zeros(N).cuda()
    mD2 = [torch.zeros((B, K)).to(x) for x in mX]
    mby = [torch.zeros((B, K)).to(x) for x in mX]
    rng = np.random.RandomState(1997) if rng is None else rng
    # randomly select the first centroid
    ind = rng.randint(0, N)
    w[0].copy_(mX[ind // B][ind % B].cuda())

    def _worker(A, B, D2, by, idx, A2=None):
        if dist == "euclidean":
            cD = torch.matmul(A, B.permute(1, 0)).mul_(-2)
            if A2 is None:
                A2 = (A * A).sum(1)
            cD.add_(A2.unsqueeze(1))
            cD.add_((B * B).sum(1).unsqueeze(0))
        else:
            cD = pairwise_dist(A, B, dist)
        D2[:, idx].copy_(cD.squeeze())
        res = D2[:, : idx + 1].min(1)
        minD2_, min_indice = res.values, res.indices
        by[:, : idx + 1].fill_(0).scatter_(1, min_indice.unsqueeze(1), 1)
        count = by[:, : idx + 1].sum(0)
        # reuse variable by to save memory
        by[:, : idx + 1].mul_(D2[:, : idx + 1])
        icdist2 = by[:, : idx + 1].sum(0) / count.clamp(min=1)
        return minD2_, icdist2, count

    for i in tqdm(range(1, K)):
        min_icdist2, min_v = 1e10, None
        for r in range(n_repeat + 1):
            # compute previous distance matrix at r = 0
            dist_idx = i - 1 if r == 0 else i
            threads, icdist2, count = [], [], []
            for j, x in enumerate(mX):
                X2 = None if mX2 is None else mX2[j]
                v_ = w[dist_idx].unsqueeze(0).to(x)
                threads.append(
                    GeneralThread(_worker, x, v_, mD2[j], mby[j], dist_idx, X2)
                )
                threads[-1].start()
            offset = 0
            for thr in threads:
                thr.join()
                minD2_, icdist2_, count_ = thr.res
                minD2[offset : offset + minD2_.shape[0]].copy_(minD2_)
                icdist2.append(icdist2_.cpu())
                count.append(count_.cpu())
                offset += minD2_.shape[0]
            icdist2 = torch.stack(icdist2)
            count = torch.stack(count)  # (N_GPU, K)
            dc_count = count.sum(0)  # (K,)
            dc_coef = count / dc_count.unsqueeze(0).clamp(min=1)
            icdist2 = float((icdist2 * dc_coef).sum(0).mean())
            if icdist2 < min_icdist2:
                min_icdist2 = icdist2
                min_v = w[dist_idx].clone().detach()
            if r <= n_repeat:
                cumD2 = torch.cumsum(minD2, 0)
                scale = float(cumD2[-1])
                ind = int(torch.searchsorted(cumD2, rng.rand() * scale))
                w[i].copy_(mX[ind // B][ind % B])  # next trial
            else:  # select the best one
                w[i].copy_(min_v)
    return w
