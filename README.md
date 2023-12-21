# KLiSH

[Project Page](https://atlantixjj.github.io/KLiSH/) | [Paper](https://arxiv.org/pdf/2211.16710.pdf)

**Extracting Semantic Knowledge from GANs with Unsupervised Learning**
Jianjin Xu, Zhaoxiang Zhang, and Xiaolin Hu.
In TPAMI.


## Setup

### Installation

We suggest using `Anaconda3` virtual environment to configure.
We use PyTorch version 1.7.0 and CUDA 11.0.
For other versions, we do not gauruantee the correctness of this code.
The installation steps are as follows:

1. Install PyTorch 1.7.0 `conda install pytorch cudatoolkit torchvision -c pytorch`.
2. Install other requirements `pip install -r requirements.txt`.
3. Run `make_folders.sh` to create neccessary folders.

### Pretrained models and datasets

1. Download the GAN model from [Dropbox](https://www.dropbox.com/sh/ro56h2hv5oltfum/AAA9G5_qwmQ1STG40g3QUGpva?dl=0) on which you want to extract semantics and put them into `models/pretrained/pytorch`.
3. Download [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada) models (afhqcat.pkl, afhqdog.pkl, afhqwild.pkl, metfaces.pkl) and put them under `thirdparty/stylegan2_ada/pretrained`.

### Compare to few-shot methods

If you want to compare our UFS method to few-shot learning methods (which is optional, only used as baselines), you need these additional steps:

1. Install [Pytorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding).
2. Download `faceparse_deeplabv3+_c15.pth` from the [Dropbox](https://www.dropbox.com/sh/ro56h2hv5oltfum/AAA9G5_qwmQ1STG40g3QUGpva?dl=0) and put it in `predictors/pretrained/`.
3. Download CelebAMask-HQ from `https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html` and extract it under `data` folder. Rename `CelebA-HQ-img` to `image` and `CelebAHQMask-HQ-mask-anno` to `raw_label`.
4. Run `python script/celebamask_merge.py` to construct the 15 classes annotations.
5. Also, download `list_part_eval.txt` from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and put it under `data/CelebAMask-HQ`.


## Reproduce results in the paper


1. Run K-means on GAN models to obtain initial clustering.

```bash
# run on all GAN models with 5 repeats
python submit.py --func cluster_kmeans --gpu <gpu_id>
```

The results and visualizations will be stored in `expr/cluster/kmeans` folder.

2. Run KLiSH on GAN models. Use one of the following command:

```bash
# run on all GAN models with repeats
python submit.py --func cluster_klish --gpu <gpu_id>
```

The results and visualizations will be stored in `expr/cluster/klish`.
You can optionally open `*class.mp4` video to decide how many clusters do you think is proper for downstream tasks.

3. Generate datasets with image and segmentation masks. The number of clusters can be modified by changing the `selected_classes` dictionary in L.65 of `script/make_dataset_multilabel.py`. You can use the following command to generate the dataset:

```bash
python script/make_dataset_multilabel.py --G-name <the name of generator>
```

4. After the datasets have been generated, you can train the downstream tasks.

To train UFS, run `python submit.py --func train_image_seg`.

To train USCS, enter the `thirdparty/spade` directory and run according to the `thirdparty/spade/READMD.md`.
