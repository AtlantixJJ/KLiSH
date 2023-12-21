"""Merge CelebAMask-HQ 19 classes to 15 classes.
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from lib.visualizer import segviz_numpy

# list1
# label_list = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']
# list2
# 4->3, 6->5, 8->7
label_list = [
    "skin",
    "nose",
    "eye_g",
    "l_eye",
    "r_eye",
    "l_brow",
    "r_brow",
    "l_ear",
    "r_ear",
    "mouth",
    "u_lip",
    "l_lip",
    "hair",
    "hat",
    "ear_r",
    "neck_l",
    "neck",
    "cloth",
]
label_list = [
    "skin",
    "nose",
    "eye_g",
    ["l_eye", "r_eye"],
    ["l_brow", "r_brow"],
    ["l_ear", "r_ear"],
    "mouth",
    "u_lip",
    "l_lip",
    "hair",
    "hat",
    "ear_r",
    ["neck_l", "neck"],
    "cloth",
]

basedir = "data/CelebAMask-HQ"
folder_base = f"{basedir}/raw_label"
folder_save = f"{basedir}/label_{len(label_list) + 1}"
folder_viz = f"{basedir}/labelviz_{len(label_list) + 1}"
img_num = 30000

if not os.path.exists(folder_save):
    os.makedirs(folder_save)

for k in tqdm(range(img_num)):
    folder_num = int(k / 2000)
    im_base = np.zeros((512, 512), dtype="uint8")
    for idx, label_group in enumerate(label_list):
        if type(label_group) is str:
            label_group = [label_group]
        for label in label_group:
            filename = os.path.join(
                folder_base,
                str(folder_num),
                str(k).rjust(5, "0") + "_" + label + ".png",
            )
            if os.path.exists(filename):
                im = cv2.imread(filename)
                im = im[:, :, 0]
                im_base[im != 0] = idx + 1
    filename_save = os.path.join(folder_save, str(k) + ".png")
    cv2.imwrite(filename_save, im_base)
    if k < 20:
        filename_viz = os.path.join(folder_viz, str(k) + ".png")
        label_viz = segviz_numpy(im_base)
        cv2.imwrite(filename_viz, label_viz[..., ::-1])
