"""Datasets."""
import torch, os, random, cv2
import pytorch_lightning as pl
import numpy as np
from skimage.measure import regionprops
from scipy.io import loadmat

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def pil_read(fpath):
    with open(os.path.join(fpath), "rb") as f:
        img = Image.open(f)
        img.load()
    return img


def dataloader_from_name(name, use_split="train", label_set=""):
    if name == "CelebAHQ-Mask":
        ds = CelebAMaskDataset(
            "data/CelebAMask-HQ", use_split=use_split, label_folder="label_c15"
        )
        ds.n_class = 15
        ds.image_size = 512
    if name == "stylegan2_ffhq":
        ds = ImageSegmentationDataset(
            "expr/data/stylegan2_ffhq_s1113",
            use_split=use_split,
            label_folder=f"label_{label_set}",
        )
        ds.n_class = int(label_set[label_set.rfind("c") + 1 :])
        ds.image_size = 512
    return DataLoader(ds)


class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
      brightness_delta (int): delta of brightness.
      contrast_range (tuple): range of contrast.
      saturation_range (tuple): range of saturation.
      hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(0, 2):
            return self.convert(
                img, beta=random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0, 2):
            return self.convert(
                img, alpha=random.uniform(self.contrast_lower, self.contrast_upper)
            )
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0, 2):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower, self.saturation_upper),
            )
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0, 2):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(int)
                + random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def __call__(self, img):
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(brightness_delta={self.brightness_delta}, "
            f"contrast_range=({self.contrast_lower}, "
            f"{self.contrast_upper}), "
            f"saturation_range=({self.saturation_lower}, "
            f"{self.saturation_upper}), "
            f"hue_delta={self.hue_delta})"
        )
        return repr_str


class SimpleDataset(torch.utils.data.Dataset):
    """
    Currently label is not available
    """

    def __init__(self, data_path, size=None, transform=transforms.ToTensor()):
        self.size = size
        self.data_path = data_path
        self.transform = transform

        self.files = sum(
            [
                [file for file in files if ".jpg" in file or ".png" in file]
                for path, dirs, files in os.walk(data_path)
                if files
            ],
            [],
        )
        self.files.sort()

    def __getitem__(self, idx):
        fpath = self.files[idx]
        with open(os.path.join(self.data_path, fpath), "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.size:
                img = img.resize(self.size, Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)


class ImageSegmentationDataset(torch.utils.data.Dataset):
    """
    Currently label is not available
    """

    def __init__(
        self, data_path, use_split="train", image_folder="image", label_folder="label"
    ):
        self.use_split = use_split
        self.root_dir = data_path
        self.image_dir = f"{data_path}/{image_folder}"
        self.label_dir = f"{data_path}/{label_folder}"

        self.distortion = PhotoMetricDistortion()

        self.imagefiles = sum(
            [
                [file for file in files if ".jpg" in file or ".png" in file]
                for path, dirs, files in os.walk(self.image_dir)
                if files
            ],
            [],
        )
        self.imagefiles.sort()
        self.labelfiles = sum(
            [
                [file for file in files if ".png" in file]
                for path, dirs, files in os.walk(self.label_dir)
                if files
            ],
            [],
        )
        self.labelfiles.sort()

        self.rng = np.random.RandomState(1)
        self.indice = np.arange(0, len(self.labelfiles))
        self.rng.shuffle(self.indice)
        self.train_size = int(len(self.labelfiles) * 0.8)
        self.val_size = int(len(self.labelfiles) * 0.1)
        if use_split == "train":
            self.indice = self.indice[: self.train_size]
        elif use_split == "val":
            self.indice = self.indice[self.train_size : self.train_size + self.val_size]
        else:
            self.indice = self.indice[self.train_size + self.val_size :]

    def transform(self, image, label):
        """Image preprocessing."""
        size = min(image.size[0], label.size[0])
        W = int(image.size[1] / image.size[0] * size)
        image = image.resize((size, W), resample=Image.BILINEAR)
        label = label.resize((size, W), resample=Image.NEAREST)
        image, label = np.asarray(image), np.asarray(label)
        if len(label.shape) == 3:
            label = label[:, :, 0]
        if self.use_split != "test":
            image = self.distortion(image)
        image_t = (torch.from_numpy(image).permute(2, 0, 1) / 127.5) - 1
        label_t = torch.from_numpy(label).long()
        if self.use_split != "test":
            if torch.rand(1)[0] < 0.5:
                image_t = image_t.flip(2)
                label_t = label_t.flip(1)
        return image_t, label_t

    def __getitem__(self, idx):
        if idx == len(self.indice) - 1:
            print(f"=> Dataloader reset")
            self.indice = np.arange(0, len(self.indice))
            self.rng.shuffle(self.indice)
        sidx = self.indice[idx]
        image = pil_read(self.image_dir + "/" + self.imagefiles[sidx])
        label = pil_read(self.label_dir + "/" + self.labelfiles[sidx])
        res = self.transform(image, label)
        return res

    def __len__(self):
        return len(self.indice)


class CelebAMaskDataset(ImageSegmentationDataset):
    """Address the train/val/test split of CelebA."""

    def __init__(
        self, data_path, use_split="train", image_folder="image", label_folder="label"
    ):
        self.use_split = use_split
        self.root_dir = data_path
        self.image_dir = f"{data_path}/{image_folder}"
        self.label_dir = f"{data_path}/{label_folder}"
        self.distortion = PhotoMetricDistortion()

        self.imagefiles = [f"{i}.jpg" for i in range(30000)]
        self.labelfiles = [f"{i}.png" for i in range(30000)]
        self.parse_mapping(f"{data_path}/CelebA-HQ-to-CelebA-mapping.txt")
        self.rng = np.random.RandomState(1)
        self.indice = np.arange(0, len(self.imagefiles))
        if use_split == "train":
            self.indice = self.indice[self.split == 0]
            self.rng.shuffle(self.indice)
        elif use_split == "val":
            self.indice = self.indice[self.split == 1]
            self.rng.shuffle(self.indice)
        elif use_split == "test":
            self.indice = self.indice[self.split == 2]
            self.rng.shuffle(self.indice)

    def statistics(self):
        train_size = (self.split == 0).sum()
        val_size = (self.split == 1).sum()
        test_size = (self.split == 2).sum()
        total = float(len(self.imagefiles))
        print(f"=> Training size: {train_size} ({train_size / total})")
        print(f"=> Validation size: {val_size} ({val_size / total})")
        print(f"=> Test size: {test_size} ({test_size / total})")

    def parse_mapping(self, path):
        """From the CelebA evaluation split file:
        162770.jpg 0
        162771.jpg 1
        182637.jpg 1
        182638.jpg 2
        """
        with open(path, "r") as f:
            lines = f.readlines()
        self.split = np.zeros((len(self.imagefiles),), dtype="uint8")
        for i, l in enumerate(lines[1:]):
            if len(l) < 1:
                continue
            orig_idx = int(l.split(" ")[-1][:-5])
            if orig_idx <= 162770:
                pass  # training is 0
            elif orig_idx <= 182637:
                self.split[i] = 1
            else:
                self.split[i] = 2


class NoiseDataset(Dataset):
    def __init__(self, epoch_size=1024, latent_size=512, fixed=False):
        super().__init__()
        self.latent_size = latent_size
        self.epoch_size = epoch_size
        self.fixed = fixed
        if fixed:
            self.z = torch.randn(epoch_size, latent_size)

    def __getitem__(self, idx):
        if self.fixed:
            return self.z[idx]
        else:
            return torch.randn(self.latent_size)

    def __len__(self):
        return self.epoch_size


# from https://github.com/twuilliam/pascal-part-py
class PascalBase(object):
    def __init__(self, obj):
        self.mask = obj["mask"]
        self.props = self._get_region_props()

    def _get_region_props(self):
        """useful properties
        It includes: area, bbox, bbox_Area, centroid
        It can also extract: filled_image, image, intensity_image, local_centroid
        """
        return regionprops(self.mask)[0]


class PascalObject(PascalBase):
    def __init__(self, obj):
        super(PascalObject, self).__init__(obj)

        self.class_name = obj["class"][0]
        self.class_ind = obj["class_ind"][0, 0]

        self.n_parts = obj["parts"].shape[1]
        self.parts = []
        if self.n_parts > 0:
            for part in obj["parts"][0, :]:
                self.parts.append(PascalPart(part))

    def __repr__(self) -> str:
        s = f"{self.class_name}({self.class_ind}): "
        for p in self.parts:
            s += f"{p.part_name} "
        return s


class PascalPart(PascalBase):
    def __init__(self, obj):
        super(PascalPart, self).__init__(obj)
        self.part_name = obj["part_name"][0]


def get_class_names():
    classes = {
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "table",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tvmonitor",
    }
    return classes


def get_pimap():
    pimap = {}

    # [aeroplane]
    pimap[1] = {}
    pimap[1]["body"] = 1
    pimap[1]["stern"] = 2
    pimap[1]["lwing"] = 3  # left wing
    pimap[1]["rwing"] = 4  # right wing
    pimap[1]["tail"] = 5
    for ii in range(1, 10 + 1):
        pimap[1][("engine_%d" % ii)] = 10 + ii  # multiple engines
    for ii in range(1, 10 + 1):
        pimap[1][("wheel_%d" % ii)] = 20 + ii  # multiple wheels

    # [bicycle]
    pimap[2] = {}
    pimap[2]["fwheel"] = 1  # front wheel
    pimap[2]["bwheel"] = 2  # back wheel
    pimap[2]["saddle"] = 3
    pimap[2]["handlebar"] = 4  # handle bar
    pimap[2]["chainwheel"] = 5  # chain wheel
    for ii in range(1, 10 + 1):
        pimap[2][("headlight_%d" % ii)] = 10 + ii

    # [bird]
    pimap[3] = {}
    pimap[3]["head"] = 1
    pimap[3]["leye"] = 2  # left eye
    pimap[3]["reye"] = 3  # right eye
    pimap[3]["beak"] = 4
    pimap[3]["torso"] = 5
    pimap[3]["neck"] = 6
    pimap[3]["lwing"] = 7  # left wing
    pimap[3]["rwing"] = 8  # right wing
    pimap[3]["lleg"] = 9  # left leg
    pimap[3]["lfoot"] = 10  # left foot
    pimap[3]["rleg"] = 11  # right leg
    pimap[3]["rfoot"] = 12  # right foot
    pimap[3]["tail"] = 13

    # [boat]
    # only has silhouette mask

    # [bottle]
    pimap[5] = {}
    pimap[5]["cap"] = 1
    pimap[5]["body"] = 2

    # [bus]
    pimap[6] = {}
    pimap[6]["frontside"] = 1
    pimap[6]["leftside"] = 2
    pimap[6]["rightside"] = 3
    pimap[6]["backside"] = 4
    pimap[6]["roofside"] = 5
    pimap[6]["leftmirror"] = 6
    pimap[6]["rightmirror"] = 7
    pimap[6]["fliplate"] = 8  # front license plate
    pimap[6]["bliplate"] = 9  # back license plate
    for ii in range(1, 10 + 1):
        pimap[6][("door_%d" % ii)] = 10 + ii
    for ii in range(1, 10 + 1):
        pimap[6][("wheel_%d" % ii)] = 20 + ii
    for ii in range(1, 10 + 1):
        pimap[6][("headlight_%d" % ii)] = 30 + ii
    for ii in range(1, 20 + 1):
        pimap[6][("window_%d" % ii)] = 40 + ii

    # [car]
    # front, side, back, roof, license_plate, wheel, light, window
    pimap[7] = pimap[6].copy()
    pimap[7] = {}
    pimap[7]["frontside"] = 1  # front
    pimap[7]["leftside"] = 2  # side
    pimap[7]["rightside"] = 2  # side
    pimap[7]["backside"] = 3  # back
    pimap[7]["roofside"] = 4  # roof
    pimap[7]["leftmirror"] = 2  # side
    pimap[7]["rightmirror"] = 2
    pimap[7]["fliplate"] = 5  # license plate
    pimap[7]["bliplate"] = 5  # license plate
    for ii in range(1, 10 + 1):
        pimap[7][("door_%d" % ii)] = 2  # side
    for ii in range(1, 10 + 1):
        pimap[7][("wheel_%d" % ii)] = 6  # wheel
    for ii in range(1, 10 + 1):
        pimap[7][("headlight_%d" % ii)] = 7  # headlight
    for ii in range(1, 20 + 1):
        pimap[7][("window_%d" % ii)] = 8  # window

    # [cat]
    pimap[8] = {}
    pimap[8]["head"] = 1
    pimap[8]["leye"] = 2  # left eye
    pimap[8]["reye"] = 3  # right eye
    pimap[8]["lear"] = 4  # left ear
    pimap[8]["rear"] = 5  # right ear
    pimap[8]["nose"] = 6
    pimap[8]["torso"] = 7
    pimap[8]["neck"] = 8
    pimap[8]["lfleg"] = 9  # left front leg
    pimap[8]["lfpa"] = 10  # left front paw
    pimap[8]["rfleg"] = 11  # right front leg
    pimap[8]["rfpa"] = 12  # right front paw
    pimap[8]["lbleg"] = 13  # left back leg
    pimap[8]["lbpa"] = 14  # left back paw
    pimap[8]["rbleg"] = 15  # right back leg
    pimap[8]["rbpa"] = 16  # right back paw
    pimap[8]["tail"] = 17

    # [chair]
    # only has sihouette mask

    # [cow]
    pimap[10] = {}
    pimap[10]["head"] = 1
    pimap[10]["leye"] = 2  # left eye
    pimap[10]["reye"] = 3  # right eye
    pimap[10]["lear"] = 4  # left ear
    pimap[10]["rear"] = 5  # right ear
    pimap[10]["muzzle"] = 6
    pimap[10]["lhorn"] = 7  # left horn
    pimap[10]["rhorn"] = 8  # right horn
    pimap[10]["torso"] = 9
    pimap[10]["neck"] = 10
    pimap[10]["lfuleg"] = 11  # left front upper leg
    pimap[10]["lflleg"] = 12  # left front lower leg
    pimap[10]["rfuleg"] = 13  # right front upper leg
    pimap[10]["rflleg"] = 14  # right front lower leg
    pimap[10]["lbuleg"] = 15  # left back upper leg
    pimap[10]["lblleg"] = 16  # left back lower leg
    pimap[10]["rbuleg"] = 17  # right back upper leg
    pimap[10]["rblleg"] = 18  # right back lower leg
    pimap[10]["tail"] = 19

    # [table]
    # only has silhouette mask

    # [dog]
    pimap[12] = pimap[8].copy()  # dog has the same set of parts with cat,
    # except for the additional
    # muzzle
    pimap[12]["muzzle"] = 20

    # [horse]
    pimap[13] = pimap[10].copy()  # horse has the same set of parts with cow,
    # except it has hoof instead of horn
    del pimap[13]["lhorn"]
    del pimap[13]["rhorn"]
    pimap[13]["lfho"] = 30
    pimap[13]["rfho"] = 31
    pimap[13]["lbho"] = 32
    pimap[13]["rbho"] = 33

    # [motorbike]
    pimap[14] = {}
    pimap[14]["fwheel"] = 1
    pimap[14]["bwheel"] = 2
    pimap[14]["handlebar"] = 3
    pimap[14]["saddle"] = 4
    for ii in range(1, 10 + 1):
        pimap[14][("headlight_%d" % ii)] = 10 + ii

    # [person]
    pimap[15] = {}
    pimap[15]["head"] = 1
    pimap[15]["leye"] = 2  # left eye
    pimap[15]["reye"] = 3  # right eye
    pimap[15]["lear"] = 4  # left ear
    pimap[15]["rear"] = 5  # right ear
    pimap[15]["lebrow"] = 6  # left eyebrow
    pimap[15]["rebrow"] = 7  # right eyebrow
    pimap[15]["nose"] = 8
    pimap[15]["mouth"] = 9
    pimap[15]["hair"] = 10

    pimap[15]["torso"] = 11
    pimap[15]["neck"] = 12
    pimap[15]["llarm"] = 13  # left lower arm
    pimap[15]["luarm"] = 14  # left upper arm
    pimap[15]["lhand"] = 15  # left hand
    pimap[15]["rlarm"] = 16  # right lower arm
    pimap[15]["ruarm"] = 17  # right upper arm
    pimap[15]["rhand"] = 18  # right hand

    pimap[15]["llleg"] = 19  # left lower leg
    pimap[15]["luleg"] = 20  # left upper leg
    pimap[15]["lfoot"] = 21  # left foot
    pimap[15]["rlleg"] = 22  # right lower leg
    pimap[15]["ruleg"] = 23  # right upper leg
    pimap[15]["rfoot"] = 24  # right foot

    # [pottedplant]
    pimap[16] = {}
    pimap[16]["pot"] = 1
    pimap[16]["plant"] = 2

    # [sheep]
    pimap[17] = pimap[10].copy()  # sheep has the same set of parts with cow

    # [sofa]
    # only has sihouette mask

    # [train]
    pimap[19] = {}
    pimap[19]["head"] = 1
    pimap[19]["hfrontside"] = 2  # head front side
    pimap[19]["hleftside"] = 3  # head left side
    pimap[19]["hrightside"] = 4  # head right side
    pimap[19]["hbackside"] = 5  # head back side
    pimap[19]["hroofside"] = 6  # head roof side

    for ii in range(1, 10 + 1):
        pimap[19][("headlight_%d" % ii)] = 10 + ii

    for ii in range(1, 10 + 1):
        pimap[19][("coach_%d" % ii)] = 20 + ii

    for ii in range(1, 10 + 1):
        pimap[19][("cfrontside_%d" % ii)] = 30 + ii  # coach front side

    for ii in range(1, 10 + 1):
        pimap[19][("cleftside_%d" % ii)] = 40 + ii  # coach left side

    for ii in range(1, 10 + 1):
        pimap[19][("crightside_%d" % ii)] = 50 + ii  # coach right side

    for ii in range(1, 10 + 1):
        pimap[19][("cbackside_%d" % ii)] = 60 + ii  # coach back side

    for ii in range(1, 10 + 1):
        pimap[19][("croofside_%d" % ii)] = 70 + ii  # coach roof side

    # [tvmonitor]
    pimap[20] = {}
    pimap[20]["screen"] = 1

    return pimap


PIMAP = get_pimap()


def parse_pascal_part_ann(ann_path, class_name):
    """Parse pascal part annotation."""
    data = loadmat(ann_path)["anno"][0, 0]
    objects = [PascalObject(obj) for obj in data["objects"][0, :]]

    shape = objects[0].mask.shape
    part_mask = np.zeros(shape, dtype=np.uint8)
    # inst_mask = np.zeros(shape, dtype=np.uint8)
    # part_mask = np.zeros(shape, dtype=np.uint8)
    for i, obj in enumerate(objects):
        # mask = obj.mask
        # inst_mask[mask > 0] = i + 1
        # cls_mask[mask > 0] = obj.class_ind
        if obj.class_name != "car":
            continue
        if obj.n_parts > 0:
            for p in obj.parts:
                part_name = p.part_name
                pid = PIMAP[obj.class_ind][part_name]
                part_mask[p.mask > 0] = pid
    return part_mask


class NoiseDataModule(pl.LightningDataModule):
    """Noise Data Module."""

    def __init__(self, train_size=1024, val_size=1024, latent_size=512, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.train_ds = NoiseDataset(train_size, latent_size)
        self.val_ds = NoiseDataset(val_size, latent_size, fixed=True)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)
