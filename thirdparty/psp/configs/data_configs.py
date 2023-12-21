from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    "ffhq_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["ffhq"],
        "train_target_root": dataset_paths["ffhq"],
        "test_source_root": dataset_paths["celeba_test"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "ffhq_frontalize": {
        "transforms": transforms_config.FrontalizationTransforms,
        "train_source_root": dataset_paths["ffhq"],
        "train_target_root": dataset_paths["ffhq"],
        "test_source_root": dataset_paths["celeba_test"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "celebs_sketch_to_face": {
        "transforms": transforms_config.SketchToImageTransforms,
        "train_source_root": dataset_paths["celeba_train_sketch"],
        "train_target_root": dataset_paths["celeba_train"],
        "test_source_root": dataset_paths["celeba_test_sketch"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "celebs_seg_to_face": {
        "transforms": transforms_config.SegToImageTransforms,
        "train_source_root": dataset_paths["celeba_train_segmentation"],
        "train_target_root": dataset_paths["celeba_train"],
        "test_source_root": dataset_paths["celeba_test_segmentation"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "celebs_super_resolution": {
        "transforms": transforms_config.SuperResTransforms,
        "train_source_root": dataset_paths["celeba_train"],
        "train_target_root": dataset_paths["celeba_train"],
        "test_source_root": dataset_paths["celeba_test"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "stylegan2_ffhq_klish": {
        "transforms": transforms_config.SegToImageTransforms,
        "train_source_root": dataset_paths["stylegan2_ffhq_klish"] + "/label_klish_c26",
        "train_target_root": dataset_paths["stylegan2_ffhq_klish"] + "/image",
        "test_source_root": dataset_paths["stylegan2_ffhq_klish"] + "/label_klish_c26",
        "test_target_root": dataset_paths["stylegan2_ffhq_klish"] + "/image",
    },
    "stylegan2_car_klish": {
        "transforms": transforms_config.SegToCarTransforms,
        "train_source_root": dataset_paths["stylegan2_car_klish"] + "/label_klish_c12",
        "train_target_root": dataset_paths["stylegan2_car_klish"] + "/image",
        "test_source_root": dataset_paths["stylegan2_car_klish"] + "/label_klish_c12",
        "test_target_root": dataset_paths["stylegan2_car_klish"] + "/image",
    },
    "ada_cat_klish": {
        "transforms": transforms_config.SegToImageTransforms,
        "train_source_root": dataset_paths["ada_cat_klish"] + "/label_klish_c7",
        "train_target_root": dataset_paths["ada_cat_klish"] + "/image",
        "test_source_root": dataset_paths["ada_cat_klish"] + "/label_klish_c7",
        "test_target_root": dataset_paths["ada_cat_klish"] + "/image",
    },
    "ada_dog_klish": {
        "transforms": transforms_config.SegToImageTransforms,
        "train_source_root": dataset_paths["ada_dog_klish"] + "/label_klish_c9",
        "train_target_root": dataset_paths["ada_dog_klish"] + "/image",
        "test_source_root": dataset_paths["ada_dog_klish"] + "/label_klish_c9",
        "test_target_root": dataset_paths["ada_dog_klish"] + "/image",
    },
    "ada_metface_klish": {
        "transforms": transforms_config.SegToImageTransforms,
        "train_source_root": dataset_paths["ada_metface_klish"] + "/label_klish_c18",
        "train_target_root": dataset_paths["ada_metface_klish"] + "/image",
        "test_source_root": dataset_paths["ada_metface_klish"] + "/label_klish_c18",
        "test_target_root": dataset_paths["ada_metface_klish"] + "/image",
    },
    "ada_wild_klish": {
        "transforms": transforms_config.SegToImageTransforms,
        "train_source_root": dataset_paths["ada_wild_klish"] + "/label_klish_c22",
        "train_target_root": dataset_paths["ada_wild_klish"] + "/image",
        "test_source_root": dataset_paths["ada_wild_klish"] + "/label_klish_c22",
        "test_target_root": dataset_paths["ada_wild_klish"] + "/image",
    },
    "stylegan2_ffhq_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["stylegan2_ffhq_klish"] + "/image",
        "train_target_root": dataset_paths["stylegan2_ffhq_klish"] + "/image",
        "test_source_root": dataset_paths["stylegan2_ffhq_klish"] + "/image",
        "test_target_root": dataset_paths["stylegan2_ffhq_klish"] + "/image",
    },
    "stylegan2_car_encode": {
        "transforms": transforms_config.EncodeCarTransforms,
        "train_source_root": dataset_paths["stylegan2_car_klish"] + "/image",
        "train_target_root": dataset_paths["stylegan2_car_klish"] + "/image",
        "test_source_root": dataset_paths["stylegan2_car_klish"] + "/image",
        "test_target_root": dataset_paths["stylegan2_car_klish"] + "/image",
    },
    "ada_cat_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["ada_cat_klish"] + "/image",
        "train_target_root": dataset_paths["ada_cat_klish"] + "/image",
        "test_source_root": dataset_paths["ada_cat_klish"] + "/image",
        "test_target_root": dataset_paths["ada_cat_klish"] + "/image",
    },
    "ada_dog_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["ada_dog_klish"] + "/image",
        "train_target_root": dataset_paths["ada_dog_klish"] + "/image",
        "test_source_root": dataset_paths["ada_dog_klish"] + "/image",
        "test_target_root": dataset_paths["ada_dog_klish"] + "/image",
    },
    "ada_metface_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["ada_metface_klish"] + "/image",
        "train_target_root": dataset_paths["ada_metface_klish"] + "/image",
        "test_source_root": dataset_paths["ada_metface_klish"] + "/image",
        "test_target_root": dataset_paths["ada_metface_klish"] + "/image",
    },
    "ada_wild_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["ada_wild_klish"] + "/image",
        "train_target_root": dataset_paths["ada_wild_klish"] + "/image",
        "test_source_root": dataset_paths["ada_wild_klish"] + "/image",
        "test_target_root": dataset_paths["ada_wild_klish"] + "/image",
    },
}