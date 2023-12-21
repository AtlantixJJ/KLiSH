# Training GauGAN on Synthetic Dataset

```bash
export DATA_DIR=../../data/generated/

python train.py --name stylegan2_ffhq_klish_512 --load_size 512 --crop_size 512 --dataset_mode custom --label_dir $DATA_DIR/stylegan2_ffhq_s1113/label_klish_c26  --image_dir $DATA_DIR/stylegan2_ffhq_s1113/image --label_nc 26 --no_instance --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7 --use_vae

python train.py --name stylegan2_car_klish_512 --preprocess_mode fixed --aspect_ratio 1.3333 --crop_size 512 --dataset_mode custom --label_dir $DATA_DIR/stylegan2_car_s1113/label_klish_c12  --image_dir $DATA_DIR/stylegan2_car_s1113/image --label_nc 12 --no_instance --batchSize 4 --use_vae --gpu_ids 0,1,2,3,4,5,6,7 

python train.py --name ada_cat_klish_512 --load_size 512 --crop_size 512 --dataset_mode custom --label_dir $DATA_DIR/ada_cat_s1113/label_klish_c7  --image_dir $DATA_DIR/ada_cat_s1113/image --label_nc 7 --no_instance --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7 --use_vae

python train.py --name ada_dog_klish_512 --load_size 512 --crop_size 512 --dataset_mode custom --label_dir $DATA_DIR/ada_dog_s1113/label_klish_c9  --image_dir $DATA_DIR/ada_dog_s1113/image --label_nc 9 --no_instance --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7 --use_vae


python train.py --name stylegan2_ffhq_klish_512 --load_size 512 --crop_size 512 --dataset_mode custom --label_dir $DATA_DIR/stylegan2_ffhq_s1113/label_klish_c26  --image_dir $DATA_DIR/stylegan2_ffhq_s1113/image --label_nc 26 --no_instance --batchSize 4 --gpu_ids 6,7,8,9 --use_vae && python train.py --name stylegan2_car_klish_512 --preprocess_mode fixed --aspect_ratio 1.3333 --crop_size 512 --dataset_mode custom --label_dir $DATA_DIR/stylegan2_car_s1113/label_klish_c12  --image_dir $DATA_DIR/stylegan2_car_s1113/image --label_nc 12 --no_instance --batchSize 8 --use_vae --gpu_ids 6,7,8,9 && python train.py --name ada_cat_klish_512 --load_size 512 --crop_size 512 --dataset_mode custom --label_dir $DATA_DIR/ada_cat_s1113/label_klish_c7  --image_dir $DATA_DIR/ada_cat_s1113/image --label_nc 7 --no_instance --batchSize 4 --gpu_ids 6,7,8,9 --use_vae && python train.py --name ada_dog_klish_512 --load_size 512 --crop_size 512 --dataset_mode custom --label_dir $DATA_DIR/ada_dog_s1113/label_klish_c9  --image_dir $DATA_DIR/ada_dog_s1113/image --label_nc 9 --no_instance --batchSize 8 --gpu_ids 6,7,8,9 --use_vae


python train.py --name ada_wild_klish_512 --load_size 512 --crop_size 512 --dataset_mode custom --label_dir $DATA_DIR/ada_wild_s1113/label_klish_c22  --image_dir $DATA_DIR/ada_wild_s1113/image --label_nc 24 --no_instance --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7 --use_vae && python train.py --name ada_metface_klish_512 --load_size 512 --crop_size 512 --dataset_mode custom --label_dir $DATA_DIR/ada_metface_s1113/label_klish_c18  --image_dir $DATA_DIR/ada_metface_s1113/image --label_nc 18 --no_instance --batchSize 8 --gpu_ids 0,1,2,3,4,5,7,8 --use_vae
```