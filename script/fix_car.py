import glob, tqdm
from lib.misc import imread, imwrite

data_dir = "data/generated/stylegan2_car_s1113/image"
imfiles = glob.glob(f"{data_dir}/*.jpg")
imfiles.sort()
for fp in tqdm.tqdm(imfiles):
    img = imread(fp)[64:-64]
    imwrite(fp, img)
