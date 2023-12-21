import sys
sys.path.insert(0, "../..")
sys.path.insert(0, "..")
from models.helper import build_generator

G = build_generator("stylegan2_ffhq")