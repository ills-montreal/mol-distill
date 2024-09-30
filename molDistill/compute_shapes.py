import numpy as np
import os


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="ClinTox")
args = parser.parse_args()

DIR = "/export/gauss/vision//emir/molecule/data_new/" + args.data

def compute_shapes(path):
    if path.endswith(".npy") and not path.endswith("_shape.npy"):
        embs = np.load(path)
        shape = embs.shape
        np.save(path.replace(".npy", "_shape.npy"), shape)




for file in os.listdir(DIR):
    compute_shapes(os.path.join(DIR, file))