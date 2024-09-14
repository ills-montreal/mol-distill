import numpy as np
import os

DIR = "/export/gauss/vision/Pformont/emir/molecule/data_new/ZINC"

def compute_shapes(path):
    if path.endswith(".npy") and not path.endswith("_shape.npy"):
        embs = np.load(path)
        shape = embs.shape
        np.save(path.replace(".npy", "_shape.npy"), shape)




for file in os.listdir(DIR):
    compute_shapes(os.path.join(DIR, file))