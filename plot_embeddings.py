import os

import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


PATH = "../data/MOSES"
MODELS = [
        "GraphMVP",
        "GROVER",
        "GraphLog",
        "GraphCL",
        "InfoGraph",
        "ChemBertMLM-10M",
        "ChemBertMTR-10M",
        "ChemGPT-4.7M",
        "DenoisingPretrainingPQCMv4",
        "FRAD_QM9",
        "MolR_gat",
        "ThreeDInfomax",
    ]

def get_all_embeddings(path, models):
    """
    Returns a dictionary with all the embeddings for the MOSES dataset
    """
    embeddings = {}
    for dir in tqdm(models):
        if os.path.isdir(os.path.join(path, dir)):
            for file in os.listdir(os.path.join(path, dir)):
                if file.endswith("_0.npy"):
                    embeddings[file] = np.load(os.path.join(path, dir, file))
    return embeddings



if __name__ == "__main__":
    embeddings = get_all_embeddings(PATH, MODELS)
    print(embeddings.keys())
    N = len(embeddings.keys())

    fig,axes = plt.subplots(N//4 +1, 4, figsize=(10, 10))
    axes = axes.flatten()
    for i, key in enumerate(embeddings.keys()):
        value = embeddings[key]
        sns.scatterplot(x=value[:, 0], y=value[:, 1], ax=axes[i], alpha=0.1)
        axes[i].set_title(key)
    plt.tight_layout()
    plt.show()



