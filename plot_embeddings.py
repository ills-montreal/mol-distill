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
                if file.endswith(".npy") and not "shape" in file:
                    print(file)
                    embeddings[file] = np.load(os.path.join(path, dir, file))
    return embeddings



if __name__ == "__main__":
    embeddings = get_all_embeddings(PATH, MODELS)
    import pandas as pd
    from sklearn.decomposition import PCA
    df = pd.DataFrame()
    pca = PCA(2)
    pca.fit(
        np.concatenate([embeddings[k] for k in embeddings.keys()])
    )

    import json
    with open(os.path.join(PATH, "smiles.json")) as f:
        smiles = json.load(f)


    for key in embeddings:
        embs = pca.transform(embeddings[key])
        print(f"{key} -- Mean L2 --\t {np.mean(embs, axis=0)}")
        tmp_df = pd.DataFrame(embs, columns=['x','y'])
        tmp_df['key'] = key
        df = pd.concat([df, tmp_df])
    df['smiles'] = smiles
    df = df.sample(10000)
    import datamol as dm
    df_mols_prop = dm.descriptors.batch_compute_many_descriptors([dm.to_mol(s) for s in df['smiles']])
    sns.scatterplot(df, x='x',y='y', hue='key', legend=False, alpha=0.5)
    plt.show()

    print("Done")
    df = pd.concat([df.reset_index(), df_mols_prop], axis=1)
    for key in df_mols_prop.columns:
        sns.histplot(df, x=key, hue="key", legend=False, alpha=0.5)
        plt.show()





