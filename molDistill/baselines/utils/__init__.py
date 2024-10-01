import json
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    )
)

from typing import List, Optional

import datamol as dm
import numpy as np
import torch

from molDistill.baselines.models.model_paths import get_model_path
from molDistill.baselines.utils.model_factory import ModelFactory


class MolecularFeatureExtractor:
    def __init__(
        self,
        device: str = "cpu",
        dataset: str = "ClinTox",
        normalize: bool = True,
        data_dir: str = "../data",
        path_ckpt: str = "backbone_pretrained_models",
        i_file=None,
    ):
        self.graph_input = None
        self.device = device

        self.dataset = dataset
        self.normalize = normalize
        self.data_dir = os.path.join(data_dir, dataset)
        self.model_path = get_model_path(path_ckpt)
        self.path_ckpt = path_ckpt
        self.i_file = i_file

    @property
    def mols_smiles(self):
        if self.i_file is None:
            with open(os.path.join(self.data_dir, "smiles.json"), "r") as f:
                smiles = json.load(f)
            mols = dm.read_sdf(os.path.join(self.data_dir, "preprocessed.sdf"))
        else:
            df = dm.read_sdf(
                os.path.join(
                    self.data_dir,
                    "preprocessed",
                    f"preprocessed_{self.i_file}.sdf",
                ),
                as_df=True,
                mol_column="mols",
            )

            mols = df["mols"].tolist()
            print(df.head())
            smiles = df["smiles"].iloc[:, 1].tolist()
        return mols, smiles

    def get_features(
        self,
        smiles: List[str],
        name: str,  # if custom : "custom:ckpt_path"
        mols: Optional[List[dm.Mol]] = None,
        i_file=None,
    ):
        device = self.device
        dataset = self.dataset
        normalize = self.normalize

        if name in self.model_path:
            path = self.model_path[name]
        elif "custom" in name:
            path = name.replace("custom:", "")
        else:
            path = self.path_ckpt
        if i_file is None:
            embedding_path = os.path.join(self.data_dir, f"{name}.npy")
        else:
            embedding_path = os.path.join(self.data_dir, name, f"{name}_{i_file}.npy")

        if os.path.exists(embedding_path):
            molecular_embedding = torch.tensor(np.load(embedding_path), device=device)
        else:
            if smiles is None:
                mols, smiles = self.mols_smiles
            molecular_embedding = ModelFactory(name)(
                smiles,
                mols=mols,
                path=path,
                transformer_name=name,
                device=device,
                dataset=dataset,
                DATA_PATH=self.data_dir,
                i_file=i_file,
            )
            if not name.startswith("custom:"):
                os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
                np.save(embedding_path, molecular_embedding.cpu().numpy())
                np.save(
                    embedding_path.replace(".npy", "_shape.npy"),
                    np.array(molecular_embedding.cpu().numpy().shape),
                )

        if normalize and i_file is None:
            molecular_embedding = (
                molecular_embedding - molecular_embedding.mean(dim=0)
            ) / (molecular_embedding.std(dim=0) + 1e-8)

        return molecular_embedding


def compute_embeddings(args, i_file=None):
    mfe = MolecularFeatureExtractor(
        dataset=args.dataset,
        data_dir=args.data_dir,
        device=device,
        i_file=i_file,
    )

    embs = {}
    np.random.shuffle(args.model_names)
    for name in tqdm(args.model_names, desc="Computing embeddings :"):
        print(f"Computing embeddings for {name}")
        embs[name] = mfe.get_features(None, name, mols=None, i_file=i_file)

    print("Done!")


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="MOSES*-1")
    parser.add_argument(
        "--model-names",
        nargs="+",
        type=str,
        default=[
            "ContextPred",
            "GPT-GNN",
            "GraphMVP",
            "GROVER",
            "AttributeMask",
            "GraphLog",
            "GraphCL",
            "InfoGraph",
            "MolBert",
            "ChemBertMLM-5M",
            "ChemBertMLM-10M",
            "ChemBertMLM-77M",
            "ChemBertMTR-5M",
            "ChemBertMTR-10M",
            "ChemBertMTR-77M",
            "ChemGPT-1.2B",
            "ChemGPT-19M",
            "ChemGPT-4.7M",
            "DenoisingPretrainingPQCMv4",
            "FRAD_QM9",
            "MolR_gat",
            "MolR_gcn",
            "MolR_tag",
            # "MoleOOD_OGB_GIN",
            # "MoleOOD_OGB_GCN",
            # "MoleOOD_OGB_SAGE",
            "ThreeDInfomax",
        ],
    )

    args = parser.parse_args()

    i_file = None
    if "*" in args.dataset:
        args.dataset, i_file = args.dataset.split("*")
        i_file = int(i_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if i_file is None or i_file >= 0:
        compute_embeddings(
            args,
            i_file=i_file,
        )
    else:
        data_files = os.listdir(
            os.path.join(args.data_dir, args.dataset, "preprocessed")
        )
        i_files = [int(f.split("_")[-1].replace(".sdf", "")) for f in data_files]
        np.random.shuffle(i_files)
        p_bar = tqdm(total=len(i_files), desc="Computing embeddings for all files")
        for i_file in i_files:
            p_bar.desc = f"Computing embeddings for file {i_file} :"
            p_bar.update(1)
            compute_embeddings(args, i_file=i_file)
