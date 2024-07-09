import json
import os
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
    ):
        self.graph_input = None
        self.device = device

        self.dataset = dataset
        self.normalize = normalize
        self.data_dir = os.path.join(data_dir, dataset)
        self.model_path = get_model_path(path_ckpt)
        self.path_ckpt = path_ckpt

    def get_features(
        self,
        smiles: List[str],
        name: str,  # if custom : "custom:ckpt_path"
        mols: Optional[List[dm.Mol]] = None,
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

        if os.path.exists(f"{self.data_dir}/{name}.npy"):
            molecular_embedding = torch.tensor(
                np.load(f"{self.data_dir}/{name}.npy"), device=device
            )
        else:
            molecular_embedding = ModelFactory(name)(
                smiles,
                mols=mols,
                path=path,
                transformer_name=name,
                device=device,
                dataset=dataset,
                DATA_PATH=self.data_dir,
            )
            if not name.startswith("custom:"):
                np.save(f"{self.data_dir}/{name}.npy", molecular_embedding.cpu().numpy())

        if normalize:
            molecular_embedding = (
                molecular_embedding - molecular_embedding.mean(dim=0)
            ) / (molecular_embedding.std(dim=0) + 1e-8)

        return molecular_embedding


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="hERG")
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
            # "ChemGPT-1.2B",
            # "ChemGPT-19M",
            "ChemGPT-4.7M",
            "DenoisingPretrainingPQCMv4",
            "FRAD_QM9",
            "MolR_gat",
            "MolR_gcn",
            "MolR_tag",
            "MoleOOD_OGB_GIN",
            "MoleOOD_OGB_GCN",
            # "MoleOOD_OGB_SAGE",
            "ThreeDInfomax",
        ],
    )

    args = parser.parse_args()

    mfe = MolecularFeatureExtractor()
    with open(os.path.join(args.data_dir, args.dataset, "smiles.json"), "r") as f:
        smiles = json.load(f)

    mols = dm.read_sdf(os.path.join(args.data_dir, args.dataset, "preprocessed.sdf"))

    embs = {}
    for name in tqdm(args.model_names):
        embs[name] = mfe.get_features(smiles, name, mols=mols)

    print("Done!")
