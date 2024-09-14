import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.utils.data as data


@dataclass(frozen=True)
class Embedding:
    embedding: torch.Tensor
    smiles: str
    model_name: str = field(default=None)


class EmbeddingDataset(data.Dataset):
    def __init__(
        self, data_dir, dataset, model_names, model_files, idx=None, initial_pointer=0
    ):
        with open(os.path.join(data_dir, dataset, "smiles.json"), "r") as f:
            self.all_smiles = json.loads(f.read())
        self.all_smiles = np.array(self.all_smiles)
        self.smiles = self.all_smiles[initial_pointer:]
        if not idx is None:
            self.smiles = self.smiles[idx]

        self.data = []
        self.embs_dim = []

        self.data_model = {}

        for model_name, model_file in zip(model_names, model_files):
            embs = np.load(os.path.join(data_dir, dataset, model_file), mmap_mode="r")

            if not os.path.exists(
                os.path.join(data_dir, dataset, model_file).replace(
                    ".npy", "_shape.npy"
                )
            ):
                np.save(
                    os.path.join(data_dir, dataset, model_file).replace(
                        ".npy", "_shape.npy"
                    ),
                    np.array(embs.shape),
                )

            if not idx is None:
                embs = embs[idx]

            self.data_model[model_name] = embs
            n_data = embs.shape[0]

            # embs = torch.tensor(embs, dtype=torch.float)
            # self.data.append(
            #    [
            #        Embedding(embedding, smiles, model_name)
            #        for embedding, smiles in zip(embs, self.smiles)
            #    ]
            # )
            self.embs_dim.append(embs.shape[1])

        self.smiles = self.smiles[:n_data]

    def update_idx(self, idx):
        self.smiles = self.smiles[idx]
        for model_name in self.data_model.keys():
            self.data_model[model_name] = self.data_model[model_name][idx]

    def __getitem__(self, index):
        return [
            Embedding(
                torch.tensor(self.data_model[model_name][index], dtype=torch.float),
                self.smiles[index],
                model_name,
            )
            for model_name in self.data_model.keys()
        ]

    def __len__(self):
        return len(self.smiles)
