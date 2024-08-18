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
    def __init__(self, data_dir, dataset, model_names, model_files, idx = None, initial_pointer = 0):
        with open(os.path.join(data_dir, dataset, "smiles.json"), "r") as f:
            self.smiles = np.array(json.load(f))[initial_pointer:]
            if not idx is None:
                self.smiles = self.smiles[idx]

        self.data = []
        self.embs_dim = []
        for model_name, model_file in zip(model_names, model_files):
            embs = np.load(os.path.join(data_dir, dataset, model_file))

            if not os.path.exists(os.path.join(data_dir, dataset, model_file).replace(".npy", "_shape.npy")):
                np.save(os.path.join(data_dir, dataset, model_file).replace(".npy", "_shape.npy"), np.array(embs.shape))

            if not idx is None:
                embs = embs[idx]
            embs = torch.tensor(embs, dtype=torch.float)
            self.data.append(
                [
                    Embedding(embedding, smiles, model_name)
                    for embedding, smiles in zip(embs, self.smiles)
                ]
            )
            self.embs_dim.append(embs.shape[1])
        self.smiles = self.smiles[:len(self.data[0])]

    def update_idx(self, idx):
        self.smiles = self.smiles[idx]
        for i in range(len(self.data)):
            self.data[i] = [self.data[i][j] for j in idx]

    def __getitem__(self, index):
        return [embeddings[index] for embeddings in self.data]

    def __len__(self):
        return len(self.smiles)




