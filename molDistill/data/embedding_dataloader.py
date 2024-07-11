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
    def __init__(self, data_dir, dataset, model_names, model_files, idx = None):
        with open(os.path.join(data_dir, dataset, "smiles.json"), "r") as f:
            self.smiles = np.array(json.load(f))
            if not idx is None:
                self.smiles = self.smiles[idx]

        self.data = []
        self.embs_dim = []
        for model_name, model_file in zip(model_names, model_files):
            embs = np.load(os.path.join(data_dir, dataset, model_file))
            if not idx is None:
                embs = embs[idx]
            embs = (embs - embs.mean(axis=0)) / (embs.std(axis=0) + 1e-8)
            embs = torch.tensor(embs, dtype=torch.float)

            self.data.append(
                [
                    Embedding(embedding, smiles, model_name)
                    for embedding, smiles in zip(embs, self.smiles)
                ]
            )
            self.embs_dim.append(embs.shape[1])

    def update_idx(self, idx):
        self.smiles = self.smiles[idx]
        for i in range(len(self.data)):
            self.data[i] = [self.data[i][j] for j in idx]

    def __getitem__(self, index):
        return [embeddings[index] for embeddings in self.data]

    def __len__(self):
        return len(self.smiles)


def collate_fn(batch):
    n_embs = len(batch[0])
    embeddings = [[] for _ in range(n_embs)]
    smiles = [[] for _ in range(n_embs)]

    for embeddings_models in batch:
        for i in range(n_embs):
            embeddings[i].append(embeddings_models[i].embedding)
            smiles[i].append(embeddings_models[i].smiles)
    embeddings = [torch.stack(e) for e in embeddings]
    assert all(smiles[0] == s for s in smiles)
    return embeddings, smiles[0]


def get_embedding_loader(args):

    model_files = [f"{model_name}.npy" for model_name in args.embedders_to_simulate]
    dataset_train = EmbeddingDataset(
        args.data_dir, args.dataset, args.embedders_to_simulate, model_files
    )
    n_data = len(dataset_train)
    idx_train = torch.randperm(n_data)
    idx_valid = idx_train[: int(n_data * args.valid_prop)].tolist()
    idx_train = idx_train[int(n_data * args.valid_prop) :].tolist()

    dataset_train.update_idx(idx_train)
    dataset_valid = EmbeddingDataset(
        args.data_dir, args.dataset, args.embedders_to_simulate, model_files, idx_valid
    )

    emb_loader = data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    emb_loader_valid = data.DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return emb_loader, emb_loader_valid, dataset_train.embs_dim, idx_train, idx_valid
