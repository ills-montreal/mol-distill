import json
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.utils.data as tdata
from torch_geometric.loader import DataLoader


@dataclass(frozen=True)
class Embedding:
    embedding: torch.Tensor
    smiles: str
    model_name: str = field(default=None)


class EmbeddingDataset(tdata.Dataset):
    def __init__(self, embeddings_models, smiles_models, model_names):
        self.embeddings = []
        for embeddings, smiles, model_name in zip(embeddings_models, smiles_models, model_names):
            self.embeddings.append(
                [
                    Embedding(embedding, smiles, model_name)
                    for embedding, smiles in zip(embeddings, smiles)
                ]
            )

    def __getitem__(self, index):
        return [embeddings[index] for embeddings in self.embeddings]

    def __len__(self):
        return len(self.embeddings[0])


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
    all_embedders = []
    embs_dim = []
    model_names = []
    with open(os.path.join(args.data_dir, args.dataset, "smiles.json"), "r") as f:
        smiles = np.array(json.load(f))
    for embedder in args.embedders_to_simulate:
        embs = np.load(os.path.join(args.data_dir, args.dataset, f"{embedder}.npy"))
        embs = (embs - embs.mean(axis=0)) / (embs.std(axis=0) + 1e-8)
        embs = torch.tensor(embs, dtype=torch.float)

        all_embedders.append(embs)
        embs_dim.append(embs.shape[1])
        model_names.append(embedder)

    idx_train = torch.randperm(all_embedders[0].size(0))
    idx_valid = idx_train[: int(all_embedders[0].size(0) * args.valid_prop)].tolist()
    idx_train = idx_train[int(all_embedders[0].size(0) * args.valid_prop) :].tolist()

    dataset_train = EmbeddingDataset(
        [e[idx_train] for e in all_embedders],
        [smiles[idx_train] for _ in all_embedders],
        model_names
    )
    dataset_valid = EmbeddingDataset(
        [e[idx_valid] for e in all_embedders],
        [smiles[idx_valid] for _ in all_embedders],
        model_names
    )

    emb_loader = tdata.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    emb_loader_valid = tdata.DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return emb_loader, emb_loader_valid, embs_dim, idx_train, idx_valid
