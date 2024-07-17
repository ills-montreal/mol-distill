import os
import json

import torch
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.data import Batch

from molDistill.data.data_encoding import DistillGraphDataset
from molDistill.data.embedding_dataloader import EmbeddingDataset


class DistillDataset(IterableDataset):
    def __init__(self, data_dir, dataset, model_names, idx=None, shuffle=False):
        self.data_dir = data_dir
        self.dataset = dataset
        self.model_names = model_names
        self.graph_dataset = DistillGraphDataset(os.path.join(data_dir, dataset))
        if idx is not None:
            self.graph_dataset = self.graph_dataset[idx]
        self.graph_dataset_idx = 0
        self.embedder_dataset = None

        model_files = []
        for m in model_names:
            if os.path.exists(os.path.join(data_dir, dataset, m + ".npy")):
                model_files.append(m + ".npy")
            else:
                model_files.append(m)

        if model_files[0].endswith(".npy"):
            self.embedder_files = [model_files]
        else:
            self.embedder_files = []
            for i in range(100):
                if not os.path.exists(
                    os.path.join(
                        data_dir, dataset, model_files[0], f"{model_files[0]}_{i}.npy"
                    )
                ):
                    break
                else:
                    self.embedder_files.append([])
                    for model_file in model_files:
                        self.embedder_files[-1].append(
                            os.path.join(model_file, f"{model_file}_{i}.npy")
                        )
        self.idx = idx

        self.shuffle = shuffle
        self.file_conditioned_idx = None
        self.len_prev_files = 0
        self.current_file_id = -1

    def update_idx(self, idx):
        assert self.embedder_dataset is None
        self.idx = idx
        self.graph_dataset = self.graph_dataset[idx]

    def load_next_file(self):
        self.current_file_id += 1
        if self.current_file_id < len(self.embedder_files):
            self.embedder_dataset = EmbeddingDataset(
                self.data_dir,
                self.dataset,
                self.model_names,
                self.embedder_files[self.current_file_id],
                initial_pointer=self.len_prev_files,
            )
            n_data = len(self.embedder_dataset)
            if not self.idx is None:
                self.file_conditioned_idx = [
                    idx - self.len_prev_files
                    for idx in self.idx
                    if idx >= self.len_prev_files and idx < self.len_prev_files + n_data
                ]
                self.embedder_dataset.update_idx(self.file_conditioned_idx)

            self.len_prev_files += n_data

        else:
            self.embedder_dataset = None
            self.current_file_id = -1
            self.len_prev_files = 0
            self.graph_dataset_idx = 0

    def __iter__(self):
        while not self.embedder_dataset is None:
            for i in range(len(self.embedder_dataset)):
                graph = self.graph_dataset[self.graph_dataset_idx]
                embs = self.embedder_dataset[i]
                assert graph.smiles == embs[0].smiles
                yield graph, embs
                self.graph_dataset_idx += 1
            self.load_next_file()
        self.load_next_file()

    def __len__(self):
        return len(self.graph_dataset)


def collate_fn(batch_tot):
    graph_batch, emb_batch = list(zip(*batch_tot))
    n_embs = len(emb_batch[0])
    embeddings = [[] for _ in range(n_embs)]
    smiles = [[] for _ in range(n_embs)]

    for embeddings_models in emb_batch:
        for i in range(n_embs):
            embeddings[i].append(embeddings_models[i].embedding)
            smiles[i].append(embeddings_models[i].smiles)
    embeddings = [torch.stack(e) for e in embeddings]
    assert all(smiles[0] == s for s in smiles)
    return Batch.from_data_list(graph_batch), embeddings


def get_embedding_loader(args):
    model_files = [f"{model_name}.npy" for model_name in args.embedders_to_simulate]
    dataset_train = DistillDataset(
        args.data_dir,
        args.dataset,
        args.embedders_to_simulate,
    )
    n_data = len(dataset_train)
    idx_train = torch.randperm(n_data)
    idx_valid = idx_train[: int(n_data * args.valid_prop)].tolist()
    idx_valid.sort()
    idx_train = idx_train[int(n_data * args.valid_prop) :].tolist()
    idx_train.sort()

    dataset_train.update_idx(idx_train)
    dataset_valid = DistillDataset(
        args.data_dir, args.dataset, args.embedders_to_simulate, idx_valid
    )

    dataset_train.load_next_file()
    dataset_valid.load_next_file()

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=1,
        drop_last=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        num_workers=1,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, dataset_valid.embedder_dataset.embs_dim


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="MOSES")

    args = parser.parse_args()

    data_dir = args.data_dir
    data = args.dataset
    data_path = f"{data_dir}/{data}"

    dataset = DistillDataset(data_dir, data, ["GraphMVP"])
    dataset.load_next_file()

    for graph, embs in tqdm(dataset.__iter__()):
        assert graph.smiles == embs[0].smiles

    dataset = DistillDataset(data_dir, data, ["GraphMVP"])
    idx = np.random.choice(len(dataset), 500)
    idx.sort()
    dataset.update_idx(idx)
    dataset.load_next_file()

    for graph, embs in tqdm(dataset.__iter__()):
        assert graph.smiles == embs[0].smiles

