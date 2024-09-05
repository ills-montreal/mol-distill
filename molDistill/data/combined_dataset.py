import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

import json
from functools import partial
import numpy as np

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

            shape = np.load(
                os.path.join(
                    self.data_dir, self.dataset, self.embedder_files[self.current_file_id][0].replace(".npy", "_shape.npy")
                )
            )
            n_data = shape[0]
            if not self.idx is None:
                self.file_conditioned_idx = [
                    idx - self.len_prev_files
                    for idx in self.idx
                    if idx >= self.len_prev_files and idx < self.len_prev_files + n_data
                ]

            if self.file_conditioned_idx == []:
                self.len_prev_files += n_data
                self.load_next_file()
            else:
                self.embedder_dataset = EmbeddingDataset(
                    self.data_dir,
                    self.dataset,
                    self.model_names,
                    self.embedder_files[self.current_file_id],
                    initial_pointer=self.len_prev_files,
                )
                if self.idx is not None:
                    self.embedder_dataset.update_idx(self.file_conditioned_idx)

                self.len_prev_files += n_data

        else:
            self.embedder_dataset = None
            self.current_file_id = -1
            self.len_prev_files = 0
            self.graph_dataset_idx = 0

    def __iter__(self):
        if self.embedder_dataset is None:
            self.load_next_file()

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


def worker_init_factory(idx):
    def worker_init_fn(worker_id, idx):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        if idx is None:
            idx = list(range(len(dataset)))
        n_data = len(idx)
        n_data_per_worker = [0] + [
            (1 + k) * (n_data // worker_info.num_workers)
            + (k == 0) * (n_data % worker_info.num_workers)
            for k in range(worker_info.num_workers)
        ]
        idx_split = idx[n_data_per_worker[worker_id] : n_data_per_worker[worker_id + 1]]
        dataset.update_idx(idx_split)

    return partial(worker_init_fn, idx=idx)


def get_embedding_loader(args):
    model_files = [f"{model_name}.npy" for model_name in args.embedders_to_simulate]
    with open(os.path.join(args.data_dir, args.dataset, "smiles.json")) as f:
        smiles = json.load(f)

    n_data = len(smiles)
    idx_train = torch.randperm(n_data)
    idx_valid = idx_train[: int(n_data * args.valid_prop)].tolist()
    idx_valid.sort()
    idx_train = idx_train[int(n_data * args.valid_prop) :].tolist()
    idx_train.sort() # idx are used by the workers when initialized

    dataset_train = DistillDataset(
        args.data_dir, args.dataset, args.embedders_to_simulate
    )
    dataset_valid = DistillDataset(
        args.data_dir, args.dataset, args.embedders_to_simulate
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_factory(idx_train),
        prefetch_factor=1000,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_factory(idx_valid),
        prefetch_factor=1000,
        pin_memory=True,
    )

    # clear hack
    dummy = dataset_train = DistillDataset(
        args.data_dir, args.dataset, args.embedders_to_simulate
    )
    dummy.load_next_file()
    embs_dim = dummy.embedder_dataset.embs_dim
    del dummy

    return train_loader, valid_loader, embs_dim, {"train": len(idx_train), "valid": len(idx_valid)}


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="hERG")

    args = parser.parse_args()

    MODELS = [
        "GraphMVP",
        "GROVER",
        "GraphLog",
        "GraphCL",
        "InfoGraph",
        "ChemBertMLM-10M",
        "ChemGPT-4.7M",
        "DenoisingPretrainingPQCMv4",
        "FRAD_QM9",
        "MolR_gat",
        "MolR_gcn",
        "MolR_tag",
        "ThreeDInfomax",
    ]

    data_dir = args.data_dir
    data = args.dataset
    data_path = f"{data_dir}/{data}"

    dataset = DistillDataset(data_dir, data, MODELS)

    for graph, embs in tqdm(dataset.__iter__()):
        pass

    dataset = DistillDataset(data_dir, data, MODELS)
    idx = np.random.choice(len(dataset), 500)
    idx.sort()
    dataset.update_idx(idx)

    for graph, embs in tqdm(dataset.__iter__()):
        pass

    dataset = DistillDataset(data_dir, data, ["GraphMVP"])
    idx = np.random.choice(len(dataset), 500)
    idx.sort()

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=10,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_factory(idx),
    )
    n_observed = 0
    for batch in tqdm(dataloader):
        n_observed += len(batch[0])
    assert n_observed == 500

    dataset = DistillDataset(data_dir, data, ["GraphMVP"])
    idx = np.random.choice(len(dataset), 500)
    idx.sort()

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=2,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_factory(idx),
    )
    n_observed = 0
    for batch in tqdm(dataloader):
        n_observed += len(batch[0])
    assert n_observed == 500

    print("fully passed")
