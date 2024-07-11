import os
import json

from torch.utils.data import IterableDataset

from molDistill.data.data_encoding import DistillGraphDataset
from molDistill.data.embedding_dataloader import EmbeddingDataset


class DistillDataset(IterableDataset):
    def __init__(
        self, data_dir, dataset, model_names, model_files, idx=None, shuffle=False
    ):
        self.data_dir = data_dir
        self.graph_dataset = DistillGraphDataset(os.path.join(data_dir, dataset))
        self.embedder_dataset = None

        if model_file[0].endswith(".npy"):
            self.embedder_files = model_files
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
        self.load_next_file()

    def load_next_file(self):
        self.current_file_id += 1
        if self.current_file_id < len(self.embedder_files):
            if not self.embedder_dataset is None:
                self.len_prev_files += len(self.embedder_dataset)

            self.embedder_dataset = EmbeddingDataset(
                self.data_dir,
                self.dataset,
                self.model_names,
                self.embedder_files[self.current_file_id],
            )
            n_data = len(self.embedder_dataset)
            self.file_conditioned_idx = [
                idx - self.len_prev_files
                for idx in self.idx
                if idx >= self.len_prev_files and idx < self.len_prev_files + n_data
            ]
            self.embedder_dataset.update_idx(self.file_conditioned_idx)

        else:
            self.embedder_dataset = None
            self.current_file_id = -1
            self.len_prev_files = 0
            self.load_next_file()

    def __iter__(self):
        if self.has_been_seen.all():
            self.load_next_file()
        if self.current_file_id == 0:
            raise StopIteration

        for i in range(len(self.embedder_dataset)):
            yield self.graph_dataset[i + self.len_prev_files], self.embedder_dataset[i]

    def __next__(self):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="hERG_Karim")

    args = parser.parse_args()

    data_dir = args.data_dir
    dataset = args.dataset
    data_path = f"{data_dir}/{dataset}"

    dataset = DistillDataset(
        data_dir, dataset, ["GraphMVP"], ["GraphMVP.npy"], idx=None, shuffle=False
    )

    for graph, embs in dataset:
        print(graph, embs)

    print(f"Saved graph data object")