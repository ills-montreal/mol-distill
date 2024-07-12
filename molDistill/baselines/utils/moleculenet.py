import os
from typing import List, Optional

import datamol as dm
import torch
import torch_geometric.nn.pool as tgp
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from molDistill.baselines.models.moleculenet_models import GNN
from molDistill.baselines.utils.moleculenet_encoding import mol_to_graph_data_obj_simple

MODEL_PARAMS = {
    "num_layer": 5,
    "emb_dim": 300,
    "JK": "last",
    "drop_ratio": 0.0,
    "gnn_type": "gin",
}


@torch.no_grad()
def get_embeddings_from_model_moleculenet(
    smiles: List[str],
    mols: Optional[List[dm.Mol]] = None,
    path: str = "backbone_pretrained_models/GROVER/grover.pth",
    pooling_method=tgp.global_mean_pool,
    transformer_name: str = "",
    device: str = "cpu",
    batch_size: int = 2048,
    dataset: Optional[str] = None,
    DATA_PATH: str = "data",
    i_file=None,
    **kwargs,
):
    if i_file is None:
        graph_input_path = f"{DATA_PATH}/graph_input" if dataset is not None else None
    else:
        graph_input_path = f"{DATA_PATH}/graph_input/graph_input_{i_file}"
    embeddings = []
    molecule_model = GNN(**MODEL_PARAMS).to(device)
    if not path == "":
        molecule_model.load_state_dict(torch.load(path))

    molecule_model.eval()

    if graph_input_path is None or not os.path.exists(graph_input_path):
        graph_input = []
        for s in tqdm(smiles, desc="Converting smiles to graph data object"):
            try:
                graph_input.append(
                    mol_to_graph_data_obj_simple(dm.to_mol(s), smiles=s).to(device)
                )
            except Exception as e:
                print(f"Failed to convert {s} to graph data object.")
                raise e
        dataset = InMemoryDataset.save(graph_input, graph_input_path)

    graph_input = InMemoryDataset()
    graph_input.load(graph_input_path)

    dataloader = DataLoader(
        graph_input,
        batch_size=batch_size,
        shuffle=False,
    )

    for b in tqdm(
        dataloader,
        desc="Computing embeddings from model",
        total=len(graph_input) // batch_size + 1,
    ):
        emb = pooling_method(molecule_model(b.x, b.edge_index, b.edge_attr), b.batch)
        embeddings.append(emb)
    embeddings = torch.cat(embeddings, dim=0)
    del molecule_model
    return embeddings
