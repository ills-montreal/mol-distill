import os
from argparse import Namespace

import torch
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from molDistill.data.data_encoding import DistillGraphDataset
from molDistill.model.model_gm import Model_GM
from molDistill.model.std_gnn import GNN_graphpred, GNN


@torch.no_grad()
def get_embeddings_from_distill_model(
    smiles,
    mols,
    path,
    transformer_name,
    device,
    dataset,
    DATA_PATH,
):
    graph_input = DistillGraphDataset(DATA_PATH)
    dataloader = DataLoader(
        graph_input,
        batch_size=128,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
    )
    args_path = os.path.dirname(path)
    with open(os.path.join(args_path, "args.yaml"), "r") as f:
        args = yaml.safe_load(f)
    gnn = GNN(**args)
    model = GNN_graphpred(Namespace(**args), gnn)
    model = Model_GM(model)
    model.load_state_dict(torch.load(path, map_location=device))

    model.eval()
    model = model.to(device)
    embeddings = []
    for graph in tqdm(dataloader, desc="Extracting embeddings"):
        graph = graph.to(device)
        e = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
            size=len(graph.smiles),
        )
        embeddings.append(e.cpu())
    return torch.cat(embeddings)
