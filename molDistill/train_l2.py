import argparse
import json
import os
import numpy as np

import torch
import torch.utils.data as tdata
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset

from model.model import Model
from model.std_gnn import GNN_graphpred, GNN
from trainer.trainer import Trainer

import wandb
import logging


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="hERG_Karim")

    parser.add_argument(
        "--embedders-to-simulate",
        nargs="+",
        type=str,
        default=["FRAD_QM9", "ChemBertMTR-10M"],
    )

    # training parameters
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--valid-prop", type=float, default=0.1)

    # model parameters
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gine",
        choices=[
            "gin",
            "gine",
            "gcn",
            "gat",
            "graphsage",
            "tag",
            "arma",
            "gatv2"
        ],
    )
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--drop-ratio", type=float, default=0.0)
    parser.add_argument("--batch-norm-type", type=str, default="layer")
    parser.add_argument("--graph-pooling", type=str, default="mean")

    # other parameters
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="../results")
    parser.add_argument("--save-name", type=str, default="tmp")

    parser.add_argument("--wandb", action="store_true")

    return parser


def main(args):
    # get all embeddings datasets
    all_embedders = []
    all_embedders_valid = []
    embs_dim = []
    idx_valid = None
    idx_train = None
    for embedder in args.embedders_to_simulate:
        embs = np.load(os.path.join(args.data_dir, args.dataset, f"{embedder}.npy"))
        embs = (embs - embs.mean(axis=0)) / (embs.std(axis=0) + 1e-8)
        embs = torch.tensor(embs, dtype=torch.float)
        if idx_valid is None:
            idx = torch.randperm(embs.size(0))
            idx_valid = idx[: int(embs.size(0) * args.valid_prop)]
            idx_train = idx[int(embs.size(0) * args.valid_prop) :]
        all_embedders.append(embs[idx_train])
        all_embedders_valid.append(embs[idx_valid])
        embs_dim.append(embs.shape[1])
    emb_ds = tdata.TensorDataset(*all_embedders)
    emb_loader = tdata.DataLoader(
        emb_ds, batch_size=args.batch_size, num_workers=0, drop_last=True
    )

    emb_ds_valid = tdata.TensorDataset(*all_embedders_valid)
    emb_loader_valid = tdata.DataLoader(
        emb_ds_valid, batch_size=args.batch_size, num_workers=0
    )

    # get graph dataset

    graph_input = DistillGraphDataset(os.path.join(args.data_dir, args.dataset))
    graph_loader = DataLoader(
        graph_input[idx_train],
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True,
    )

    graph_loader_valid = DataLoader(
        graph_input[idx_valid],
        batch_size=args.batch_size,
        num_workers=0,
    )

    # get model
    gnn = GNN_graphpred(args, GNN(**args.__dict__))
    model = Model(
        gnn,
        embs_dim,
    )
    model = model.to(args.device)
    model = torch.compile(model, fullgraph=True, disable=True)
    print(model)
    # get optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=(args.num_epochs * 4) // 10, eta_min=args.lr / 100, T_mult=1
    )

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        scheduler=scheduler,
        device=args.device,
        batch_size=args.batch_size,
        wandb=args.wandb,
        embedder_name_list=args.embedders_to_simulate,
    )

    trainer.train(
        graph_loader,
        emb_loader,
        graph_loader_valid,
        emb_loader_valid,
        args.num_epochs,
        args.log_interval,
        tmp=all_embedders,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="mol-distill", config=args)
        wandb.config.update(args)

        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("eval_loss", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")

    main(args)
