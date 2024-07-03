import argparse
import json
import yaml
import os
import numpy as np

import torch
import torch.utils.data as tdata
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset

from model.model_gm import Model_GM
from model.std_gnn import GNN_graphpred, GNN
from trainer.trainer_gm import TrainerGM
from data.embedding_dataloader import EmbeddingDataset, get_embedding_loader
from data.data_encoding import DistillGraphDataset

import wandb
import logging
from emir.estimators.knife import KNIFE
from emir.estimators.knife_estimator import KNIFEArgs


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="hERG_Karim")

    parser.add_argument(
        "--embedders-to-simulate",
        nargs="+",
        type=str,
        default=[
            "FRAD_QM9",
            "ChemBertMTR-10M",
            "ChemBertMLM-10M",
            "GraphCL",
            "GraphMVP",
        ],
    )

    # training parameters
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--valid-prop", type=float, default=0.1)

    # model parameters
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gine",
        choices=["gin", "gine", "gcn", "gat", "graphsage", "tag", "arma", "gatv2"],
    )
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--drop-ratio", type=float, default=0.0)
    parser.add_argument("--batch-norm-type", type=str, default="batch")
    parser.add_argument("--graph-pooling", type=str, default="mean")

    parser.add_argument("--knifes-config", type=str, default="hp/knifes.yaml")

    # other parameters
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="../results")
    parser.add_argument("--save-name", type=str, default="tmp")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results")

    return parser


def main(args):
    # get all embeddings datasets
    emb_loader, emb_loader_valid, embs_dim, idx_train, idx_valid = get_embedding_loader(
        args
    )

    # get graph dataset

    graph_input = DistillGraphDataset(os.path.join(args.data_dir, args.dataset))
    graph_loader = DataLoader(
        graph_input[idx_train],
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True,
        shuffle=False,
    )

    graph_loader_valid = DataLoader(
        graph_input[idx_valid],
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
    )

    # get model
    gnn = GNN_graphpred(args, GNN(**args.__dict__))
    model = Model_GM(
        gnn,
    )
    model = torch.compile(model, fullgraph=True, dynamic=True)
    if os.path.exists(args.knifes_config):
        with open(args.knifes_config, "r") as f:
            knifes_config = yaml.safe_load(f)
            knifes_config = KNIFEArgs(**knifes_config)
    else:
        knifes_config = KNIFEArgs(device=args.device)
        with open(args.knifes_config, "w") as f:
            yaml.dump(knifes_config.__dict__, f)

    knifes = torch.nn.ModuleList(
        [
            KNIFE(
                args=knifes_config,
                zc_dim=args.dim,
                zd_dim=emb_dim,
            )
            for emb_dim in embs_dim
        ]
    )

    model = model.to(args.device)
    print(model)
    # get optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.L1Loss()
    scheduler = None  # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    # optimizer, T_0=(args.num_epochs * 4) // 10, eta_min=args.lr / 100, T_mult=1
    # )

    trainer = TrainerGM(
        model,
        knifes,
        optimizer,
        criterion,
        scheduler=scheduler,
        device=args.device,
        batch_size=args.batch_size,
        wandb=args.wandb,
        embedder_name_list=args.embedders_to_simulate,
        out_dir=args.out_dir,
    )

    trainer.train(
        graph_loader,
        emb_loader,
        graph_loader_valid,
        emb_loader_valid,
        args.num_epochs,
        args.log_interval,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    args.out_dir = os.path.join(args.out_dir, args.dataset)

    if args.wandb:
        wandb.init(
            project="mol-distill",
            allow_val_change=True,
        )
        args.out_dir = os.path.join(args.out_dir, wandb.run.name)

        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("eval_loss", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        for embedder in args.embedders_to_simulate:
            wandb.define_metric(f"train_loss_{embedder}", step_metric="epoch")
        wandb.config.update(args)

    os.makedirs(args.out_dir, exist_ok=True)
    # save args
    with open(os.path.join(args.out_dir, "args.yaml"), "w") as f:
        yaml.dump(args.__dict__, f)

    main(args)
