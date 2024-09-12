import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import torch
import torch.utils.data as tdata
import wandb
from torch_geometric.loader import DataLoader

from molDistill.model.model import Model
from molDistill.model.std_gnn import GNN_graphpred, GNN
from molDistill.trainer.trainer_l2 import Trainer_L2
from molDistill.data.combined_dataset import get_embedding_loader

GROUPED_MODELS = {
    "GNN": [
        # "ContextPred",
        # "GPT-GNN",
        "GraphMVP",
        # "GROVER",
        # "AttributeMask",
        "GraphLog",
        "GraphCL",
        # "InfoGraph",
    ],
    "BERT": [
        # "MolBert",
        # "ChemBertMLM-5M",
        # "ChemBertMLM-10M",
        # "ChemBertMLM-77M",
        "ChemBertMTR-5M",
        "ChemBertMTR-10M",
        "ChemBertMTR-77M",
    ],
    "GPT": [
        # "ChemGPT-1.2B",
        # "ChemGPT-19M",
        # "ChemGPT-4.7M",
    ],
    "Denoising": [
        "DenoisingPretrainingPQCMv4",
        "FRAD_QM9",
    ],
    "MolR": [
        "MolR_gat",
        # "MolR_gcn",
        "MolR_tag",
    ],
    "MoleOOD": [
        # "MoleOOD_OGB_GIN",
        # "MoleOOD_OGB_GCN",
        # "MoleOOD_OGB_SAGE",
    ],
    "ThreeD": [
        "ThreeDInfomax",
    ],
}


def update_grouped_models(embedders):
    new_embedders = []
    for embedder in embedders:
        if embedder in GROUPED_MODELS:
            new_embedders += GROUPED_MODELS[embedder]
        else:
            new_embedders.append(embedder)
    new_embedders = list(set(new_embedders))
    return new_embedders


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="MOSES")

    parser.add_argument(
        "--embedders-to-simulate",
        nargs="+",
        type=str,
        default=["GNN", "BERT", "GPT", "Denoising", "ThreeD", "MolR"],
    )

    # training parameters
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=1000)
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
    parser.add_argument("--n-layer", type=int, default=10)
    parser.add_argument("--n-MLP-layer", type=int, default=1)
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
    parser.add_argument("--out-dir", type=str, default="results")

    parser.add_argument("--wandb", action="store_true")

    return parser


def main(args):
    # get all embeddings datasets
    train_loader, valid_loader, embs_dim, sizes = get_embedding_loader(args)

    # get model
    gnn = GNN(**args.__dict__)
    mol_model = GNN_graphpred(args, gnn)
    model = Model(
        mol_model,
        embs_dim,
    )
    model = model.to(args.device)
    # get optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = None  # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    # optimizer, T_0=(args.num_epochs * 4) // 10, eta_min=args.lr / 100, T_mult=1
    # )

    trainer = Trainer_L2(
        model,
        optimizer,
        scheduler=scheduler,
        device=args.device,
        batch_size=args.batch_size,
        wandb=args.wandb,
        embedder_name_list=args.embedders_to_simulate,
        out_dir=args.out_dir,
        sizes=sizes,
    )

    trainer.train(
        train_loader,
        valid_loader,
        args.num_epochs,
        args.log_interval,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.embedders_to_simulate = update_grouped_models(args.embedders_to_simulate)

    if args.wandb:
        wandb.init(project="mol-distill", config=args)
        wandb.config.update(args)

        if not wandb.run.name is None:
            args.out_dir = os.path.join(args.out_dir, wandb.run.name)

        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("eval_loss", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")

    main(args)
    with open("stop.txt", "w") as f:
        f.write("Training finished")
