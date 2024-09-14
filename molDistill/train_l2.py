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
from molDistill.utils.parser import update_grouped_models, get_pretraining_args


def get_parser():
    parser = get_pretraining_args()
    # model parameters
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
        wandb.init(project="mol-distill", config=args, allow_val_change=True)
        if not wandb.run.name is None:
            args.out_dir = os.path.join(args.out_dir, wandb.run.name)

        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("eval_loss", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        for embedder in args.embedders_to_simulate:
            wandb.define_metric(f"train_loss_{embedder}", step_metric="epoch")
            wandb.define_metric(f"test_loss_{embedder}", step_metric="epoch")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.yaml"), "w") as f:
        yaml.dump(args.__dict__, f)

    main(args)
    with open(os.path.join(args.out_dir, "stop.txt"), "w") as f:
        f.write("Training finished")
