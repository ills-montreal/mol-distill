import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import wandb
import yaml
from emir.estimators.knife import KNIFE
from emir.estimators.knife_estimator import KNIFEArgs
from torch_geometric.loader import DataLoader

from molDistill.data.data_encoding import DistillGraphDataset
from molDistill.data.combined_dataset import get_embedding_loader
from molDistill.model.model_gm import Model_GM
from molDistill.model.std_gnn import GNN_graphpred, GNN
from molDistill.trainer.trainer_gm import TrainerGM
from molDistill.utils.parser import update_grouped_models, get_pretraining_args


def get_parser():
    parser = get_pretraining_args()

    # model parameters

    parser.add_argument("--knifes-config", type=str, default="hp/knifes.yaml")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--use-teacher-bn", action="store_true")
    parser.set_defaults(use_teacher_bn=True)

    return parser


def main(args):
    # get all embeddings datasets
    train_loader, valid_loader, embs_dim, sizes = get_embedding_loader(args)

    # get model
    gnn = GNN(**args.__dict__)
    mol_model = GNN_graphpred(args, gnn)
    if args.checkpoint is not None:
        mol_model.from_pretrained(args.checkpoint)
    model = Model_GM(
        mol_model,
    )
    if os.path.exists(args.knifes_config):
        with open(args.knifes_config, "r") as f:
            knifes_config = yaml.safe_load(f)
            knifes_config = KNIFEArgs(**knifes_config)
    else:
        knifes_config = KNIFEArgs(device=args.device)
        os.makedirs(os.path.dirname(args.knifes_config), exist_ok=True)
        with open(args.knifes_config, "w") as f:
            yaml.dump(knifes_config.__dict__, f)

    knifes = []
    if args.use_teacher_bn:
        teacher_bn = []
    else:
        teacher_bn = None
    for emb_dm in embs_dim:
        knife = KNIFE(
            args=knifes_config,
            zc_dim=args.dim,
            zd_dim=emb_dm,
        ).kernel_cond

        knifes.append(knife)
        if args.use_teacher_bn:
            teacher_bn.append(
                torch.nn.BatchNorm1d(emb_dm, affine=False, device=args.device)
            )

    knifes = torch.nn.ModuleList(knifes)
    if args.use_teacher_bn:
        teacher_bn = torch.nn.ModuleList(teacher_bn)

    model = model.to(args.device)
    # get optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = None  # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    # optimizer, T_0=(args.num_epochs * 4) // 10, eta_min=args.lr / 100, T_mult=1
    # )

    trainer = TrainerGM(
        model,
        knifes,
        optimizer,
        scheduler=scheduler,
        device=args.device,
        batch_size=args.batch_size,
        wandb=args.wandb,
        embedder_name_list=args.embedders_to_simulate,
        out_dir=args.out_dir,
        sizes=sizes,
        teacher_bn=teacher_bn,
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
        wandb.init(
            project="mol-distill",
            allow_val_change=True,
        )

        if not wandb.run.name is None:
            args.out_dir = os.path.join(args.out_dir, wandb.run.name)
        print(args.out_dir)

        wandb.config.update(args)

        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("eval_loss", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        for embedder in args.embedders_to_simulate:
            wandb.define_metric(f"train_loss_{embedder}", step_metric="epoch")
            wandb.define_metric(f"test_loss_{embedder}", step_metric="epoch")

    os.makedirs(args.out_dir, exist_ok=True)
    # save args
    with open(os.path.join(args.out_dir, "args.yaml"), "w") as f:
        yaml.dump(args.__dict__, f)

    main(args)

    # Create a stop.txt file to indicate the end of the training
    with open(os.path.join(args.out_dir, "stop.txt"), "w") as f:
        f.write("Training finished")
