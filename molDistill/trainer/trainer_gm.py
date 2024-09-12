import os
import time

import torch
from tqdm import tqdm

from molDistill.tracing_decorator import tracing_decorator
from molDistill.trainer import Trainer


class TrainerGM(Trainer):
    def __init__(
        self,
        model,
        knifes,
        optimizer,
        device,
        batch_size,
        sizes,
        scheduler=None,
        wandb=False,
        embedder_name_list=None,
        out_dir=None,
        teacher_bn=None,
    ):
        super().__init__(
            model,
            optimizer,
            device,
            batch_size,
            sizes,
            scheduler=scheduler,
            wandb=wandb,
            embedder_name_list=embedder_name_list,
            out_dir=out_dir,
        )
        self.knifes = knifes
        self.knife_optimizer = torch.optim.AdamW(
            knifes.parameters(), lr=self.optimizer.param_groups[0]["lr"]
        )
        self.teacher_bn = teacher_bn

    @tracing_decorator("knife")
    def get_knife_loss(self, embeddings, embs, loss_per_embedder=None):
        if not self.teacher_bn is None:
            for i in range(len(embs)):
                embs[i] = self.teacher_bn[i](embs[i])
        losses = [
            self.knifes[i](embeddings, embs[i]) / embs[i].shape[1]
            for i in range(len(embs))
        ]
        loss = sum(losses)
        if loss_per_embedder is not None:
            for i, l in enumerate(losses):
                loss_per_embedder[self.embedder_name_list[i]] += l
        return loss

    def get_loss(
        self,
        graph,
        embs,
        backward=True,
        loss_per_embedder=None,
    ):
        embeddings = self.model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
            size=len(graph.smiles),
        )
        for i in range(len(embs)):
            embs[i] = embs[i].to(self.device, non_blocking=True)
        loss = self.get_knife_loss(
            embeddings, embs, loss_per_embedder=loss_per_embedder
        )
        if backward:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.knife_optimizer.step()
        return loss

    def set_train(self):
        self.model.train()
        self.knifes.train()
        if self.teacher_bn is not None:
            self.teacher_bn.train()

    def set_eval(self):
        self.model.eval()
        self.knifes.eval()
        if self.teacher_bn is not None:
            self.teacher_bn.eval()

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.knife_optimizer.zero_grad()
