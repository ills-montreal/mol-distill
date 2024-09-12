import time

import torch
from tqdm import tqdm

from molDistill.trainer import Trainer


class Trainer_L2(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        device,
        batch_size,
        sizes,
        scheduler=None,
        wandb=False,
        embedder_name_list=None,
        out_dir=None,
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
        self.criterion = torch.nn.MSELoss()

    def get_loss(self, graph, embs, backward=True, loss_per_embedder=None):
        embeddings = self.model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
            size=len(graph.smiles),
        )
        for i in range(len(embs)):
            embs[i] = embs[i].to(self.device, non_blocking=True)
        losses = [
            self.criterion(emb_pred, emb) for emb_pred, emb in zip(embeddings, embs)
        ]
        loss = sum(losses)
        if loss_per_embedder is not None:
            for i, name in enumerate(self.embedder_name_list):
                loss_per_embedder[name] += losses[i]
        if backward:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
        return loss
