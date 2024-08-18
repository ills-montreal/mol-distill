import os
import time

import torch
from tqdm import tqdm

from molDistill.tracing_decorator import tracing_decorator


class TrainerGM:
    def __init__(
        self,
        model,
        knifes,
        optimizer,
        criterion,
        device,
        batch_size,
        sizes,
        scheduler=None,
        wandb=False,
        embedder_name_list=None,
        out_dir=None,
        teacher_bn = None,
    ):
        self.model = model
        self.knifes = knifes
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.wandb = wandb
        self.embedder_name_list = embedder_name_list
        self.batch_size = batch_size
        self.knife_optimizer = torch.optim.AdamW(
            knifes.parameters(), lr=self.optimizer.param_groups[0]["lr"]
        )
        self.out_dir = out_dir

        self.sizes = sizes
        self.teacher_bn = teacher_bn

    @tracing_decorator("knife")
    def get_knife_loss(self, embeddings, embs, loss_per_embedder=None):
        if not self.teacher_bn is None:
            for i in range(len(embs)):
                embs[i] = self.teacher_bn[i](embs[i])
        losses = [
            self.knifes[i](embeddings, embs[i])# / embs[i].shape[1]
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
        return_embs=False,
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
            self.optimizer.step()
            self.knife_optimizer.step()
            self.optimizer.zero_grad()
            self.knife_optimizer.zero_grad()
        if return_embs:
            return loss, embeddings
        return loss

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        self.knifes.train()
        if self.teacher_bn is not None:
            self.teacher_bn.train()

        train_loss = 0
        train_loss_per_embedder = {name: 0 for name in self.embedder_name_list}
        for batch_idx, (graph, embs) in enumerate(
            tqdm(
                train_loader,
                desc=f"Training || epoch {epoch} ||  ",
                total=len(train_loader),
            ),
        ):
            self.optimizer.zero_grad()
            self.knife_optimizer.zero_grad()

            graph = graph.to(self.device)

            loss = self.get_loss(
                graph,
                embs,
                backward=True,
                loss_per_embedder=train_loss_per_embedder,
            )
            train_loss += loss
        for name in self.embedder_name_list:
            train_loss_per_embedder[name] = (
                train_loss_per_embedder[name].item() / self.sizes["train"]
            )
        return train_loss.item() / self.sizes["train"], train_loss_per_embedder

    def train(
        self,
        train_loader,
        valid_loader,
        num_epochs,
        log_interval,
    ):
        mean_time = 0
        min_eval_loss = float("inf")

        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss, train_loss_per_embedder = self.train_epoch(train_loader, epoch)

            t1 = time.time()

            dict_to_log = {
                "train_loss": train_loss,
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            # z = self.encode_loader(input_loader)
            mean_time += t1 - t0
            if epoch % log_interval == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.out_dir, f"model_{epoch}.pth"),
                )

                mean_time /= log_interval
                eval_loss, test_loss_per_embedder = self.eval(valid_loader, epoch)
                print(f"Epoch {epoch}, Loss: {train_loss} --- {mean_time:.2f} s/epoch")
                print(f"----\tEval Loss: {eval_loss}")
                mean_time = 0
                dict_to_log["eval_loss"] = eval_loss

                if eval_loss < min_eval_loss:
                    min_eval_loss = eval_loss
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.out_dir, "best_model.pth"),
                    )

            if self.wandb:
                import wandb

                for name, loss in train_loss_per_embedder.items():
                    dict_to_log[f"train_loss_{name}"] = loss
                for name, loss in test_loss_per_embedder.items():
                    dict_to_log[f"eval_loss_{name}"] = loss
                wandb.log(dict_to_log)
            if self.scheduler is not None:
                self.scheduler.step()
        return

    @torch.no_grad()
    def eval(self, valid_loader, epoch):
        self.model.eval()
        self.knifes.eval()
        if self.teacher_bn is not None:
            self.teacher_bn.eval()

        eval_loss = 0
        embeddings = []
        test_loss_per_embedder = {name: 0 for name in self.embedder_name_list}
        for batch_idx, (graph, embs) in enumerate(
            tqdm(
                valid_loader,
                desc=f"Eval || epoch {epoch} ||  ",
                total=len(valid_loader),
            )
        ):
            graph = graph.to(self.device)
            l = self.get_loss(
                graph,
                embs,
                backward=False,
                loss_per_embedder=test_loss_per_embedder,
            )
            eval_loss += l
        for name in self.embedder_name_list:
            test_loss_per_embedder[name] = (
                test_loss_per_embedder[name].item() / self.sizes["valid"]
            )
        return eval_loss.item() / self.sizes["valid"], test_loss_per_embedder

    @torch.no_grad()
    def encode_loader(self, input_loader):
        self.model.eval()
        embeddings = []
        for graph in input_loader:
            graph = graph.to(self.device)
            embeddings.append(
                self.model(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    graph.batch,
                    size=len(graph.smiles),
                )[0]
            )
        return torch.concatenate(embeddings, dim=0)
