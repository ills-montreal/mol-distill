import os
import time

import torch
from tqdm import tqdm


class Trainer:
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
        **kwargs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.wandb = wandb
        self.embedder_name_list = embedder_name_list
        self.batch_size = batch_size
        self.out_dir = out_dir

        self.sizes = sizes

    def get_loss(self, graph, embs, backward=True, loss_per_embedder=None):
        raise NotImplementedError

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def train_epoch(self, train_loader, epoch, prof=None):
        self.set_train()
        train_loss = 0
        train_loss_per_embedder = {name: 0 for name in self.embedder_name_list}
        for batch_idx, (graph, embs) in enumerate(
            tqdm(
                train_loader,
                desc=f"Training || epoch {epoch} ||  ",
                total=len(train_loader),
            ),
        ):
            self.zero_grad()

            graph = graph.to(self.device, non_blocking=True)

            loss = self.get_loss(
                graph,
                embs,
                backward=True,
                loss_per_embedder=train_loss_per_embedder,
            )
            train_loss += loss
            if prof is not None:
                prof.step()
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

            train_loss, train_loss_per_embedder = self.train_epoch(
                train_loader,
                epoch,
            )

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
        self.set_eval()

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
