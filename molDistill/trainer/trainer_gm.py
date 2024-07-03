import os

import time
import torch
from tqdm import tqdm


class TrainerGM:
    def __init__(
        self,
        model,
        knifes,
        optimizer,
        criterion,
        device,
        batch_size,
        scheduler=None,
        wandb=False,
        embedder_name_list=None,
        out_dir=None,
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
        self.knife_optimizer = torch.optim.AdamW(knifes.parameters(), lr=1e-3)
        self.out_dir = out_dir

    def get_loss(
        self,
        graph,
        embs,
        backward=True,
        train_loss_per_embedder=None,
        return_embs=False,
    ):
        embeddings = self.model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
            size=len(graph.smiles),
        )
        loss = 0
        for i, emb in enumerate(embs):
            emb = emb.to(self.device)
            knife = self.knifes[i]
            emb_loss = knife.kernel_cond(embeddings, emb)
            if train_loss_per_embedder is not None:
                train_loss_per_embedder[self.embedder_name_list[i]] += emb_loss

            loss += emb_loss
        if backward:
            loss.backward()
            self.optimizer.step()
            self.knife_optimizer.step()
            self.optimizer.zero_grad()
            self.knife_optimizer.zero_grad()
        if return_embs:
            return loss, embeddings
        return loss

    def train_epoch(self, input_loader, embedding_loader, epoch):
        self.model.train()
        train_loss = 0
        train_loss_per_embedder = {name: 0 for name in self.embedder_name_list}
        for batch_idx, (graph, embs) in enumerate(
            zip(
                input_loader,
                tqdm(
                    embedding_loader,
                    desc=f"Training || epoch {epoch} ||  ",
                    total=len(input_loader),
                ),
            )
        ):
            embs, smiles = embs
            assert graph.smiles == smiles
            self.optimizer.zero_grad()
            self.knife_optimizer.zero_grad()
            graph = graph.to(self.device)
            assert graph.smiles == smiles

            loss = self.get_loss(
                graph,
                embs,
                backward=True,
                train_loss_per_embedder=train_loss_per_embedder,
            )
            train_loss += loss
        for name in self.embedder_name_list:
            train_loss_per_embedder[name] = train_loss_per_embedder[name].item() / len(
                input_loader
            )
        return train_loss.item() / len(input_loader), train_loss_per_embedder

    def train(
        self,
        input_loader,
        embedding_loader,
        input_loader_valid,
        embedding_loader_valid,
        num_epochs,
        log_interval,
    ):
        mean_time = 0
        min_eval_loss = float("inf")
        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss, train_loss_per_embedder = self.train_epoch(
                input_loader, embedding_loader, epoch
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
                mean_time /= log_interval
                eval_loss = self.eval(input_loader_valid, embedding_loader_valid)
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
                wandb.log(dict_to_log)
            if self.scheduler is not None:
                self.scheduler.step()
        return

    @torch.no_grad()
    def eval(self, input_loader, embedding_loader):
        self.model.eval()
        eval_loss = 0
        embeddings = []
        for batch_idx, (graph, embs) in enumerate(zip(input_loader, embedding_loader)):
            embs, smiles = embs
            graph = graph.to(self.device)
            l, embs = self.get_loss(graph, embs, backward=False, return_embs=True)
            eval_loss += l
        return eval_loss.item() / len(input_loader)

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
