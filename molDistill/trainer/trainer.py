import time
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        batch_size,
        scheduler=None,
        wandb=False,
        embedder_name_list=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.wandb = wandb
        self.embedder_name_list = embedder_name_list
        self.batch_size = batch_size

    def train_epoch(self, input_loader, embedding_loader, epoch):
        self.model.train()
        train_loss = 0
        train_loss_per_embedder = {name: 0 for name in self.embedder_name_list}
        for batch_idx, (graph, embs) in enumerate(
            zip(
                input_loader,
                tqdm(embedding_loader, desc=f"Training || epoch {epoch} ||  ", total=len(input_loader)),
            )
        ):
            graph = graph.to(self.device)
            self.optimizer.zero_grad()
            embeddings = self.model(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch,
                size=len(graph.smiles),
            )
            for i, (emb, emb_pred) in enumerate(zip(embs, embeddings)):
                emb = emb.to(self.device)
                loss = self.criterion(emb_pred, emb)
                loss.backward(retain_graph=i < len(embs) - 1)
                train_loss += loss.item()
                train_loss_per_embedder[self.embedder_name_list[i]] += loss.item()
            self.optimizer.step()
        for name in self.embedder_name_list:
            train_loss_per_embedder[name] /= len(input_loader)
        return train_loss / len(input_loader), train_loss_per_embedder

    def train(
        self,
        input_loader,
        embedding_loader,
        input_loader_valid,
        embedding_loader_valid,
        num_epochs,
        log_interval,
        tmp,
    ):
        mean_time = 0
        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss, train_loss_per_embedder = self.train_epoch(
                input_loader, embedding_loader, epoch
            )
            t1 = time.time()
            # z = self.encode_loader(input_loader)
            mean_time += t1 - t0
            if epoch % log_interval == 0:
                mean_time /= log_interval
                eval_loss = self.eval(input_loader_valid, embedding_loader_valid)
                print(f"Epoch {epoch}, Loss: {train_loss} --- {mean_time:.2f} s/epoch")
                print(f"----\tEval Loss: {eval_loss}")
                mean_time = 0
                if self.wandb:
                    import wandb

                    dict_to_log = {
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        "epoch": epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
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
        for batch_idx, (graph, embs) in enumerate(zip(input_loader, embedding_loader)):
            graph = graph.to(self.device)
            embeddings = self.model(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch,
                size=len(graph.smiles),
            )
            for emb, emb_pred in zip(embs, embeddings):
                loss = self.criterion(emb_pred, emb.to(self.device))
                eval_loss += loss.item()
        return eval_loss / len(input_loader)

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
