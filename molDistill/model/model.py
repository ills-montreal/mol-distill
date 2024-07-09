from typing import List

import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tasks: List[int],
    ):
        super(Model, self).__init__()
        self.backbone = backbone
        self.out_dim = backbone.emb_dim
        self.heads = nn.ModuleList()
        for size in tasks:
            head = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim),
                nn.ReLU(),
                nn.Linear(self.out_dim, size),
                nn.BatchNorm1d(size),
            )
            self.heads.append(head)

    def forward(self, x, edge_index, edge_attr, batch, size):
        emb = self.backbone(x, edge_index, edge_attr, batch, size)

        results = []
        for head in self.heads:
            results.append(head(emb))
        return results

    def encode(self, x, edge_index, edge_attr, batch, size):
        return self.backbone(x, edge_index, edge_attr, batch, size)
