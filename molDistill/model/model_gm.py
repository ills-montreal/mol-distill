from typing import List

import torch.nn as nn
from torch_geometric.data import InMemoryDataset
from torch.profiler import record_function

class Model_GM(nn.Module):
    def __init__(
        self,
        backbone: nn.Module
    ):
        super(Model_GM, self).__init__()
        self.backbone = backbone
        self.out_dim = backbone.emb_dim


    def forward(self, x, edge_index, edge_attr, batch, size):
        with record_function("forward"):
            return self.backbone(x, edge_index, edge_attr, batch, size)

