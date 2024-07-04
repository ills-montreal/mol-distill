import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.conv import (
    GINConv,
    GCNConv,
    GATConv,
    SAGEConv,
    GINEConv,
    TAGConv,
    ARMAConv,
    GATv2Conv,
)
from torch_geometric.nn.models import MLP

from data.data_encoding import (
    allowable_features,
    allowable_features_edge,
    node_embedding_order,
    edge_embedding_order,
)


class GNN(nn.Module):
    def __init__(
        self,
        n_layer,
        dim,
        drop_ratio=0.0,
        gnn_type="gin",
        batch_norm_type="batch",
        node_embedding_keys=["possible_atomic_num_list", "possible_chirality_list"],
        edge_embedding_keys=["possible_bonds"],
        **kwargs,
    ):
        super(GNN, self).__init__()
        if n_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.n_layer = n_layer
        self.gnn_type = gnn_type
        self.dim = dim

        self.drop_ratio = drop_ratio
        self.node_embedding_keys = node_embedding_keys
        self.edge_embedding_keys = edge_embedding_keys

        self.x_embedding = nn.ModuleList(
            [
                nn.Embedding(len(allowable_features[key]), dim)
                for key in node_embedding_keys
            ]
        )
        self.edge_embedding = nn.ModuleList(
            [
                nn.Embedding(len(allowable_features_edge[key]), dim)
                for key in edge_embedding_keys
            ]
        )

        self.gnns = nn.ModuleList()
        for i, layer in enumerate(range(n_layer)):
            if gnn_type == "gin":
                h_theta = MLP([dim, dim, dim])
                self.gnns.append(GINConv(h_theta))
            elif gnn_type == "gine":
                h_theta = MLP([dim, dim, dim])
                self.gnns.append(GINEConv(h_theta, edge_dim=dim))
            elif gnn_type == "gcn":
                self.gnns.append(
                    GCNConv(dim, dim, add_self_loops=False, normalize=False)
                )
            elif gnn_type == "gat":
                self.gnns.append(GATConv(dim, dim, edge_dim=dim))
            elif gnn_type == "graphsage":
                self.gnns.append(SAGEConv(dim, dim))
            elif gnn_type == "tag":
                self.gnns.append(TAGConv(dim, dim, 2))
            elif gnn_type == "arma":
                self.gnns.append(ARMAConv(dim, dim, num_stacks=1))
            elif gnn_type == "gatv2":
                self.gnns.append(GATv2Conv(dim, dim, edge_dim=dim))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(n_layer):
            if batch_norm_type == "batch":
                self.batch_norms.append(nn.BatchNorm1d(dim))
            elif batch_norm_type == "layer":
                self.batch_norms.append(nn.LayerNorm(dim))
            else:
                self.batch_norms.append(nn.Identity())

    def forward_embeddings(self, x, edge_index, edge_attr):
        h = x

        for layer in range(self.n_layer):
            if self.gnn_type in ["gine", "gat", "gatv2"]:
                h = self.gnns[layer](h, edge_index, edge_attr)
            else:
                h = self.gnns[layer](h, edge_index)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.n_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        return h

    def forward(self, x, edge_index, edge_attr):
        x = sum(
            [
                self.x_embedding[i](
                    x[:, node_embedding_order[self.node_embedding_keys[i]]]
                )
                for i in range(len(self.node_embedding_keys))
            ]
        )

        if self.gnn_type in ["gine", "gat", "gatv2"]:
            edge_attr = sum(
                [
                    self.edge_embedding[i](
                        edge_attr[:, edge_embedding_order[self.edge_embedding_keys[i]]]
                    )
                    for i in range(len(self.edge_embedding_keys))
                ]
            )

        return self.forward_embeddings(x, edge_index, edge_attr)


class GNN_graphpred(nn.Module):
    def __init__(self, args, molecule_model=None):
        super(GNN_graphpred, self).__init__()

        if args.n_layer < 2:
            raise ValueError("# layers must > 1.")

        self.molecule_model = molecule_model
        self.n_layer = args.n_layer
        self.emb_dim = args.dim

        # Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        if args.batch_norm_type == "batch":
            self.batch_norm = nn.BatchNorm1d
        elif args.batch_norm_type == "layer":
            self.batch_norm = nn.LayerNorm
        else:
            self.batch_norm = lambda x: nn.Identity

        sequential_args = []
        for i in range(args.n_MLP_layer):
            if i == 0:
                sequential_args.append(
                    (f"fc-{i}", nn.Linear(self.emb_dim, self.emb_dim))
                )
            else:
                sequential_args.append((f"relu-{i}", nn.ReLU()))
                sequential_args.append(
                    (f"fc-{i}", nn.Linear(self.emb_dim, self.emb_dim))
                )
                sequential_args.append((f"bn-{i}", self.batch_norm(self.emb_dim)))

        self.fully_connected = nn.Sequential(OrderedDict(sequential_args))

    def from_pretrained(self, model_file):
        if model_file == "":
            return
        self.molecule_model.load_state_dict(torch.load(model_file))
        return

    def forward(self, x, edge_index, edge_attr, batch, size=None):
        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch, size=size)
        graph_representation = self.fully_connected(graph_representation)

        return graph_representation
