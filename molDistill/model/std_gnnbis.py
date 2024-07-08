import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATConv, SAGEConv, TAGConv
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GNN(torch.nn.Module):
    def __init__(self, gnn, n_layer, feature_len, dim):
        super(GNN, self).__init__()
        self.gnn = gnn
        self.n_layer = n_layer
        self.feature_len = feature_len
        self.dim = dim
        self.gnn_layers = ModuleList([])
        if gnn in ["gcn", "gat", "sage", "tag"]:
            for i in range(n_layer):
                if gnn == "gcn":
                    self.gnn_layers.append(
                        GraphConv(
                            in_feats=feature_len if i == 0 else dim,
                            out_feats=dim,
                            activation=None if i == n_layer - 1 else torch.relu,
                        )
                    )
                elif gnn == "gat":
                    num_heads = 16  # make sure that dim is dividable by num_heads
                    self.gnn_layers.append(
                        GATConv(
                            in_feats=feature_len if i == 0 else dim,
                            out_feats=dim // num_heads,
                            activation=None if i == n_layer - 1 else torch.relu,
                            num_heads=num_heads,
                        )
                    )
                elif gnn == "sage":
                    agg = "pool"
                    self.gnn_layers.append(
                        SAGEConv(
                            in_feats=feature_len if i == 0 else dim,
                            out_feats=dim,
                            activation=None if i == n_layer - 1 else torch.relu,
                            aggregator_type=agg,
                        )
                    )
                elif gnn == "tag":
                    hops = 2
                    self.gnn_layers.append(
                        TAGConv(
                            in_feats=feature_len if i == 0 else dim,
                            out_feats=dim,
                            activation=None if i == n_layer - 1 else torch.relu,
                            k=hops,
                        )
                    )
        elif gnn == "sgc":
            self.gnn_layers.append(
                SGConv(in_feats=feature_len, out_feats=dim, k=n_layer)
            )
        else:
            raise ValueError("unknown GNN model")
        self.pooling_layer = SumPooling()
        self.factor = None

    def forward(self, graph):
        feature = graph.ndata["feature"]
        h = one_hot(feature, num_classes=self.feature_len)
        h = torch.sum(h, dim=1, dtype=torch.float)
        for layer in self.gnn_layers:
            h = layer(graph, h)
            if self.gnn == "gat":
                h = torch.reshape(h, [h.size()[0], -1])
        if self.factor is None:
            self.factor = math.sqrt(self.dim) / float(
                torch.mean(torch.linalg.norm(h, dim=1))
            )
        h *= self.factor
        graph_embedding = self.pooling_layer(graph, h)
        return graph_embedding


from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add


class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0.0, gnn_type="gin"):
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNN, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward_embeddings(self, x, edge_index, edge_attr):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        return node_representation

    # def forward(self, x, edge_index, edge_attr):

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        return self.forward_embeddings(x, edge_index, edge_attr)


class GNN_graphpred(nn.Module):
    def __init__(self, args, num_tasks, molecule_model=None):
        super(GNN_graphpred, self).__init__()

        if args.num_layer < 2:
            raise ValueError("# layers must > 1.")

        self.molecule_model = molecule_model
        self.num_layer = args.num_layer
        self.emb_dim = args.emb_dim
        self.num_tasks = num_tasks
        self.JK = args.JK

        # Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(
                self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks
            )
        else:
            self.graph_pred_linear = nn.Linear(self.mult * self.emb_dim, self.num_tasks)
        return

    def from_pretrained(self, model_file):
        if model_file == "":
            return
        self.molecule_model.load_state_dict(torch.load(model_file))
        return

    def get_graph_representation(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        pred = self.graph_pred_linear(graph_representation)

        return graph_representation, pred

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        output = self.graph_pred_linear(graph_representation)

        return output
