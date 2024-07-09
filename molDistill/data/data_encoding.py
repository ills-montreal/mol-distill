import os
from typing import Optional, Type
import json

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.separate import separate
from torch_geometric.io import fs
from tqdm import tqdm

allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_hybridization_list": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}
allowable_features_edge = {
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}
node_embedding_order = {
    "possible_atomic_num_list": 0,
    "possible_chirality_list": 1,
    "possible_hybridization_list": 2,
    "possible_formal_charge_list": 3,
    "possible_numH_list": 4,
    "possible_implicit_valence_list": 5,
    "possible_degree_list": 6,
}
edge_embedding_order = {
    "possible_bonds": 0,
    "possible_bond_dirs": 1,
}


def mol_to_graph_data_obj_simple(
    mol: Chem.rdchem.Mol,
    y: Optional[float] = None,
    smiles: Optional[str] = None,
):
    """used in MoleculeDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr"""

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = (
            [allowable_features["possible_atomic_num_list"].index(atom.GetAtomicNum())]
            + [allowable_features["possible_chirality_list"].index(atom.GetChiralTag())]
            + [
                allowable_features["possible_hybridization_list"].index(
                    atom.GetHybridization()
                )
            ]
            + [
                allowable_features["possible_formal_charge_list"].index(
                    atom.GetFormalCharge()
                )
            ]
            + [allowable_features["possible_numH_list"].index(atom.GetTotalNumHs())]
            + [
                allowable_features["possible_implicit_valence_list"].index(
                    atom.GetImplicitValence()
                )
            ]
            + [allowable_features["possible_degree_list"].index(atom.GetDegree())]
        )
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 2  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                allowable_features_edge["possible_bonds"].index(bond.GetBondType())
            ] + [allowable_features_edge["possible_bond_dirs"].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)

    return data


class DistillGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, max_len=50000):
        self.root = root
        self.max_len = max_len
        self.data = []
        os.makedirs(os.path.join(self.root, "raw"), exist_ok=True)
        if not os.path.exists(self.raw_paths[0]):
            with open(self.raw_paths[0], "w") as f:
                smiles = json.load(open(os.path.join(self.root, "smiles.json")))
                json.dump(smiles, f)

        with open(self.raw_paths[0], "r") as f:
            self.smiles = json.load(f)

        self.num_files = len(self.smiles) // self.max_len + 1

        super(DistillGraphDataset, self).__init__(root, transform, pre_transform)

        self.load_processed()

    @property
    def raw_file_names(self):
        return ["smiles.json"]

    @property
    def processed_file_names(self):
        return [os.path.join(f"graph_input_di_{i}.pt") for i in range(self.num_files)]

    def download(self):
        pass

    def process(self):
        graph_input = []
        for s in tqdm(self.smiles):
            graph = mol_to_graph_data_obj_simple(Chem.MolFromSmiles(s), smiles=s)
            graph_input.append(graph)
        n_saves = len(graph_input) // self.max_len + 1
        for i in range(n_saves):
            self.save(
                graph_input[i * self.max_len : (i + 1) * self.max_len],
                self.processed_paths[i],
            )
            self.num_files += 1

    def load_processed(self, data_cls: Type[BaseData] = Data) -> None:
        self.data = []
        for p in self.processed_paths:
            if not os.path.exists(p):
                break
            out = fs.torch_load(p)
            assert isinstance(out, tuple)
            assert len(out) == 2 or len(out) == 3

            if len(out) == 2:  # Backward compatibility.
                data, slice = out
            else:
                data, slice, data_cls = out

            if not isinstance(data, dict):  # Backward compatibility.
                data = data
            else:
                data = data_cls.from_dict(data)
            for idx in range(len(data.smiles)):
                graph = separate(
                    cls=data.__class__,
                    batch=data,
                    idx=idx,
                    slice_dict=slice,
                    decrement=False,
                )
                self.data.append(graph)

    def get(self, idx):
        return self.data[idx]

    def len(self) -> int:
        return len(self.data)

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="hERG_Karim")

    args = parser.parse_args()

    data_dir = args.data_dir
    dataset = args.dataset
    data_path = f"{data_dir}/{dataset}"

    dataset = DistillGraphDataset(data_path)

    print(f"Saved graph data object")
