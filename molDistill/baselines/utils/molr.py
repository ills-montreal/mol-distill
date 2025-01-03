from typing import List

import torch
from molecule.models.molr import MolR


@torch.no_grad()
def get_embeddings_from_molr(
    smiles: List[str],
    transformer_name: str = "",
    batch_size: int = 2048,
    device: str = "cpu",
    **kwargs,
):
    model = MolR(transformer_name, batch_size=batch_size, device=device)

    return model(smiles)
