from typing import List

from molecule.models.threedinfomax import ThreeDInfoMax


def get_embeddings_from_model_threedinfomax(
    smiles: List[str],
    transformer_name: str = "",
    batch_size: int = 2048,
    device: str = "cpu",
    path: str = "",
    **kwargs,
):
    model = ThreeDInfoMax(smiles, device=device, batch_size=batch_size, path=path)
    embeddings = model.featurize_data()
    return embeddings
