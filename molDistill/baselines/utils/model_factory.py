moleculenet_models = [
    "Not-trained",
    "AttributeMask",
    "ContextPred",
    "EdgePred",
    "GPT-GNN",
    "InfoGraph",
    "GraphCL",
    "GraphLog",
    "GraphMVP",
    "GROVER",
    "InfoGraph",
]

denoising_models = ["DenoisingPretrainingPQCMv4", "FRAD_QM9"]


class ModelFactory:
    def __new__(cls, name: str):
        if name in moleculenet_models:
            from .moleculenet import get_embeddings_from_model_moleculenet

            return get_embeddings_from_model_moleculenet
        elif name in denoising_models:
            from .denoising_models import get_embeddings_from_model_denoising

            return get_embeddings_from_model_denoising

        elif name.startswith("MolR"):
            from .molr import get_embeddings_from_molr

            return get_embeddings_from_molr

        elif name.startswith("MoleOOD"):
            from .moleood import get_embeddings_from_moleood

            return get_embeddings_from_moleood

        elif name == "ThreeDInfomax":
            from .threedinfomax import get_embeddings_from_model_threedinfomax

            return get_embeddings_from_model_threedinfomax
        elif name.startswith("custom:"):
            from molDistill.utils.embeddings import get_embeddings_from_distill_model

            return get_embeddings_from_distill_model
        else:
            from .transformers_models import get_embeddings_from_transformers

            return get_embeddings_from_transformers
